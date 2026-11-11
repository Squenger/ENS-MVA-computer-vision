"""Train a baseline or OpenCLIP model and create a Kaggle submission."""

from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import CLASS_NAMES, make_loaders, predict
from model import build_model, freeze_backbone, train, unfreeze_last_blocks

SUPPORTED_MODELS = (
    "resnet18",
    "resnet50",
    "vit_b_16",
    "openclip_vit_b32",
    "openclip_vit_b16",
    "openclip_vit_l14",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("RECVIS_INPUT_DIR", "/kaggle/input/recvis-mva-a3-2026")),
        help="Directory containing train.npz, validation.npz, test.npz, and sample_submission.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("RECVIS_OUTPUT_DIR", ".")),
        help="Directory for the checkpoint, history, and submission.",
    )
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="openclip_vit_b16")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights.")
    parser.add_argument("--linear-probe", action="store_true", help="Train only the classifier head.")
    parser.add_argument("--unfreeze-blocks", type=int, default=0, help="Number of last transformer blocks to fine-tune after linear probing.")
    parser.add_argument("--ft-epochs", type=int, default=3, help="Epochs for the fine-tuning stage (if --unfreeze-blocks > 0).")
    parser.add_argument("--ft-lr", type=float, default=5e-5, help="Learning rate for the fine-tuning stage.")
    parser.add_argument("--wise-ft-alpha", type=float, default=0.5, help="WiSE-FT weight interpolation factor (1.0 = pure fine-tuned, 0.5 = 50/50 mix with zero-shot).")
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to 32, or 224 for pretrained/ViT/OpenCLIP.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--workers",
        type=int,
        default=0 if os.name == "nt" else 2,
        help="DataLoader workers. Default 0 on Windows to avoid Jupyter/spawn freezes.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_inputs(data_dir: Path) -> None:
    required = ("train.npz", "validation.npz", "test.npz", "sample_submission.csv")
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {missing}")
    with np.load(data_dir / "test.npz") as archive:
        if "labels" in archive.files:
            raise ValueError("test.npz must not contain labels.")


def main() -> None:
    args = parse_args()
    is_openclip = args.model.startswith("openclip_")
    if is_openclip:
        args.pretrained = True

    if args.linear_probe and not args.pretrained:
        raise ValueError("--linear-probe requires --pretrained.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    validate_inputs(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    image_size = args.image_size or (224 if args.pretrained or args.model == "vit_b_16" or is_openclip else 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model} image_size={image_size} data={args.data}")

    train_loader, validation_loader = make_loaders(
        args.data,
        image_size=image_size,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        workers=args.workers,
        use_clip=is_openclip,
    )
    model = build_model(args.model, pretrained=args.pretrained, image_size=image_size)

    # Save initial Zero-Shot state for WiSE-FT interpolation
    zero_shot_state = copy.deepcopy(model.state_dict()) if is_openclip else None

    if args.linear_probe or is_openclip:
        print("--- Stage 1: Linear Probing (frozen backbone) ---")
        freeze_backbone(model)

    history = train(
        model,
        train_loader,
        validation_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )
    stage_metrics = [dict(history.attrs.get("training_metrics", {}), stage="linear_probe")]

    if is_openclip and args.unfreeze_blocks > 0 and args.ft_epochs > 0:
        print(f"--- Stage 2: Fine-Tuning last {args.unfreeze_blocks} blocks ---")
        unfreeze_last_blocks(model, num_blocks=args.unfreeze_blocks)
        ft_history = train(
            model,
            train_loader,
            validation_loader,
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            weight_decay=1e-3,
            device=device,
        )
        stage_metrics.append(dict(ft_history.attrs.get("training_metrics", {}), stage="finetune"))
        history = pd.concat([history, ft_history], ignore_index=True)

    if is_openclip and zero_shot_state is not None and 0.0 < args.wise_ft_alpha < 1.0:
        print(f"--- Applying WiSE-FT interpolation (alpha={args.wise_ft_alpha}) ---")
        fine_tuned_state = model.state_dict()
        alpha = args.wise_ft_alpha
        mixed_state = {}
        for key, value in fine_tuned_state.items():
            if key in zero_shot_state and torch.is_floating_point(value):
                mixed_state[key] = (1.0 - alpha) * zero_shot_state[key].to(value.device, dtype=value.dtype) + alpha * value
            else:
                mixed_state[key] = value
        model.load_state_dict(mixed_state)

    history_path = args.output / "history.csv"
    history.to_csv(history_path, index=False)
    metrics_path = args.output / "training_metrics.csv"
    metrics_df = pd.DataFrame(stage_metrics)
    if len(metrics_df):
        metrics_df["total_training_time_sec_all_stages"] = metrics_df["total_training_time_sec"].sum()
        metrics_df["best_val_accuracy_all_stages"] = metrics_df["best_val_accuracy"].max()
        metrics_df["best_val_loss_all_stages"] = metrics_df["best_val_loss"].min()
    metrics_df.to_csv(metrics_path, index=False)

    checkpoint_path = args.output / "model.pth"
    torch.save(
        {
            "model_name": args.model,
            "pretrained": args.pretrained,
            "image_size": image_size,
            "class_names": CLASS_NAMES,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    submission = predict(
        model,
        args.data / "test.npz",
        image_size=image_size,
        pretrained=args.pretrained,
        batch_size=max(args.batch_size, 256),
        workers=args.workers,
        device=device,
        use_clip=is_openclip,
    )
    sample = pd.read_csv(args.data / "sample_submission.csv")
    if sample.columns.tolist() != ["id", "label"]:
        raise ValueError("sample_submission.csv must contain exactly the columns id,label.")
    if not np.array_equal(sample["id"].to_numpy(), submission["id"].to_numpy()):
        raise ValueError("Submission IDs are not aligned with sample_submission.csv.")
    if not submission["label"].between(0, 9).all():
        raise ValueError("Predicted labels must be integers from 0 to 9.")

    submission_path = args.output / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"saved {checkpoint_path}")
    print(f"saved {history_path}")
    print(f"saved {metrics_path}")
    print(f"saved {submission_path}")


if __name__ == "__main__":
    main() 