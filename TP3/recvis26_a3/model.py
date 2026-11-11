"""Model definitions and training utilities."""

from __future__ import annotations

import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from data import CLASS_NAMES


class OpenCLIPClassifier(nn.Module):
    """OpenCLIP visual backbone with a cosine-scaled linear head."""

    def __init__(
        self,
        arch: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        num_classes: int = len(CLASS_NAMES),
        init_head_with_zero_shot: bool = True,
    ):
        super().__init__()
        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.visual = clip_model.visual

        # Infer visual feature dimension automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.visual(dummy).shape[-1]

        self.fc = nn.Linear(feat_dim, num_classes, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(100.0)))

        if init_head_with_zero_shot:
            self._init_zero_shot_weights(clip_model, arch)

        # Keep only the visual tower + classification head in this module.
        del clip_model

    @torch.no_grad()
    def _init_zero_shot_weights(self, clip_model, arch: str) -> None:
        """Initialize the linear classification head with multi-prompt text embeddings."""
        import open_clip

        tokenizer = open_clip.get_tokenizer(arch)
        templates = [
            "a photo of a {}.",
            "a synthetic image of a {}.",
            "a 3d rendering of a {}.",
            "a digital illustration of a {}.",
            "a low resolution photo of a {}.",
        ]
        device = next(clip_model.parameters()).device
        clip_model.eval()
        text_features_list = []
        for name in CLASS_NAMES:
            prompts = tokenizer([t.format(name) for t in templates]).to(device)
            embeddings = clip_model.encode_text(prompts)
            embeddings = F.normalize(embeddings.float(), dim=-1)
            mean_embedding = F.normalize(embeddings.mean(dim=0), dim=-1)
            text_features_list.append(mean_embedding.cpu())

        text_features = torch.stack(text_features_list, dim=0)
        if text_features.shape != self.fc.weight.shape:
            raise ValueError(
                f"Zero-shot text features shape {tuple(text_features.shape)} "
                f"does not match classifier weight shape {tuple(self.fc.weight.shape)}."
            )
        self.fc.weight.copy_(text_features.to(self.fc.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.visual(x)
        features = F.normalize(features, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * self.fc(features)


def build_model(name="resnet18", pretrained=False, image_size=32):
    """Build one of the supported supervised or OpenCLIP models."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        if image_size == 32 and not pretrained:
            model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        return model
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        return model
    if name == "vit_b_16":
        if image_size != 224:
            raise ValueError("The ViT-B/16 baseline requires 224px inputs.")
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASS_NAMES))
        return model

    # OpenCLIP models (require 224px inputs)
    openclip_configs = {
        "openclip_vit_b32": ("ViT-B-32", "laion2b_s34b_b79k"),
        "openclip_vit_b16": ("ViT-B-16", "laion2b_s34b_b88k"),
        "openclip_vit_l14": ("ViT-L-14", "laion2b_s32b_b82k"),
    }
    if name in openclip_configs:
        if image_size != 224:
            raise ValueError(f"{name} requires image_size=224.")
        arch, ckpt = openclip_configs[name]
        return OpenCLIPClassifier(arch=arch, pretrained=ckpt, init_head_with_zero_shot=True)

    raise ValueError(f"Unknown model: {name}")


def freeze_backbone(model) -> None:
    """Freeze all parameters except the classification head."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.fc if hasattr(model, "fc") else model.heads
    for parameter in head.parameters():
        parameter.requires_grad = True
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True


def unfreeze_last_blocks(model, num_blocks: int = 2) -> None:
    """Unfreeze the last N transformer blocks of an OpenCLIP visual backbone for fine-tuning."""
    freeze_backbone(model)
    if hasattr(model, "visual") and hasattr(model.visual, "transformer"):
        blocks = model.visual.transformer.resblocks
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(model.visual, "ln_post"):
            for param in model.visual.ln_post.parameters():
                param.requires_grad = True
        if hasattr(model.visual, "proj") and isinstance(model.visual.proj, nn.Parameter):
            model.visual.proj.requires_grad = True


def run_epoch(model, loader, optimizer, device, scaler=None, desc="epoch", label_smoothing=0.1):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing if training else 0.0)
    total_loss = total_correct = total = 0
    amp = device.type == "cuda"
    epoch_start = time.perf_counter()
    progress = tqdm(loader, desc=desc, leave=False)
    for images, labels in progress:
        iter_start = time.perf_counter()
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = len(labels)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += batch_size
        iter_time = time.perf_counter() - iter_start
        progress.set_postfix(
            loss=f"{total_loss / total:.4f}",
            acc=f"{total_correct / total:.4f}",
            iter_s=f"{iter_time:.3f}",
        )
    epoch_time = time.perf_counter() - epoch_start
    n_iters = max(len(loader), 1)
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "epoch_time_sec": epoch_time,
        "avg_iter_time_sec": epoch_time / n_iters,
    }


def train(model, train_loader, validation_loader, epochs=10, lr=3e-4, weight_decay=1e-4, device=None):
    """Train a model, restore its best validation weights, and return metrics."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    best_accuracy, best_val_loss, best_state = -1.0, float("inf"), None
    best_epoch = 0
    history = []
    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, scaler, desc=f"train {epoch}/{epochs}"
        )
        scheduler.step()
        validation_metrics = run_epoch(
            model, validation_loader, None, device, desc=f"val {epoch}/{epochs}"
        )
        epoch_time = train_metrics["epoch_time_sec"] + validation_metrics["epoch_time_sec"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "train_epoch_time_sec": train_metrics["epoch_time_sec"],
            "val_epoch_time_sec": validation_metrics["epoch_time_sec"],
            "epoch_time_sec": epoch_time,
            "train_avg_iter_time_sec": train_metrics["avg_iter_time_sec"],
            "val_avg_iter_time_sec": validation_metrics["avg_iter_time_sec"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}: train_acc={row['train_accuracy']:.4f}, "
            f"val_acc={row['val_accuracy']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, "
            f"epoch_time={epoch_time:.2f}s "
            f"(train_iter={row['train_avg_iter_time_sec']:.3f}s, "
            f"val_iter={row['val_avg_iter_time_sec']:.3f}s)"
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if validation_metrics["loss"] < best_val_loss:
            best_val_loss = validation_metrics["loss"]
    total_training_time_sec = time.perf_counter() - training_start
    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    history_df.attrs["training_metrics"] = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "best_val_loss": best_val_loss,
        "total_training_time_sec": total_training_time_sec,
        "mean_epoch_time_sec": float(history_df["epoch_time_sec"].mean()),
        "mean_train_iter_time_sec": float(history_df["train_avg_iter_time_sec"].mean()),
        "mean_val_iter_time_sec": float(history_df["val_avg_iter_time_sec"].mean()),
    }
    print(
        f"done in {total_training_time_sec:.2f}s | "
        f"best val_acc={best_accuracy:.4f} (epoch {best_epoch}) | "
        f"best val_loss={best_val_loss:.4f}"
    )
    return history_df