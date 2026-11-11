"""Dataset, transform, loader, inference, and submission utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#ajout pour CLIP
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class NpzImageDataset(Dataset):
    """Read the public NPZ archive format."""

    def __init__(self, path: str | Path, transform=None, require_labels: bool = False):
        with np.load(path) as archive:
            if "images" not in archive:
                raise ValueError(f"{path} has no 'images' array.")
            self.images = archive["images"].astype(np.uint8)
            self.ids = archive["ids"].astype(np.int64) if "ids" in archive else np.arange(len(self.images))
            self.labels = archive["labels"].astype(np.int64) if "labels" in archive else None
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError(f"Expected N x H x W x 3 uint8 images, got {self.images.shape}.")
        if require_labels and self.labels is None:
            raise ValueError(f"{path} has no labels; use it for inference only.")
        if self.labels is not None and len(self.labels) != len(self.images):
            raise ValueError("The image and label counts differ.")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index], mode="RGB")
        if self.transform:
            image = self.transform(image)
        if self.labels is None:
            return image, int(self.ids[index])
        return image, int(self.labels[index])


def make_transforms(image_size: int = 32, pretrained: bool = False, use_clip: bool = False):
    if use_clip:
        mean, std = CLIP_MEAN, CLIP_STD
    elif pretrained:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD

    if image_size == 32:
        train_transform = T.Compose(
            [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)]
        )
        evaluation_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    else:
        train_ops = [
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
        ]
        if use_clip:
            train_ops.append(T.RandAugment(num_ops=2, magnitude=9))
        train_ops.extend([T.ToTensor(), T.Normalize(mean, std)])
        train_transform = T.Compose(train_ops)
        evaluation_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    return train_transform, evaluation_transform


def make_loaders(input_dir: str | Path, image_size=32, pretrained=False, batch_size=128, workers=0, use_clip=False):
    input_dir = Path(input_dir)
    train_transform, evaluation_transform = make_transforms(image_size, pretrained, use_clip=use_clip)
    train_set = NpzImageDataset(input_dir / "train.npz", train_transform, require_labels=True)
    validation_set = NpzImageDataset(
        input_dir / "validation.npz", evaluation_transform, require_labels=True
    )
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, **options),
        DataLoader(validation_set, shuffle=False, **options),
    )


@torch.inference_mode()
def predict(model, test_archive: str | Path, image_size=32, pretrained=False, batch_size=256, workers=0, device=None, use_clip=False):
    """Return an id/label DataFrame for the unlabeled competition test archive."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, transform = make_transforms(image_size, pretrained, use_clip=use_clip)
    dataset = NpzImageDataset(test_archive, transform=transform, require_labels=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model.to(device).eval()
    ids, labels = [], []
    for images, batch_ids in loader:
        labels.extend(model(images.to(device)).argmax(1).cpu().tolist())
        ids.extend(batch_ids.tolist())
    return pd.DataFrame({"id": ids, "label": labels})


@torch.inference_mode()
def clip_zero_shot(
    test_archive: str | Path,
    output_path: str | Path,
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    batch_size=128,
):
    """Enhanced multi-prompt OpenCLIP zero-shot baseline for synthetic domain."""
    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    dataset = NpzImageDataset(test_archive, transform=preprocess, require_labels=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    templates = [
        "a photo of a {}.",
        "a synthetic image of a {}.",
        "a 3d rendering of a {}.",
        "a digital illustration of a {}.",
        "a low resolution photo of a {}.",
    ]
    
    model.to(device).eval()
    text_features_list = []
    for name in CLASS_NAMES:
        prompts = tokenizer([template.format(name) for template in templates]).to(device)
        class_embeddings = model.encode_text(prompts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features_list.append(class_embedding)
    
    text_features = torch.stack(text_features_list, dim=0).to(device)

    ids, labels = [], []
    for images, batch_ids in loader:
        image_features = model.encode_image(images.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        labels.extend((100 * image_features @ text_features.T).argmax(1).cpu().tolist())
        ids.extend(batch_ids.tolist())
    submission = pd.DataFrame({"id": ids, "label": labels})
    submission.to_csv(output_path, index=False)
    return submission