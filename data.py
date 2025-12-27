from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from config import DataConfig


def _to_tensor(image: Image.Image, img_size: int) -> torch.Tensor:
    resample = getattr(Image, "Resampling", Image).BILINEAR
    image = image.convert("L").resize((img_size, img_size), resample)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
    return tensor


class BirdDroneFolder(Dataset):
    def __init__(self, root: Path, img_size: int):
        self.root = root
        self.img_size = img_size
        self.classes = ["bird", "drone"]
        self.samples: list[tuple[Path, int]] = []
        for label, cls in enumerate(self.classes):
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            for path in cls_dir.rglob("*"):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {root}. Expected {root}/bird and {root}/drone."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            tensor = _to_tensor(img, self.img_size)
        return tensor, torch.tensor(label, dtype=torch.long)


class SyntheticBirdDrone(Dataset):
    """Simple synthetic patterns to make the pipeline runnable without a dataset."""

    def __init__(self, num_samples: int, img_size: int, seed: int = 0):
        self.num_samples = num_samples
        self.img_size = img_size
        rng = np.random.default_rng(seed)
        self.images = rng.normal(loc=0.0, scale=0.2, size=(num_samples, img_size, img_size)).astype(
            np.float32
        )
        self.labels = rng.integers(low=0, high=2, size=(num_samples,))
        # carve simple shapes for a tiny bit of structure
        for i in range(num_samples):
            label = self.labels[i]
            if label == 0:
                # bird-ish: centered gaussian blob
                cx, cy = img_size // 2, img_size // 2
                xv, yv = np.meshgrid(np.arange(img_size), np.arange(img_size))
                blob = np.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2 * (img_size / 6) ** 2))
                self.images[i] += blob * 0.5
            else:
                # drone-ish: four bright corners
                patch = img_size // 5
                self.images[i][:patch, :patch] += 0.5
                self.images[i][:patch, -patch:] += 0.5
                self.images[i][-patch:, :patch] += 0.5
                self.images[i][-patch:, -patch:] += 0.5
        self.images = np.clip(self.images, 0.0, 1.0)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return img, label


def build_datasets(cfg: DataConfig, seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    root = Path(cfg.data_root)
    if not cfg.use_synthetic and root.exists():
        train_dir = root / "train"
        val_dir = root / "val"
        test_dir = root / "test"
        if not (train_dir.exists() and val_dir.exists() and test_dir.exists()):
            raise FileNotFoundError(
                f"Expected train/val/test subfolders with bird/drone classes under {root}"
            )
        train_ds = BirdDroneFolder(train_dir, cfg.img_size)
        val_ds = BirdDroneFolder(val_dir, cfg.img_size)
        test_ds = BirdDroneFolder(test_dir, cfg.img_size)
        return train_ds, val_ds, test_ds

    # synthetic fallback
    train_count = cfg.synthetic_samples
    val_count = max(1, int(cfg.synthetic_samples * cfg.val_split))
    test_count = max(1, cfg.synthetic_samples - train_count - val_count)
    generator = torch.Generator().manual_seed(seed)
    full = SyntheticBirdDrone(
        num_samples=train_count + val_count + test_count,
        img_size=cfg.img_size,
        seed=seed,
    )
    train_len = int(len(full) * cfg.train_split)
    val_len = int(len(full) * cfg.val_split)
    test_len = len(full) - train_len - val_len
    train_ds, val_ds, test_ds = random_split(
        full, [train_len, val_len, test_len], generator=generator
    )
    return train_ds, val_ds, test_ds


def build_dataloaders(cfg: DataConfig, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = build_datasets(cfg, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
