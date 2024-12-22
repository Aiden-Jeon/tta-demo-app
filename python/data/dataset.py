from pathlib import Path
from typing import List

import torch
import numpy as np
from torch.utils.data import Dataset
from robustbench.data import load_cifar10c, load_cifar100c

CORRUPTION_DOMAINS = [
    "shot_noise",
    "motion_blur",
    "snow",
    "pixelate",
    "gaussian_noise",
    "defocus_blur",
    "brightness",
    "fog",
    "zoom_blur",
    "frost",
    "glass_blur",
    "impulse_noise",
    "contrast",
    "jpeg_compression",
    "elastic_transform",
]

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


class CifarDataset(Dataset):
    def __init__(
        self,
        data_dir: str = str(DEFAULT_DATA_DIR),
        corrupt_domain_orders: List[str] = CORRUPTION_DOMAINS,
        severity: int = 5,
        num_samples: int = 100,
    ):
        super().__init__()
        self.data_dir = data_dir
        assert 1 <= severity <= 5
        self.corrupt_domain_orders = corrupt_domain_orders
        self.severity = severity
        self.num_samples = num_samples
        self.X = None
        self.y_domain = None
        self.y_label = None
        self.prepare_dataset()

    def prepare_dataset(self):
        X = []
        y_label = []
        y_domain = []
        for corrupt in self.corrupt_domain_orders:
            x_, y_label_ = load_cifar10c(
                n_examples=self.num_samples,
                severity=self.severity,
                data_dir=self.data_dir,
                shuffle=False,
                corruptions=[corrupt],
            )
            y_domain_ = np.array([corrupt] * self.num_samples)

            X += [x_]
            y_label += [y_label_]
            y_domain += [y_domain_]
        self.X = torch.cat(X)
        self.y_label = torch.cat(y_label)
        self.y_domain = np.concatenate(y_domain)

    def __len__(self):
        return len(self.y_label)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        X_ = self.X[idx]
        y_label_ = self.y_label[idx]
        y_domain_ = self.y_domain[idx]
        return X_, y_label_, y_domain_
