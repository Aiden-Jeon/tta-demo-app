from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset


class CifarDataset(Dataset):
    def __init__(
        self,
        dataset_root: str = "dataset/CIFAR-10-C/",
        severity_level: int = 0,
        num_samples: int = 1000,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        assert 0 <= severity_level <= 5
        self.severity_level = severity_level
        self.num_samples = num_samples
        self.X = None
        self.y_corrupt = None
        self.y_label = None
        self.prepare_dataset()

    def prepare_dataset(self):
        if self.severity_level == 0:
            dataset_folder = self.dataset_root / "origin"
            X = np.load(dataset_folder / "original.npy")
            y_corrupt = np.array(["normal"] * len(X))
            y_label = np.load(dataset_folder / "labels.npy")

            self.X = torch.from_numpy(X[: self.num_samples])
            self.y_corrupt = y_corrupt[: self.num_samples]
            self.y_label = torch.from_numpy(y_label[: self.num_samples])
        else:
            dataset_folder = (
                self.dataset_root / "corrupted" / f"severity-{self.severity_level}"
            )
            corrupt_filepaths = list(
                filter(lambda x: "label" not in str(x), dataset_folder.glob("*.npy"))
            )
            corrupt_num_samples = [self.num_samples // len(corrupt_filepaths)] * len(
                corrupt_filepaths
            )
            corrupt_num_samples[-1] += self.num_samples % len(corrupt_filepaths)

            X = []
            y_corrupt = []
            y_label = []
            y_label_ = np.load(dataset_folder / "labels.npy")
            for idx, corrupt_filepath in enumerate(corrupt_filepaths):
                sub_n_samples = corrupt_num_samples[idx]
                x_ = np.load(corrupt_filepath)
                y_corrupt_ = np.array(
                    [str(corrupt_filepath).replace(".npy", "")] * len(x_)
                )
                X += [torch.from_numpy(x_[:sub_n_samples])]
                y_corrupt += [y_corrupt_[:sub_n_samples]]
                y_label += [torch.from_numpy(y_label_[:sub_n_samples])]
            self.X = torch.cat(X)
            self.y_corrupt = np.concatenate(y_corrupt)
            self.y_label = torch.cat(y_label)

        # change NHWC to NCHW format
        self.X = self.X.permute(0, 3, 1, 2)
        # make it compatible with our models (normalize)
        self.X = self.X / 255.0

    def __len__(self):
        return len(self.y_label)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        X_ = self.X[idx]
        y_corrupt_ = self.y_corrupt[idx]
        y_label_ = self.y_label[idx]
        return X_, y_corrupt_, y_label_
