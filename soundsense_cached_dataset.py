"""
soundsense_cached_dataset.py

High-performance PyTorch Dataset for SoundSense that loads precomputed log-mel
spectrogram features directly from .npy cache files on disk.

Preserves the exact tensor shapes, dtypes, and label ordering of SoundSenseDataset
while eliminating the CPU waveform decoding and STFT computation during training.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# The exact 8 coarse classes in DCASE taxonomy order
COARSE_LABELS = [
    "engine",
    "machinery_impact",
    "non_machinery_impact",
    "powered_saw",
    "alert_signal",
    "music",
    "human_voice",
    "dog"
]


class SoundSenseCachedDataset(Dataset):
    """
    Loads precomputed log-mel spectrogram features (.npy) from disk.
    
    Output shapes per item:
      - spectrogram : torch.Tensor of shape [1, 128, 938] (float32)
      - target      : torch.Tensor of shape [8] (float32)
    """

    def __init__(
        self,
        metadata_path=os.path.join("soundsense_data", "metadata.csv"),
        features_root=os.path.join("soundsense_data", "features"),
        split="train"
    ):
        super().__init__()

        valid_splits = ["train", "validate", "test"]
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}.")

        self.metadata_path = metadata_path
        self.features_root = features_root
        self.split = split
        self.split_feature_dir = os.path.join(self.features_root, self.split)

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        # 1. Load metadata and filter by split
        all_df = pd.read_csv(self.metadata_path)
        self.df = all_df[all_df["split"] == self.split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No records found for split '{self.split}' in {self.metadata_path}")

        # 2. Extract targets as contiguous numpy float32 array
        self.targets = self.df[COARSE_LABELS].values.astype(np.float32)

        # 3. Pre-build list of feature paths for fast indexing
        self.feature_paths = [
            os.path.join(self.split_feature_dir, os.path.splitext(fname)[0] + ".npy")
            for fname in self.df["audio_filename"]
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feat_path = self.feature_paths[idx]

        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Cached feature not found at: {feat_path}. "
                f"Please ensure features are precomputed for split '{self.split}'."
            )

        # Load precomputed (128, 938) float32 array
        # mmap_mode=None loads cleanly into memory
        feature = np.load(feat_path)

        # Convert to PyTorch tensor with channel dimension -> [1, 128, 938]
        spectrogram_tensor = torch.from_numpy(feature).float().unsqueeze(0)

        # Target tensor -> [8]
        target_tensor = torch.from_numpy(self.targets[idx]).float()

        return spectrogram_tensor, target_tensor


def get_cached_dataloaders(
    metadata_path=os.path.join("soundsense_data", "metadata.csv"),
    features_root=os.path.join("soundsense_data", "features"),
    batch_size=16,
    num_workers=0
):
    """
    Constructs high-speed train, validation, and test PyTorch DataLoaders
    backed by precomputed .npy features.
    """
    use_pin_memory = torch.cuda.is_available()

    train_ds = SoundSenseCachedDataset(metadata_path, features_root, split="train")
    val_ds = SoundSenseCachedDataset(metadata_path, features_root, split="validate")
    test_ds = SoundSenseCachedDataset(metadata_path, features_root, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
