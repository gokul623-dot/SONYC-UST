"""
soundsense_dataset.py

Step 5: PyTorch Dataset and DataLoader for the SoundSense ML Project.

Implements the SoundSenseDataset class, which loads SONYC-UST 10-second urban audio
recordings on-the-fly and converts them dynamically into log-mel spectrograms
for multi-label classification.
"""

import os
import sys
import numpy as np
import pandas as pd

# Check for required audio and ML dependencies
missing_deps = []
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    missing_deps.append("torch")

try:
    import librosa
except ImportError:
    missing_deps.append("librosa")

if missing_deps:
    print("=" * 70)
    print("SOUNDSENSE DATASET - DEPENDENCY CHECK")
    print("=" * 70)
    print(f"Error: Missing required package(s): {', '.join(missing_deps)}")
    print("\nPlease install the missing dependencies:")
    print(f"    pip install {' '.join(missing_deps)}")
    print("=" * 70)
    sys.exit(1)


# The 8 coarse label categories in strict taxonomy order
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


class SoundSenseDataset(Dataset):
    """
    PyTorch Dataset for SONYC-UST urban sound classification in SoundSense.
    
    Loads audio dynamically and computes log-mel spectrograms on-the-fly to
    avoid excessive disk usage and RAM consumption.
    """

    def __init__(
        self,
        metadata_path=os.path.join("soundsense_data", "metadata.csv"),
        audio_root=".",
        split="train",
        sample_rate=48000,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    ):
        super().__init__()
        
        valid_splits = ["train", "validate", "test"]
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}.")

        self.metadata_path = metadata_path
        self.audio_root = audio_root
        self.split = split
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        # 1. Load metadata and filter by split
        all_df = pd.read_csv(self.metadata_path)
        self.df = all_df[all_df["split"] == self.split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No records found for split '{self.split}' in {self.metadata_path}")

        # 2. Build index of audio file locations across audio-0 to audio-18
        # This provides instant O(1) path lookups without repeated disk checks
        self.audio_file_map = self._build_audio_index()

        # 3. Pre-extract targets as a contiguous NumPy array for fast indexing
        self.targets = self.df[COARSE_LABELS].values.astype(np.float32)

    def _build_audio_index(self):
        """Map each audio filename to its absolute/relative path on disk."""
        file_map = {}
        for i in range(19):
            folder = os.path.join(self.audio_root, f"audio-{i}")
            if os.path.isdir(folder):
                for fname in os.listdir(folder):
                    if fname.endswith(".wav"):
                        file_map[fname] = os.path.join(folder, fname)
        return file_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["audio_filename"]

        wav_path = self.audio_file_map.get(filename)
        if wav_path is None or not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file '{filename}' could not be located in audio subdirectories.")

        # 1. Load audio with librosa at 48,000 Hz (mono)
        y, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)

        # 2. Ensure exactly 10 seconds (480,000 samples)
        expected_samples = self.sample_rate * 10
        if len(y) < expected_samples:
            y = np.pad(y, (0, expected_samples - len(y)), mode="constant")
        elif len(y) > expected_samples:
            y = y[:expected_samples]

        # 3. Compute Mel-scaled spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # 4. Convert power to decibels (log-mel)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # 5. Convert to PyTorch tensor of shape (1, 128, 938)
        spectrogram_tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0)

        # 6. Extract 8-element target tensor in taxonomy order
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)

        return spectrogram_tensor, target_tensor


def get_soundsense_dataloaders(
    metadata_path=os.path.join("soundsense_data", "metadata.csv"),
    audio_root=".",
    batch_size=16,
    num_workers=0
):
    """
    Constructs train, validation, and test PyTorch DataLoaders.
    
    Batch size = 16 for all splits.
    shuffle = True for training; False for validation and test.
    drop_last = False to retain all samples including partial batches.
    """
    train_dataset = SoundSenseDataset(metadata_path, audio_root, split="train")
    val_dataset = SoundSenseDataset(metadata_path, audio_root, split="validate")
    test_dataset = SoundSenseDataset(metadata_path, audio_root, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def test_dataset_and_dataloader():
    """Verify the SoundSenseDataset and DataLoader implementation."""
    print("=" * 75)
    print("SOUNDSENSE PYTORCH DATASET & DATALOADER VERIFICATION")
    print("=" * 75)

    print("\n1. Instantiating Dataset splits...")
    train_ds = SoundSenseDataset(split="train")
    val_ds = SoundSenseDataset(split="validate")
    test_ds = SoundSenseDataset(split="test")

    print(f"  - Training samples   : {len(train_ds):,}")
    print(f"  - Validation samples : {len(val_ds):,}")
    print(f"  - Test samples       : {len(test_ds):,}")
    print(f"  - Total samples      : {len(train_ds) + len(val_ds) + len(test_ds):,}")

    print("\n2. Creating DataLoaders (batch_size=16)...")
    train_loader, val_loader, test_loader = get_soundsense_dataloaders(batch_size=16)

    print(f"  - Train batches : {len(train_loader)} (13538 / 16 = 846 batches, last batch has 2 samples)")
    print(f"  - Val batches   : {len(val_loader)} (4308 / 16 = 270 batches, last batch has 4 samples)")
    print(f"  - Test batches  : {len(test_loader)} (664 / 16 = 42 batches, last batch has 8 samples)")

    print("\n3. Loading exactly ONE batch from Training DataLoader...")
    batch_spectrograms, batch_targets = next(iter(train_loader))

    print(f"  - Batch Spectrogram Shape : {list(batch_spectrograms.shape)}")
    print(f"  - Batch Target Shape      : {list(batch_targets.shape)}")
    print(f"  - Spectrogram Data Type   : {batch_spectrograms.dtype}")
    print(f"  - Target Data Type        : {batch_targets.dtype}")
    print(f"  - Spectrogram Min Value   : {batch_spectrograms.min().item():.2f} dB")
    print(f"  - Spectrogram Max Value   : {batch_spectrograms.max().item():.2f} dB")

    # Inspect first sample in the batch
    first_sample_targets = batch_targets[0].numpy()
    active_labels = [label for label, val in zip(COARSE_LABELS, first_sample_targets) if val == 1.0]
    active_str = ", ".join(active_labels) if active_labels else "None (Ambient Noise)"

    print("\n4. First Sample in Batch Inspection:")
    print(f"  - Spectrogram shape : {list(batch_spectrograms[0].shape)}")
    print(f"  - Target vector     : {first_sample_targets.tolist()}")
    print(f"  - Active labels     : [{active_str}]")

    # Shape verification assertion
    expected_spec_shape = [16, 1, 128, 938]
    expected_target_shape = [16, 8]

    assert list(batch_spectrograms.shape) == expected_spec_shape, (
        f"Shape mismatch: expected {expected_spec_shape}, got {list(batch_spectrograms.shape)}"
    )
    assert list(batch_targets.shape) == expected_target_shape, (
        f"Shape mismatch: expected {expected_target_shape}, got {list(batch_targets.shape)}"
    )

    print("\n" + "=" * 75)
    print("VERIFICATION SUCCESS: All batch shapes, dimensions, and data types match specifications!")
    print("=" * 75)


if __name__ == "__main__":
    test_dataset_and_dataloader()
