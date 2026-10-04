"""
soundsense_model.py

Step 6: Custom 2D Convolutional Neural Network (CNN) for SoundSense.

Implements SoundSenseCNN, a lightweight, explainable 2D CNN architecture designed
for multi-label urban sound tagging from log-mel spectrograms.

Input:  [batch_size, 1, 128, 938]
Output: [batch_size, 8] (unnormalized raw logits for BCEWithLogitsLoss)
"""

import os
import sys
import torch
import torch.nn as nn

# The 8 target sound categories in exact taxonomy order
CLASS_NAMES = [
    "engine",
    "machinery_impact",
    "non_machinery_impact",
    "powered_saw",
    "alert_signal",
    "music",
    "human_voice",
    "dog"
]


class SoundSenseCNN(nn.Module):
    """
    Lightweight, explainable 2D CNN for SoundSense urban sound tagging.
    
    Architecture:
      - Block 1: Conv2d(1 -> 16) + BatchNorm2d + ReLU + MaxPool2d(2, 2)
      - Block 2: Conv2d(16 -> 32) + BatchNorm2d + ReLU + MaxPool2d(2, 2)
      - Block 3: Conv2d(32 -> 64) + BatchNorm2d + ReLU + MaxPool2d(2, 2)
      - Global Pooling: AdaptiveAvgPool2d((1, 1))
      - Classifier: Linear(64 -> 8)
    
    Note: No Sigmoid activation is applied in forward(); outputs are raw logits
    suitable for numerical stability with nn.BCEWithLogitsLoss.
    """

    def __init__(self, num_classes=8):
        super(SoundSenseCNN, self).__init__()
        self.num_classes = num_classes

        # Convolutional Block 1: Extracts low-level spectro-temporal features (edges, onsets)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Convolutional Block 2: Extracts mid-level acoustic textures (harmonics, formants)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Convolutional Block 3: Extracts high-level class-specific acoustic patterns (siren sweeps, engine rumbles)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global Average Pooling: Aggregates spatial features across the entire 10-second clip
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Classifier: Projects 64 feature maps to 8 class logits
        self.classifier = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Log-mel spectrogram tensor of shape [batch_size, 1, 128, 938]
            
        Returns:
            torch.Tensor: Raw logits of shape [batch_size, 8]
        """
        # Feature extraction
        x = self.block1(x)         # -> [batch_size, 16, 64, 469]
        x = self.block2(x)         # -> [batch_size, 32, 32, 234]
        x = self.block3(x)         # -> [batch_size, 64, 16, 117]

        # Spatial aggregation
        x = self.global_pool(x)    # -> [batch_size, 64, 1, 1]
        x = torch.flatten(x, 1)    # -> [batch_size, 64]

        # Classification (raw logits, no Sigmoid)
        logits = self.classifier(x) # -> [batch_size, 8]
        return logits


def test_soundsense_model():
    """Verifies model instantiation, parameter count, dummy forward pass, and real DataLoader pass."""
    print("=" * 75)
    print("SOUNDSENSE CNN MODEL ARCHITECTURE & FORWARD PASS VERIFICATION")
    print("=" * 75)

    # 1. Instantiate model
    model = SoundSenseCNN(num_classes=8)
    print("\n1. Model Architecture:")
    print(model)

    # 2. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  - Total Parameters     : {total_params:,}")
    print(f"  - Trainable Parameters : {trainable_params:,}")

    # 3. Test dummy forward pass
    print("\n2. Dummy Forward Pass Test:")
    batch_size = 16
    dummy_input = torch.randn(batch_size, 1, 128, 938)
    print(f"  - Dummy Input Shape   : {list(dummy_input.shape)}")

    model.eval()
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"  - Dummy Output Shape  : {list(dummy_output.shape)}")
    print(f"  - Output Data Type    : {dummy_output.dtype}")
    print(f"  - Expected Output     : [{batch_size}, 8]")

    assert list(dummy_output.shape) == [batch_size, 8], (
        f"Dummy output shape mismatch: expected [{batch_size}, 8], got {list(dummy_output.shape)}"
    )
    print("  - Dummy Forward Pass  : PASSED")

    # 4. Real Forward Pass using ONE batch from existing DataLoader
    print("\n3. Real DataLoader Forward Pass Test (No backprop, no weight update):")
    try:
        from soundsense_dataset import get_soundsense_dataloaders
        train_loader, _, _ = get_soundsense_dataloaders(batch_size=16)
        
        # Load exactly one real batch
        real_specs, real_targets = next(iter(train_loader))
        print(f"  - Real Batch Spec Shape   : {list(real_specs.shape)}")
        print(f"  - Real Batch Target Shape : {list(real_targets.shape)}")

        with torch.no_grad():
            real_logits = model(real_specs)

        print(f"  - Real Output Logits Shape: {list(real_logits.shape)}")
        assert list(real_logits.shape) == [16, 8], (
            f"Real logits shape mismatch: expected [16, 8], got {list(real_logits.shape)}"
        )
        print("  - Real Forward Pass       : PASSED")

        # Inspect raw logits for the first sample in the real batch
        first_sample_logits = real_logits[0].tolist()
        print(f"\n4. Inspection of First Real Sample Output (Raw Logits):")
        for cls_name, logit in zip(CLASS_NAMES, first_sample_logits):
            print(f"  - {cls_name:22s} : logit = {logit:+.4f}")

    except Exception as e:
        print(f"  - Error during real DataLoader test: {e}")
        raise e

    print("\n" + "=" * 75)
    print("VERIFICATION SUCCESS: SoundSenseCNN model is ready for ML training!")
    print("=" * 75)


if __name__ == "__main__":
    test_soundsense_model()
