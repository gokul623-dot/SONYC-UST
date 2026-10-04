"""
evaluate_thresholds.py

Reusable evaluation script for SoundSense.

Evaluates the trained SoundSense model using optimized per-class thresholds:
1. Loads soundsense_data/models/weighted_best_model.pth
2. Loads the validation dataset (soundsense_data/features/validate/)
3. Loads optimized decision thresholds from soundsense_data/optimized_thresholds.json
4. Runs inference with torch.no_grad()
5. Computes and displays per-class and global metrics (Macro F1, Micro F1)
   comparing default threshold (0.5) against optimized per-class thresholds.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from soundsense_model import SoundSenseCNN
from soundsense_cached_dataset import SoundSenseCachedDataset

# Exact 8 coarse classes in DCASE taxonomy order
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

DEFAULT_CHECKPOINT = os.path.join("soundsense_data", "models", "weighted_best_model.pth")
DEFAULT_THRESHOLDS = os.path.join("soundsense_data", "optimized_thresholds.json")


def compute_metrics(y_true, y_pred):
    """Computes precision, recall, per-class F1, macro F1, and micro F1."""
    per_class_p = {}
    per_class_r = {}
    per_class_f1 = {}

    for i, cname in enumerate(CLASS_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        per_class_p[cname] = p
        per_class_r[cname] = r
        per_class_f1[cname] = f1

    macro_f1 = float(np.mean(list(per_class_f1.values())))

    total_tp = np.sum((y_true == 1) & (y_pred == 1))
    total_fp = np.sum((y_true == 0) & (y_pred == 1))
    total_fn = np.sum((y_true == 1) & (y_pred == 0))

    micro_p = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    micro_r = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    return per_class_p, per_class_r, per_class_f1, macro_f1, micro_f1


def evaluate_model(
    checkpoint_path=DEFAULT_CHECKPOINT,
    thresholds_path=DEFAULT_THRESHOLDS,
    split="validate",
    batch_size=32
):
    print("=" * 80)
    print("SOUNDSENSE THRESHOLD EVALUATION PIPELINE")
    print("=" * 80)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # 2. Load Checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"\n1. Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = checkpoint.get("epoch", "N/A")
    print(f"   Model training epoch: {epoch}")

    model = SoundSenseCNN(num_classes=8).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Load Thresholds
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds file not found at: {thresholds_path}")

    with open(thresholds_path, "r") as f:
        thresholds_dict = json.load(f)

    threshold_vector = np.array([thresholds_dict[cname] for cname in CLASS_NAMES])
    print(f"\n2. Loaded Optimized Thresholds from: {thresholds_path}")
    for cname in CLASS_NAMES:
        print(f"   - {cname:22s} : {thresholds_dict[cname]:.2f}")

    # 4. Load Dataset
    print(f"\n3. Loading {split} dataset...")
    dataset = SoundSenseCachedDataset(
        metadata_path="soundsense_data/metadata.csv",
        features_root="soundsense_data/features",
        split=split
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    print(f"   Total {split} samples: {len(dataset):,}")

    # 5. Inference
    print(f"\n4. Running inference...")
    t_start = time.time()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for specs, targets in loader:
            specs = specs.to(device, non_blocking=torch.cuda.is_available())
            logits = model(specs)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())

    elapsed = time.time() - t_start
    print(f"   Inference finished in {elapsed:.2f}s ({len(dataset)/elapsed:.1f} samples/sec)")

    logits_matrix = np.vstack(all_logits)
    targets_matrix = np.vstack(all_targets)
    probs_matrix = 1.0 / (1.0 + np.exp(-logits_matrix))

    # 6. Apply Default (0.5) and Tuned Thresholds
    y_pred_default = (probs_matrix >= 0.5).astype(int)
    y_pred_tuned = (probs_matrix >= threshold_vector).astype(int)

    def_p, def_r, def_f1, def_macro, def_micro = compute_metrics(targets_matrix, y_pred_default)
    tuned_p, tuned_r, tuned_f1, tuned_macro, tuned_micro = compute_metrics(targets_matrix, y_pred_tuned)

    # 7. Print Comparative Report
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS ({split.upper()} SET - {len(dataset):,} SAMPLES)")
    print("=" * 80)
    print(f"{'Class Name':22s} | {'Default F1 (0.5)':16s} | {'Tuned Threshold':16s} | {'Tuned F1':10s} | {'Diff':8s}")
    print("-" * 80)
    for cname in CLASS_NAMES:
        t_val = thresholds_dict[cname]
        d_f1 = def_f1[cname]
        t_f1 = tuned_f1[cname]
        diff = t_f1 - d_f1
        print(f"{cname:22s} | {d_f1:16.4f} | {t_val:16.2f} | {t_f1:10.4f} | {diff:+8.4f}")
    print("-" * 80)

    macro_abs_diff = tuned_macro - def_macro
    macro_rel_diff = (macro_abs_diff / def_macro) * 100
    micro_abs_diff = tuned_micro - def_micro
    micro_rel_diff = (micro_abs_diff / def_micro) * 100

    print(f"\n{'Metric':28s} | {'Default (0.5)':15s} | {'Tuned Thresholds':18s} | {'Difference':18s}")
    print("-" * 85)
    print(f"{'Validation Macro F1':28s} | {def_macro:15.4f} | {tuned_macro:18.4f} | {macro_abs_diff:+18.4f} ({macro_rel_diff:+.1f}%)")
    print(f"{'Validation Micro F1':28s} | {def_micro:15.4f} | {tuned_micro:18.4f} | {micro_abs_diff:+18.4f} ({micro_rel_diff:+.1f}%)")
    print("-" * 85)

    return {
        "default_macro_f1": def_macro,
        "tuned_macro_f1": tuned_macro,
        "default_micro_f1": def_micro,
        "tuned_micro_f1": tuned_micro,
        "per_class_tuned_f1": tuned_f1
    }


if __name__ == "__main__":
    evaluate_model()
