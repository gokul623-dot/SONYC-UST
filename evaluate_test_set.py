"""
evaluate_test_set.py

FINAL HELD-OUT TEST EVALUATION for SoundSense.

Evaluates the already-trained class-weighted CNN checkpoint:
    soundsense_data/models/weighted_best_model.pth (Epoch 5)
using the frozen validation-optimized decision thresholds from:
    soundsense_data/optimized_thresholds.json

Strict constraints:
- NO retraining.
- NO weight modification.
- NO threshold modification or re-tuning.
- Test set evaluated strictly once with torch.no_grad() and model.eval().
- Produces final_test_results.csv and final_test_summary.json.
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

# 8 coarse classes in exact DCASE taxonomy order
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

CHECKPOINT_PATH = os.path.join("soundsense_data", "models", "weighted_best_model.pth")
THRESHOLDS_PATH = os.path.join("soundsense_data", "optimized_thresholds.json")
METADATA_PATH = os.path.join("soundsense_data", "metadata.csv")
TEST_FEATURE_DIR = os.path.join("soundsense_data", "features", "test")
CSV_OUTPUT_PATH = os.path.join("soundsense_data", "final_test_results.csv")
JSON_OUTPUT_PATH = os.path.join("soundsense_data", "final_test_summary.json")

EXPECTED_TEST_COUNT = 664


def calculate_metrics_for_predictions(y_true, y_pred, thresholds_dict):
    """
    Computes comprehensive per-class and global metrics (Macro/Micro)
    along with confusion matrix components (TP, FP, FN, TN).
    """
    records = []
    per_class_metrics = {}

    for i, cname in enumerate(CLASS_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))

        actual_pos = int(np.sum(yt == 1))
        pred_pos = int(np.sum(yp == 1))

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        per_class_metrics[cname] = {
            "threshold": float(thresholds_dict[cname]),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "actual_positive_count": actual_pos,
            "predicted_positive_count": pred_pos
        }

        records.append({
            "class_name": cname,
            "threshold": round(float(thresholds_dict[cname]), 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "actual_positive_count": actual_pos,
            "predicted_positive_count": pred_pos
        })

    # Macro averages
    macro_p = float(np.mean([m["precision"] for m in per_class_metrics.values()]))
    macro_r = float(np.mean([m["recall"] for m in per_class_metrics.values()]))
    macro_f1 = float(np.mean([m["f1"] for m in per_class_metrics.values()]))

    # Micro aggregates
    total_tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    total_fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    total_fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total_tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    micro_p = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    micro_r = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    summary = {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "total_true_negatives": total_tn,
        "per_class": per_class_metrics,
        "records": records
    }

    return summary


def main():
    print("=" * 85)
    print("SOUNDSENSE: FINAL HELD-OUT TEST EVALUATION")
    print("=" * 85)

    # =========================================================================
    # STEP 1: VERIFY TEST SET & ENVIRONMENT
    # =========================================================================
    print("\n--- STEP 1: VERIFYING TEST SET & CONFIGURATION ---")
    if not os.path.exists(TEST_FEATURE_DIR):
        raise FileNotFoundError(f"Test feature directory missing: {TEST_FEATURE_DIR}")

    test_npy_files = [f for f in os.listdir(TEST_FEATURE_DIR) if f.endswith(".npy")]
    num_test_files = len(test_npy_files)
    print(f"Number of test feature files (.npy) : {num_test_files}")

    if num_test_files != EXPECTED_TEST_COUNT:
        raise ValueError(
            f"FATAL: Expected exactly {EXPECTED_TEST_COUNT} test recordings, "
            f"found {num_test_files}! STOPPING."
        )

    # Verify metadata split count
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file missing: {METADATA_PATH}")
    metadata_df = pd.read_csv(METADATA_PATH)
    test_meta_df = metadata_df[metadata_df["split"] == "test"]
    num_test_labels = len(test_meta_df)
    print(f"Number of test labels in metadata    : {num_test_labels}")
    if num_test_labels != EXPECTED_TEST_COUNT:
        raise ValueError(f"FATAL: Metadata test record count mismatch: {num_test_labels} vs {EXPECTED_TEST_COUNT}!")

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device                              : {device}")
    if torch.cuda.is_available():
        print(f"GPU device name                     : {torch.cuda.get_device_name(0)}")

    print(f"Model checkpoint being loaded       : {CHECKPOINT_PATH}")

    # Load frozen validation thresholds
    if not os.path.exists(THRESHOLDS_PATH):
        raise FileNotFoundError(f"Thresholds file missing: {THRESHOLDS_PATH}")
    with open(THRESHOLDS_PATH, "r") as f:
        frozen_thresholds = json.load(f)

    print("Frozen validation thresholds being used:")
    for cname in CLASS_NAMES:
        print(f"  - {cname:22s} : {frozen_thresholds[cname]:.2f}")

    # Verify sample tensor shape from dataset
    test_dataset = SoundSenseCachedDataset(
        metadata_path=METADATA_PATH,
        features_root="soundsense_data/features",
        split="test"
    )
    sample_spec, sample_target = test_dataset[0]
    print(f"Single item spectrogram shape       : {list(sample_spec.shape)} (Expected: [1, 128, 938])")
    print(f"Single item target shape            : {list(sample_target.shape)} (Expected: [8])")
    print(f"Expected batch input shape          : [batch_size, 1, 128, 938]")
    print(f"Expected batch output shape         : [batch_size, 8]")

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    print(f"Total test batches (batch_size=32)  : {len(test_loader)}")
    print(">> STEP 1 VERIFICATION PASSED.")

    # =========================================================================
    # STEP 2: LOAD FINAL MODEL
    # =========================================================================
    print("\n--- STEP 2: LOADING FINAL MODEL CHECKPOINT ---")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    epoch_saved = checkpoint.get("epoch", "unknown")
    macro_f1_saved = checkpoint.get("macro_f1", "unknown")
    micro_f1_saved = checkpoint.get("micro_f1", "unknown")
    print(f"Loaded checkpoint saved from epoch   : {epoch_saved}")
    print(f"Stored validation Macro F1 in ckpt   : {macro_f1_saved}")
    print(f"Stored validation Micro F1 in ckpt   : {micro_f1_saved}")

    model = SoundSenseCNN(num_classes=8).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model initialized and loaded in eval() mode with frozen weights.")
    print(">> STEP 2 PASSED.")

    # =========================================================================
    # STEP 3: RUN TEST INFERENCE (torch.no_grad)
    # =========================================================================
    print("\n--- STEP 3: RUNNING TEST INFERENCE (torch.no_grad) ---")
    t0 = time.time()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (specs, targets) in enumerate(test_loader, start=1):
            specs = specs.to(device, non_blocking=torch.cuda.is_available())
            logits = model(specs)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())

    infer_elapsed = time.time() - t0
    print(f"Inference completed in {infer_elapsed:.2f}s ({len(test_dataset)/infer_elapsed:.1f} samples/sec).")

    logits_matrix = np.vstack(all_logits)
    targets_matrix = np.vstack(all_targets)
    probs_matrix = 1.0 / (1.0 + np.exp(-logits_matrix))

    print(f"Probabilities matrix shape          : {probs_matrix.shape} (Expected: [664, 8])")
    print(f"Ground-truth labels matrix shape    : {targets_matrix.shape} (Expected: [664, 8])")
    assert probs_matrix.shape == (664, 8), f"Unexpected probs shape: {probs_matrix.shape}"
    assert targets_matrix.shape == (664, 8), f"Unexpected targets shape: {targets_matrix.shape}"
    print(">> STEP 3 PASSED.")

    # =========================================================================
    # STEP 4 & 5: APPLY FROZEN THRESHOLDS & CALCULATE FINAL TEST METRICS
    # =========================================================================
    print("\n--- STEP 4 & 5: APPLYING FROZEN THRESHOLDS & COMPUTING FINAL METRICS ---")
    threshold_vector = np.array([frozen_thresholds[cname] for cname in CLASS_NAMES])
    y_pred_frozen = (probs_matrix >= threshold_vector).astype(int)

    final_summary = calculate_metrics_for_predictions(targets_matrix, y_pred_frozen, frozen_thresholds)

    # =========================================================================
    # STEP 6: ALSO CALCULATE DEFAULT 0.5 TEST METRICS (FOR COMPARISON)
    # =========================================================================
    default_thresholds = {cname: 0.50 for cname in CLASS_NAMES}
    y_pred_default = (probs_matrix >= 0.50).astype(int)
    default_summary = calculate_metrics_for_predictions(targets_matrix, y_pred_default, default_thresholds)

    # =========================================================================
    # STEP 7: CONFUSION INFORMATION & PER-CLASS BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 105)
    print("STEP 7: PER-CLASS CONFUSION MATRIX & PERFORMANCE (FINAL FROZEN THRESHOLDS)")
    print("=" * 105)
    print(
        f"{'Class Name':22s} | {'Thresh':6s} | {'TP':5s} | {'FP':5s} | {'FN':5s} | {'TN':5s} | "
        f"{'ActPos':6s} | {'PredPos':7s} | {'Prec':6s} | {'Recall':6s} | {'F1':6s}"
    )
    print("-" * 105)
    for rec in final_summary["records"]:
        print(
            f"{rec['class_name']:22s} | {rec['threshold']:6.2f} | {rec['true_positive']:5d} | "
            f"{rec['false_positive']:5d} | {rec['false_negative']:5d} | {rec['true_negative']:5d} | "
            f"{rec['actual_positive_count']:6d} | {rec['predicted_positive_count']:7d} | "
            f"{rec['precision']:6.4f} | {rec['recall']:6.4f} | {rec['f1']:6.4f}"
        )
    print("-" * 105)

    print("\n" + "=" * 105)
    print("COMPARISON: DEFAULT (0.50) THRESHOLDS vs FINAL OPTIMIZED VALIDATION THRESHOLDS (ON TEST SET)")
    print("=" * 105)
    print(
        f"{'Class Name':22s} | {'Default F1 (0.5)':16s} | {'Final F1 (Tuned)':16s} | "
        f"{'F1 Difference':14s} | {'Default Recall':14s} | {'Final Recall':14s}"
    )
    print("-" * 105)
    for cname in CLASS_NAMES:
        d_f1 = default_summary["per_class"][cname]["f1"]
        f_f1 = final_summary["per_class"][cname]["f1"]
        diff = f_f1 - d_f1
        d_r = default_summary["per_class"][cname]["recall"]
        f_r = final_summary["per_class"][cname]["recall"]
        print(
            f"{cname:22s} | {d_f1:16.4f} | {f_f1:16.4f} | {diff:+14.4f} | "
            f"{d_r:14.4f} | {f_r:14.4f}"
        )
    print("-" * 105)

    # =========================================================================
    # STEP 8: SAVE FINAL RESULTS
    # =========================================================================
    print("\n--- STEP 8: SAVING FINAL TEST ARTIFACTS ---")
    df_results = pd.DataFrame(final_summary["records"])
    df_results.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Saved CSV: {CSV_OUTPUT_PATH}")

    full_json_payload = {
        "number_of_test_samples": 664,
        "model_checkpoint": CHECKPOINT_PATH,
        "checkpoint_epoch": int(epoch_saved) if isinstance(epoch_saved, (int, np.integer)) or str(epoch_saved).isdigit() else epoch_saved,
        "thresholds_used": frozen_thresholds,
        "macro_precision": final_summary["macro_precision"],
        "macro_recall": final_summary["macro_recall"],
        "macro_f1": final_summary["macro_f1"],
        "micro_precision": final_summary["micro_precision"],
        "micro_recall": final_summary["micro_recall"],
        "micro_f1": final_summary["micro_f1"],
        "total_true_positives": final_summary["total_true_positives"],
        "total_false_positives": final_summary["total_false_positives"],
        "total_false_negatives": final_summary["total_false_negatives"],
        "total_true_negatives": final_summary["total_true_negatives"],
        "per_class_metrics": final_summary["per_class"],
        "default_0_5_comparison": {
            "macro_precision": default_summary["macro_precision"],
            "macro_recall": default_summary["macro_recall"],
            "macro_f1": default_summary["macro_f1"],
            "micro_precision": default_summary["micro_precision"],
            "micro_recall": default_summary["micro_recall"],
            "micro_f1": default_summary["micro_f1"]
        }
    }

    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(full_json_payload, f, indent=4)
    print(f"Saved JSON: {JSON_OUTPUT_PATH}")

    # =========================================================================
    # STEP 9: COMPARE VALIDATION VS TEST
    # =========================================================================
    val_macro_f1 = 0.4048
    val_micro_f1 = 0.5261

    test_macro_f1 = final_summary["macro_f1"]
    test_micro_f1 = final_summary["micro_f1"]

    macro_diff = test_macro_f1 - val_macro_f1
    micro_diff = test_micro_f1 - val_micro_f1

    print("\n" + "=" * 85)
    print("STEP 9: VALIDATION vs FINAL TEST GENERALIZATION COMPARISON")
    print("=" * 85)
    print(f"{'Metric':25s} | {'Validation':15s} | {'Final Test':15s} | {'Difference (Gen Gap)':20s}")
    print("-" * 85)
    print(f"{'Macro F1':25s} | {val_macro_f1:15.4f} | {test_macro_f1:15.4f} | {macro_diff:+20.4f}")
    print(f"{'Micro F1':25s} | {val_micro_f1:15.4f} | {test_micro_f1:15.4f} | {micro_diff:+20.4f}")
    print("-" * 85)

    # =========================================================================
    # STEP 10: PROGRESSION TABLE & FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 85)
    print("STEP 10: SOUNDSENSE COMPLETE EXPERIMENT PROGRESSION")
    print("=" * 85)
    print(f"{'Experiment':46s} | {'Validation Macro F1':20s} | {'Validation Micro F1':20s}")
    print("-" * 90)
    print(f"{'1. Original baseline (raw audio STFT)':46s} | {'0.1737':20s} | {'0.1848':20s}")
    print(f"{'2. Cached baseline (precomputed .npy)':46s} | {'0.2799':20s} | {'0.5112':20s}")
    print(f"{'3. Class-weighted BCE (Epoch 5)':46s} | {'0.3405':20s} | {'0.3593':20s}")
    print(f"{'4. Weighted BCE + Validation Threshold Tuning':46s} | {'0.4048':20s} | {'0.5261':20s}")
    print("-" * 90)

    print("\nFINAL HELD-OUT TEST PERFORMANCE:")
    print(f"  - Final Test Macro F1        : {test_macro_f1:.4f}")
    print(f"  - Final Test Micro F1        : {test_micro_f1:.4f}")
    print(f"  - Final Test Macro Precision : {final_summary['macro_precision']:.4f}")
    print(f"  - Final Test Macro Recall    : {final_summary['macro_recall']:.4f}")
    print(f"  - Final Test Micro Precision : {final_summary['micro_precision']:.4f}")
    print(f"  - Final Test Micro Recall    : {final_summary['micro_recall']:.4f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
