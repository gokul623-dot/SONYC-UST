"""
tune_thresholds.py

Controlled Per-Class Decision Threshold Tuning Experiment for SoundSense.

Loads the existing best checkpoint:
    soundsense_data/models/weighted_best_model.pth (Epoch 5, weighted BCE)
Runs inference strictly on the validation set (4,308 samples, soundsense_data/features/validate/).
NO RETRAINING. NO WEIGHT UPDATES. TEST SET UNTOUCHED.

Procedure:
1. Run inference over all 4,308 validation recordings.
2. Verify baseline metrics at default threshold = 0.5 to reproduce Epoch 5 metrics:
   Macro F1 ~ 0.3405, Micro F1 ~ 0.3593.
3. Search optimal threshold per class from 0.05 to 0.95 (step 0.01).
4. Combine optimal thresholds into a vector and evaluate combined multi-label performance.
5. Save results to soundsense_data/threshold_tuning_results.csv and soundsense_data/optimized_thresholds.json.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

CHECKPOINT_PATH = os.path.join("soundsense_data", "models", "weighted_best_model.pth")
CSV_OUTPUT_PATH = os.path.join("soundsense_data", "threshold_tuning_results.csv")
JSON_OUTPUT_PATH = os.path.join("soundsense_data", "optimized_thresholds.json")


def compute_class_prf(yt, yp):
    """Computes precision, recall, and F1 for a binary 1D array."""
    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == 0) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))

    p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def evaluate_predictions(y_true, y_pred):
    """Computes per-class and global macro/micro F1 given binary prediction matrix."""
    per_class_p = {}
    per_class_r = {}
    per_class_f1 = {}

    for i, cname in enumerate(CLASS_NAMES):
        p, r, f1 = compute_class_prf(y_true[:, i], y_pred[:, i])
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


def main():
    print("=" * 80)
    print("SOUNDSENSE: CONTROLLED PER-CLASS THRESHOLD TUNING EXPERIMENT")
    print("=" * 80)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # 2. Load Checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    print(f"\nLoading checkpoint from: {CHECKPOINT_PATH} ...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    epoch_saved = checkpoint.get("epoch", "unknown")
    ckpt_macro_f1 = checkpoint.get("macro_f1", 0.0)
    ckpt_micro_f1 = checkpoint.get("micro_f1", 0.0)
    print(f"  - Checkpoint Epoch : {epoch_saved}")
    print(f"  - Stored Macro F1  : {ckpt_macro_f1:.4f}")
    print(f"  - Stored Micro F1  : {ckpt_micro_f1:.4f}")

    model = SoundSenseCNN(num_classes=8).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model successfully loaded in eval() mode. Weights are FROZEN.")

    # 3. Load Validation Dataset
    print("\nLoading Validation Dataset (4,308 recordings)...")
    val_dataset = SoundSenseCachedDataset(
        metadata_path="soundsense_data/metadata.csv",
        features_root="soundsense_data/features",
        split="validate"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    print(f"  - Validation Samples : {len(val_dataset):,}")
    print(f"  - Total Batches      : {len(val_loader)}")

    # 4. Run Inference on Validation Set
    print("\nRunning inference on validation set (torch.no_grad)...")
    all_logits = []
    all_targets = []
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (specs, targets) in enumerate(val_loader, start=1):
            specs = specs.to(device, non_blocking=torch.cuda.is_available())
            logits = model(specs)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())

    infer_time = time.time() - t_start
    print(f"Inference complete in {infer_time:.2f}s ({len(val_dataset)/infer_time:.1f} samples/sec).")

    logits_matrix = np.vstack(all_logits)
    targets_matrix = np.vstack(all_targets)
    probs_matrix = 1.0 / (1.0 + np.exp(-logits_matrix))  # sigmoid

    print(f"  - Predictions shape : {probs_matrix.shape} (Expected: [4308, 8])")
    print(f"  - Targets shape     : {targets_matrix.shape} (Expected: [4308, 8])")
    assert probs_matrix.shape == (4308, 8), f"Unexpected probs shape: {probs_matrix.shape}"
    assert targets_matrix.shape == (4308, 8), f"Unexpected targets shape: {targets_matrix.shape}"

    # 5. Baseline Verification at Threshold = 0.5
    print("\n" + "=" * 80)
    print("STEP 4: VERIFY BASELINE REPRODUCTION AT THRESHOLD = 0.5")
    print("=" * 80)
    y_pred_default = (probs_matrix >= 0.5).astype(int)
    def_p, def_r, def_f1, def_macro, def_micro = evaluate_predictions(targets_matrix, y_pred_default)

    print(f"Default (0.5) Macro F1 : {def_macro:.4f} (Expected: ~0.3405)")
    print(f"Default (0.5) Micro F1 : {def_micro:.4f} (Expected: ~0.3593)")

    print("\nPer-Class Breakdown at Default Threshold 0.5:")
    print(f"{'Class Name':24s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s}")
    print("-" * 62)
    for cname in CLASS_NAMES:
        print(f"{cname:24s} | {def_p[cname]:10.4f} | {def_r[cname]:10.4f} | {def_f1[cname]:10.4f}")

    # Check reproduction discrepancy
    macro_discrepancy = abs(def_macro - 0.3405)
    micro_discrepancy = abs(def_micro - 0.3593)
    if macro_discrepancy > 0.005:
        raise ValueError(
            f"FATAL: Baseline reproduction discrepancy too high! "
            f"Computed Macro F1={def_macro:.4f} vs Expected=0.3405 (diff: {macro_discrepancy:.4f}). STOPPING."
        )
    print(f"\n>> Baseline reproduction VERIFIED (Macro diff: {macro_discrepancy:.6f}, Micro diff: {micro_discrepancy:.6f}).")

    # 6. Per-Class Threshold Optimization (0.05 to 0.95, step 0.01)
    print("\n" + "=" * 80)
    print("STEP 5: INDEPENDENT PER-CLASS THRESHOLD GRID SEARCH (0.05 -> 0.95)")
    print("=" * 80)
    candidate_thresholds = [round(float(t), 2) for t in np.arange(0.05, 0.951, 0.01)]
    print(f"Total candidate thresholds per class: {len(candidate_thresholds)} (from {candidate_thresholds[0]} to {candidate_thresholds[-1]})")

    best_thresholds = {}
    tuned_p = {}
    tuned_r = {}
    tuned_f1 = {}
    tuning_records = []

    for i, cname in enumerate(CLASS_NAMES):
        yt_c = targets_matrix[:, i]
        yp_probs_c = probs_matrix[:, i]

        best_t = 0.5
        best_c_f1 = -1.0
        best_c_p = 0.0
        best_c_r = 0.0

        for t in candidate_thresholds:
            pred_c = (yp_probs_c >= t).astype(int)
            p_val, r_val, f1_val = compute_class_prf(yt_c, pred_c)

            # Maximize F1; if tied, prefer threshold closest to 0.5
            if f1_val > best_c_f1 or (np.isclose(f1_val, best_c_f1) and abs(t - 0.5) < abs(best_t - 0.5)):
                best_c_f1 = f1_val
                best_c_p = p_val
                best_c_r = r_val
                best_t = t

        best_thresholds[cname] = round(best_t, 2)
        tuned_p[cname] = best_c_p
        tuned_r[cname] = best_c_r
        tuned_f1[cname] = best_c_f1

        tuning_records.append({
            "class_name": cname,
            "default_threshold": 0.50,
            "best_threshold": round(best_t, 2),
            "default_precision": round(def_p[cname], 4),
            "default_recall": round(def_r[cname], 4),
            "default_f1": round(def_f1[cname], 4),
            "tuned_precision": round(best_c_p, 4),
            "tuned_recall": round(best_c_r, 4),
            "tuned_f1": round(best_c_f1, 4)
        })

    # 7. Print Per-Class Summary Table
    print("\n" + "=" * 80)
    print("STEP 6: PER-CLASS OPTIMAL THRESHOLDS & GAINS")
    print("=" * 80)
    print(f"{'Class Name':22s} | {'Default T':9s} | {'Best T':8s} | {'Def F1':8s} | {'Tuned F1':8s} | {'Diff':8s} | {'Tuned P':8s} | {'Tuned R':8s}")
    print("-" * 96)
    for rec in tuning_records:
        diff = rec["tuned_f1"] - rec["default_f1"]
        print(f"{rec['class_name']:22s} | {rec['default_threshold']:9.2f} | {rec['best_threshold']:8.2f} | {rec['default_f1']:8.4f} | {rec['tuned_f1']:8.4f} | {diff:+8.4f} | {rec['tuned_precision']:8.4f} | {rec['tuned_recall']:8.4f}")
    print("-" * 96)

    # 8. Apply Vector of Thresholds Simultaneously
    print("\n" + "=" * 80)
    print("STEP 7: COMBINED MULTI-LABEL EVALUATION WITH OPTIMIZED THRESHOLDS")
    print("=" * 80)
    threshold_vector = np.array([best_thresholds[cname] for cname in CLASS_NAMES])
    print(f"Optimal Thresholds Vector: {threshold_vector.tolist()}")

    # Vectorized multi-label thresholding: probs_matrix >= threshold_vector
    y_pred_tuned = (probs_matrix >= threshold_vector).astype(int)
    comb_p, comb_r, comb_f1, comb_macro, comb_micro = evaluate_predictions(targets_matrix, y_pred_tuned)

    # 9. Comparison & Analysis
    macro_abs_diff = comb_macro - def_macro
    macro_rel_diff = (macro_abs_diff / def_macro) * 100
    micro_abs_diff = comb_micro - def_micro
    micro_rel_diff = (micro_abs_diff / def_micro) * 100

    print(f"\n{'Metric':28s} | {'Default (0.5)':15s} | {'Tuned Thresholds':18s} | {'Difference':18s}")
    print("-" * 85)
    print(f"{'Validation Macro F1':28s} | {def_macro:15.4f} | {comb_macro:18.4f} | {macro_abs_diff:+18.4f} ({macro_rel_diff:+.1f}%)")
    print(f"{'Validation Micro F1':28s} | {def_micro:15.4f} | {comb_micro:18.4f} | {micro_abs_diff:+18.4f} ({micro_rel_diff:+.1f}%)")
    print("-" * 85)

    # 10. Save Artifacts
    print("\n" + "=" * 80)
    print("STEP 9: SAVING RESULTS ARTIFACTS")
    print("=" * 80)
    df_results = pd.DataFrame(tuning_records)
    df_results.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Saved threshold tuning results CSV to: {CSV_OUTPUT_PATH}")

    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(best_thresholds, f, indent=4)
    print(f"Saved optimized thresholds JSON to : {JSON_OUTPUT_PATH}")
    print(f"JSON Contents:\n{json.dumps(best_thresholds, indent=4)}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE: THRESHOLD TUNING FINISHED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
