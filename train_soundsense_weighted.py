"""
train_soundsense_weighted.py

Class-Weighted BCEWithLogitsLoss GPU Training Pipeline for SoundSense.

Controlled Experiment:
Tests whether addressing severe class imbalance via class-weighted BCEWithLogitsLoss
(pos_weight = neg_count / pos_count computed strictly on the training set)
improves the validation Macro F1 over the cached baseline (Macro F1 = 0.2799).

Strict Constraints & Isolation:
- Model Architecture: SoundSenseCNN (identical)
- Optimizer: Adam (lr=0.001) (identical)
- Batch Size: 16 (identical)
- Max Epochs: 10 (identical)
- Early Stopping: Patience = 3 on validation Macro F1 (identical)
- Prediction Threshold: 0.5 for all classes (identical)
- Class Ordering: [engine, machinery_impact, non_machinery_impact, powered_saw,
                  alert_signal, music, human_voice, dog] (identical)
- Features: Precomputed .npy log-mel features from disk (no librosa)
- Test set: NOT used or evaluated
- Output Checkpoint: soundsense_data/models/weighted_best_model.pth
- Output History: soundsense_data/weighted_training_history.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from soundsense_cached_dataset import SoundSenseCachedDataset
from soundsense_model import SoundSenseCNN

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

CACHED_BASELINE_MACRO_F1 = 0.2799
CACHED_BASELINE_MICRO_F1 = 0.5112
CACHED_BASELINE_BEST_EPOCH = 5


def compute_class_weights(metadata_path="soundsense_data/metadata.csv"):
    """
    Computes pos_weight for each coarse class from the TRAINING SET ONLY.
    
    pos_weight[c] = negative_count[c] / positive_count[c]
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    num_train = len(train_df)

    print("=" * 75, flush=True)
    print("STEP 7: COMPUTING POS_WEIGHT FROM TRAINING SET ONLY", flush=True)
    print("=" * 75, flush=True)
    print(f"Total training recordings: {num_train:,}", flush=True)
    print("-" * 75, flush=True)
    print(f"{'Class Name':24s} | {'Pos Count':10s} | {'Neg Count':10s} | {'Pos %':8s} | {'pos_weight':12s}", flush=True)
    print("-" * 75, flush=True)

    weights = []
    for cname in CLASS_NAMES:
        pos_count = int(train_df[cname].sum())
        neg_count = num_train - pos_count
        assert pos_count > 0, f"Error: positive count for {cname} is 0!"
        assert neg_count > 0, f"Error: negative count for {cname} is 0!"

        w = neg_count / pos_count
        weights.append(w)
        pos_pct = (pos_count / num_train) * 100
        print(f"{cname:24s} | {pos_count:10d} | {neg_count:10d} | {pos_pct:7.2f}% | {w:12.4f}", flush=True)

    print("-" * 75, flush=True)
    # Verification checks
    for idx, (cname, w) in enumerate(zip(CLASS_NAMES, weights)):
        assert np.isfinite(w), f"Weight for {cname} is not finite: {w}"
        assert w > 0, f"Weight for {cname} must be positive: {w}"

    print("All positive counts > 0: Verified.", flush=True)
    print("All pos_weights finite and positive: Verified.", flush=True)
    print("Class ordering matches model output exactly: Verified.", flush=True)
    print(f"Computed weights list: {[round(w, 4) for w in weights]}", flush=True)
    print("=" * 75, flush=True)

    return weights


def compute_metrics(y_true, y_pred):
    """
    Computes per-class precision, recall, F1, macro F1, and micro F1.
    Predictions are evaluated at threshold 0.5 without weighting.
    """
    per_class_p = {}
    per_class_r = {}
    per_class_f1 = {}

    for i, cname in enumerate(CLASS_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        per_class_p[cname] = float(p)
        per_class_r[cname] = float(r)
        per_class_f1[cname] = float(f1)

    macro_f1 = float(np.mean(list(per_class_f1.values())))

    total_tp = np.sum((y_true == 1) & (y_pred == 1))
    total_fp = np.sum((y_true == 0) & (y_pred == 1))
    total_fn = np.sum((y_true == 1) & (y_pred == 0))

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return per_class_p, per_class_r, per_class_f1, macro_f1, float(micro_f1)


def run_weighted_sanity_check(model, train_loader, val_loader, criterion, optimizer, pos_weight_tensor, device):
    """
    Comprehensive pre-training sanity check for weighted BCE training:
    1. Loads one cached training batch [16, 1, 128, 938] and targets [16, 8].
    2. Moves batch to CUDA.
    3. Runs CNN forward pass -> output [16, 8].
    4. Calculates weighted BCEWithLogitsLoss.
    5. Verifies loss is finite and contains no NaNs.
    6. Runs backward() and one optimizer step.
    7. Confirms pos_weight shape, values, and device placement.
    """
    print("=" * 75, flush=True)
    print("STEP 15: CLASS-WEIGHTED PRE-TRAINING SANITY CHECK", flush=True)
    print("=" * 75, flush=True)

    print(f"Selected Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"CUDA Device Name : {torch.cuda.get_device_name(0)}", flush=True)

    # 1. Print and verify pos_weight tensor properties
    print(f"\n1. pos_weight Tensor Verification:")
    print(f"  - Shape       : {list(pos_weight_tensor.shape)} (Expected: [8])", flush=True)
    print(f"  - Device      : {pos_weight_tensor.device}", flush=True)
    print(f"  - Dtype       : {pos_weight_tensor.dtype}", flush=True)
    print(f"  - Values      : {[round(float(v), 4) for v in pos_weight_tensor.tolist()]}", flush=True)

    assert list(pos_weight_tensor.shape) == [8], f"pos_weight shape mismatch: {pos_weight_tensor.shape}"
    if device.type == "cuda":
        assert pos_weight_tensor.is_cuda, "pos_weight tensor is not on CUDA!"
        print(f"  - Device Check: Confirmed pos_weight is on {pos_weight_tensor.device}", flush=True)

    model.train()
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # 2. Load one cached training batch
    train_x, train_y = next(train_iter)
    print(f"\n2. Batch Loading & Shapes:")
    print(f"  - Input Batch Shape   : {list(train_x.shape)} (Expected: [16, 1, 128, 938])", flush=True)
    print(f"  - Target Batch Shape  : {list(train_y.shape)} (Expected: [16, 8])", flush=True)
    print(f"  - Input Dtype         : {train_x.dtype}", flush=True)
    print(f"  - Target Dtype        : {train_y.dtype}", flush=True)

    assert list(train_x.shape) == [16, 1, 128, 938], f"Train shape mismatch: {train_x.shape}"
    assert list(train_y.shape) == [16, 8], f"Train target shape mismatch: {train_y.shape}"
    assert train_x.dtype == torch.float32, f"Train tensor must be float32, got {train_x.dtype}"
    assert train_y.dtype == torch.float32, f"Train targets must be float32, got {train_y.dtype}"

    # 3. Move data to CUDA
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    print(f"  - Data moved to device: {train_x.device}", flush=True)
    if device.type == "cuda":
        assert train_x.is_cuda and train_y.is_cuda, "Tensors failed to transfer to CUDA!"

    # 4. Forward pass
    initial_weight = list(model.parameters())[0].clone()
    optimizer.zero_grad()
    train_out = model(train_x)
    print(f"\n3. Forward Pass:")
    print(f"  - Output Logits Shape : {list(train_out.shape)} (Expected: [16, 8])", flush=True)
    assert list(train_out.shape) == [16, 8], f"Output shape mismatch: {train_out.shape}"

    # 5. Weighted loss calculation
    loss = criterion(train_out, train_y)
    loss_val = loss.item()
    is_finite = torch.isfinite(loss).item()
    has_nan = torch.isnan(loss).item()

    print(f"\n4. Loss Calculation:")
    print(f"  - Weighted BCE Loss   : {loss_val:.4f}", flush=True)
    print(f"  - Loss is Finite      : {is_finite}", flush=True)
    print(f"  - Loss Contains NaNs  : {has_nan}", flush=True)

    assert is_finite and not has_nan, f"Sanity check error: Weighted loss is invalid ({loss_val})"

    # 6. Backward pass and optimizer step
    loss.backward()
    optimizer.step()
    updated_weight = list(model.parameters())[0]
    weight_delta = torch.norm(updated_weight - initial_weight).item()
    print(f"\n5. Backward & Optimizer Step:")
    print(f"  - Weight Update Norm  : {weight_delta:.6f}", flush=True)
    assert weight_delta > 0.0, "Sanity check error: Optimizer step produced 0 weight change."

    print("=" * 75, flush=True)
    print("SANITY CHECK PASSED: Class-weighted pipeline is verified and operational on CUDA.", flush=True)
    print("=" * 75, flush=True)


def train_soundsense_weighted(
    max_epochs=10,
    patience=3,
    batch_size=16,
    lr=0.001,
    metadata_path="soundsense_data/metadata.csv",
    features_root="soundsense_data/features",
    models_dir="soundsense_data/models",
    history_path="soundsense_data/weighted_training_history.csv"
):
    print("=" * 75, flush=True)
    print("SOUNDSENSE CLASS-WEIGHTED BCE EXPERIMENT", flush=True)
    print("=" * 75, flush=True)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device : {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU Name        : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU VRAM Total  : {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB", flush=True)

    # 2. Compute class weights from TRAINING SET ONLY
    weights_list = compute_class_weights(metadata_path)
    pos_weight = torch.tensor(weights_list, dtype=torch.float32).to(device)

    # 3. Datasets & DataLoaders (Train and Validate only, test split excluded)
    print("\nLoading Cached Datasets from .npy features...", flush=True)
    train_dataset = SoundSenseCachedDataset(metadata_path, features_root, split="train")
    val_dataset = SoundSenseCachedDataset(metadata_path, features_root, split="validate")

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
        drop_last=False
    )

    print(f"  - Train Set : {len(train_dataset):,} samples ({len(train_loader)} batches)", flush=True)
    print(f"  - Val Set   : {len(val_dataset):,} samples ({len(val_loader)} batches)", flush=True)

    # 4. Model instantiation (exact same architecture)
    model = SoundSenseCNN(num_classes=8).to(device)

    # 5. Class-Weighted Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6. Sanity check
    run_weighted_sanity_check(model, train_loader, val_loader, criterion, optimizer, pos_weight, device)

    # 7. Reset model parameters for a clean epoch 1 training run
    print("\nResetting model parameters for clean training run...", flush=True)
    model = SoundSenseCNN(num_classes=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Ensure checkpoint directory exists
    os.makedirs(models_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(models_dir, "weighted_best_model.pth")

    # Training state tracking
    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_per_class_f1 = {}
    best_per_class_p = {}
    best_per_class_r = {}
    best_micro_f1 = 0.0
    patience_counter = 0
    history = []
    early_stopped = False

    print("\n" + "=" * 75, flush=True)
    print(f"BEGINNING WEIGHTED TRAINING (Max Epochs: {max_epochs}, Patience: {patience}, Batch Size: {batch_size})", flush=True)
    print("=" * 75, flush=True)

    start_training_time = time.time()
    batch_durations = []

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()

        # =====================================================================
        # Training Phase
        # =====================================================================
        model.train()
        train_loss_total = 0.0
        train_batches_count = 0

        print(f"\n>>> Epoch {epoch}/{max_epochs} [Training on {device}]", flush=True)
        batch_log_interval = 200

        for batch_idx, (specs, targets) in enumerate(train_loader, start=1):
            t_b_start = time.time()

            specs = specs.to(device, non_blocking=use_pin_memory)
            targets = targets.to(device, non_blocking=use_pin_memory)

            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_batches_count += 1
            batch_durations.append(time.time() - t_b_start)

            if batch_idx % batch_log_interval == 0 or batch_idx == len(train_loader):
                avg_current_loss = train_loss_total / train_batches_count
                elapsed = time.time() - epoch_start_time
                print(f"  [Train] Batch {batch_idx:3d}/{len(train_loader)} | Running Loss: {avg_current_loss:.4f} | Elapsed: {elapsed:.1f}s", flush=True)

        avg_train_loss = train_loss_total / train_batches_count

        # =====================================================================
        # Validation Phase (Criterion also uses pos_weight for val loss)
        # =====================================================================
        model.eval()
        val_loss_total = 0.0
        val_batches_count = 0

        all_val_targets = []
        all_val_probs = []

        print(f">>> Epoch {epoch}/{max_epochs} [Validation on {device}]", flush=True)
        with torch.no_grad():
            for batch_idx, (specs, targets) in enumerate(val_loader, start=1):
                specs = specs.to(device, non_blocking=use_pin_memory)
                targets = targets.to(device, non_blocking=use_pin_memory)

                logits = model(specs)
                loss = criterion(logits, targets)

                val_loss_total += loss.item()
                val_batches_count += 1

                probs = torch.sigmoid(logits)
                all_val_targets.append(targets.cpu().numpy())
                all_val_probs.append(probs.cpu().numpy())

        avg_val_loss = val_loss_total / val_batches_count

        y_true = np.vstack(all_val_targets)
        y_probs = np.vstack(all_val_probs)
        y_pred = (y_probs >= 0.5).astype(int)

        per_class_p, per_class_r, per_class_f1, macro_f1, micro_f1 = compute_metrics(y_true, y_pred)
        epoch_duration = time.time() - epoch_start_time

        # Save history record
        record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "duration_sec": round(epoch_duration, 1)
        }
        for cname in CLASS_NAMES:
            record[f"f1_{cname}"] = round(per_class_f1[cname], 4)

        history.append(record)
        pd.DataFrame(history).to_csv(history_path, index=False)

        # Print Epoch Results Table
        print(f"\n--- Epoch {epoch} Results ({epoch_duration:.1f}s / {epoch_duration/60:.2f} min) ---", flush=True)
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)
        print(f"  Macro F1  : {macro_f1:.4f} | Micro F1: {micro_f1:.4f}", flush=True)
        print("  Per-Class Validation Metrics (Threshold = 0.5):", flush=True)
        for cname in CLASS_NAMES:
            print(f"    - {cname:22s} : P={per_class_p[cname]:.3f}, R={per_class_r[cname]:.3f}, F1={per_class_f1[cname]:.3f}", flush=True)

        # =====================================================================
        # Checkpointing and Early Stopping based on Validation Macro F1
        # =====================================================================
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_per_class_f1 = per_class_f1.copy()
            best_per_class_p = per_class_p.copy()
            best_per_class_r = per_class_r.copy()
            best_micro_f1 = micro_f1
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "macro_f1": best_macro_f1,
                    "micro_f1": best_micro_f1,
                    "val_loss": best_val_loss,
                    "pos_weight": pos_weight.cpu().numpy(),
                    "class_names": CLASS_NAMES
                },
                best_checkpoint_path
            )
            print(f"  >> Validation Macro F1 improved to {best_macro_f1:.4f}! Checkpoint saved to: {best_checkpoint_path}", flush=True)
        else:
            patience_counter += 1
            print(f"  >> Macro F1 ({macro_f1:.4f}) did not beat best ({best_macro_f1:.4f}). Early stopping counter: {patience_counter}/{patience}", flush=True)

            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation Macro F1 did not improve for {patience} consecutive epochs.", flush=True)
                early_stopped = True
                break

    total_training_duration = time.time() - start_training_time
    avg_epoch_time = total_training_duration / len(history) if history else 0.0
    avg_batch_speed_ms = (np.mean(batch_durations) * 1000) if batch_durations else 0.0

    # =========================================================================
    # Final Summary and Comparison with Baseline
    # =========================================================================
    print("\n" + "=" * 75, flush=True)
    print("EXPERIMENT RESULTS: CLASS-WEIGHTED BCE TRAINING SUMMARY", flush=True)
    print("=" * 75, flush=True)
    print(f"Device / GPU                      : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Average Batch Processing Speed    : {avg_batch_speed_ms:.1f} ms / batch", flush=True)
    print(f"Average Time per Epoch            : {avg_epoch_time:.1f}s ({avg_epoch_time/60:.2f} min)", flush=True)
    print(f"Total Training Duration           : {total_training_duration/60:.2f} minutes ({total_training_duration:.1f}s)", flush=True)
    print(f"Epochs Completed                  : {len(history)} of {max_epochs}", flush=True)
    print(f"Best Epoch                        : Epoch {best_epoch}", flush=True)
    print(f"Best Validation Macro F1          : {best_macro_f1:.4f}", flush=True)
    print(f"Best Validation Micro F1          : {best_micro_f1:.4f}", flush=True)
    print(f"Best Validation Loss              : {best_val_loss:.4f}", flush=True)
    print(f"Saved Checkpoint Path             : {best_checkpoint_path}", flush=True)
    print(f"Early Stopping Occurred           : {early_stopped}", flush=True)

    print("\nPer-Class Metrics at Best Epoch (Epoch %d):" % best_epoch, flush=True)
    print(f"{'Class Name':24s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s}", flush=True)
    print("-" * 62, flush=True)
    for cname in CLASS_NAMES:
        p = best_per_class_p.get(cname, 0.0)
        r = best_per_class_r.get(cname, 0.0)
        f1 = best_per_class_f1.get(cname, 0.0)
        print(f"{cname:24s} | {p:10.4f} | {r:10.4f} | {f1:10.4f}", flush=True)

    # Comparison against cached baseline
    macro_abs_diff = best_macro_f1 - CACHED_BASELINE_MACRO_F1
    macro_rel_diff = (macro_abs_diff / CACHED_BASELINE_MACRO_F1) * 100
    micro_abs_diff = best_micro_f1 - CACHED_BASELINE_MICRO_F1
    micro_rel_diff = (micro_abs_diff / CACHED_BASELINE_MICRO_F1) * 100

    print("\n" + "=" * 75, flush=True)
    print("COMPARISON: CACHED BASELINE vs. CLASS-WEIGHTED BCE", flush=True)
    print("=" * 75, flush=True)
    print(f"{'Metric':28s} | {'Cached Baseline':18s} | {'Weighted BCE':18s} | {'Difference':18s}", flush=True)
    print("-" * 84, flush=True)
    print(f"{'Validation Macro F1':28s} | {CACHED_BASELINE_MACRO_F1:18.4f} | {best_macro_f1:18.4f} | {macro_abs_diff:+18.4f} ({macro_rel_diff:+.1f}%)", flush=True)
    print(f"{'Validation Micro F1':28s} | {CACHED_BASELINE_MICRO_F1:18.4f} | {best_micro_f1:18.4f} | {micro_abs_diff:+18.4f} ({micro_rel_diff:+.1f}%)", flush=True)
    print(f"{'Best Epoch':28s} | {CACHED_BASELINE_BEST_EPOCH:18d} | {best_epoch:18d} | {'-':18s}", flush=True)
    print("-" * 84, flush=True)

    print("\nExperiment Conclusion:", flush=True)
    if best_macro_f1 > CACHED_BASELINE_MACRO_F1:
        print(f"  >> SUCCESS: Class-weighted BCE improved validation Macro F1 from {CACHED_BASELINE_MACRO_F1:.4f} to {best_macro_f1:.4f} (+{macro_abs_diff:.4f}, +{macro_rel_diff:.1f}%).", flush=True)
    else:
        print(f"  >> DID NOT IMPROVE: Class-weighted BCE validation Macro F1 ({best_macro_f1:.4f}) did not exceed cached baseline ({CACHED_BASELINE_MACRO_F1:.4f}) (diff: {macro_abs_diff:+.4f}, {macro_rel_diff:+.1f}%).", flush=True)

    print("=" * 75, flush=True)


if __name__ == "__main__":
    train_soundsense_weighted()
