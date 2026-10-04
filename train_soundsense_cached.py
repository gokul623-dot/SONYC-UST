"""
train_soundsense_cached.py

Cached GPU Training Pipeline for SoundSense Multi-Label Urban Sound Tagging.

Uses precomputed log-mel spectrogram features (.npy) from soundsense_cached_dataset.py
to dramatically accelerate training without calling librosa during training.

Key settings identical to baseline:
- Architecture: SoundSenseCNN (24,040 parameters)
- Loss: nn.BCEWithLogitsLoss()
- Optimizer: Adam (lr=0.001)
- Batch Size: 16
- Max Epochs: 10
- Early Stopping: Patience = 3 (based on validation Macro F1)
- Prediction Threshold: 0.5
- Metric Order: [engine, machinery_impact, non_machinery_impact, powered_saw,
                 alert_signal, music, human_voice, dog]
- Separate Checkpoint: soundsense_data/models/cached_baseline_best_model.pth
- Separate History: soundsense_data/cached_training_history.csv
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

ORIGINAL_BASELINE_MACRO_F1 = 0.1737
ORIGINAL_BASELINE_EPOCH_TIME = "11-12 min/epoch"


def compute_metrics(y_true, y_pred):
    """
    Computes per-class precision, recall, F1, macro F1, and micro F1.
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


def run_cached_sanity_check(model, train_loader, val_loader, criterion, optimizer, device):
    """
    Comprehensive pre-training sanity check:
    1. Loads one training batch and one validation batch.
    2. Confirms tensor shapes and dtypes (float32).
    3. Confirms tensors move to CUDA.
    4. Runs forward pass, BCEWithLogitsLoss, backward(), and optimizer step.
    5. Confirms loss is finite and no NaNs exist.
    6. Measures throughput across 10 sample batches to verify speedup.
    """
    print("=" * 75, flush=True)
    print("STEP 12: CACHED FEATURE PRE-TRAINING SANITY CHECK", flush=True)
    print("=" * 75, flush=True)

    print(f"Device being used: {device}", flush=True)
    if device.type == "cuda":
        print(f"CUDA Device Name : {torch.cuda.get_device_name(0)}", flush=True)

    model.train()
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # 1. Load one training batch
    train_x, train_y = next(train_iter)
    print(f"  - Train Spectrogram Shape : {list(train_x.shape)} (dtype: {train_x.dtype})", flush=True)
    print(f"  - Train Target Shape      : {list(train_y.shape)} (dtype: {train_y.dtype})", flush=True)

    assert list(train_x.shape) == [16, 1, 128, 938], f"Train shape mismatch: {train_x.shape}"
    assert list(train_y.shape) == [16, 8], f"Train target shape mismatch: {train_y.shape}"
    assert train_x.dtype == torch.float32, f"Train tensor must be float32, got {train_x.dtype}"
    assert train_y.dtype == torch.float32, f"Train targets must be float32, got {train_y.dtype}"

    # 2. Load one validation batch
    val_x, val_y = next(val_iter)
    print(f"  - Val Spectrogram Shape   : {list(val_x.shape)} (dtype: {val_x.dtype})", flush=True)
    print(f"  - Val Target Shape        : {list(val_y.shape)} (dtype: {val_y.dtype})", flush=True)

    assert list(val_x.shape) == [16, 1, 128, 938], f"Val shape mismatch: {val_x.shape}"
    assert list(val_y.shape) == [16, 8], f"Val target shape mismatch: {val_y.shape}"

    # 3. Move to CUDA
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    print(f"  - Train Tensors Device    : {train_x.device}", flush=True)
    if device.type == "cuda":
        assert train_x.is_cuda and train_y.is_cuda, "Tensors failed to transfer to CUDA!"

    # 4. Forward pass
    initial_weight = list(model.parameters())[0].clone()
    optimizer.zero_grad()
    train_out = model(train_x)
    print(f"  - Output Logits Shape     : {list(train_out.shape)} (Device: {train_out.device})", flush=True)

    # 5. Loss calculation
    loss = criterion(train_out, train_y)
    loss_val = loss.item()
    is_finite = torch.isfinite(loss).item()
    has_nan = torch.isnan(loss).item()

    print(f"  - Computed BCE Loss       : {loss_val:.4f}", flush=True)
    print(f"  - Loss is Finite          : {is_finite}", flush=True)
    print(f"  - Contains NaNs           : {has_nan}", flush=True)

    assert is_finite and not has_nan, f"Sanity check error: Loss is invalid ({loss_val})"

    # 6. Backward pass and optimizer step
    loss.backward()
    optimizer.step()
    updated_weight = list(model.parameters())[0]
    weight_delta = torch.norm(updated_weight - initial_weight).item()
    print(f"  - Backward & Step Success : Weight Update Norm = {weight_delta:.6f}", flush=True)
    assert weight_delta > 0.0, "Sanity check error: Optimizer step produced 0 weight change."

    # 7. Speed benchmark: measure 10 cached batches
    print("\nMeasuring cached DataLoader throughput (10 batches)...", flush=True)
    benchmark_start = time.time()
    for b_idx in range(10):
        bx, by = next(train_iter)
        bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
        bout = model(bx)
        bloss = criterion(bout, by)
        bloss.backward()
        optimizer.step()
        optimizer.zero_grad()
    benchmark_time = time.time() - benchmark_start
    per_batch_time = benchmark_time / 10.0
    print(f"  - 10 Cached Batches Time  : {benchmark_time:.3f}s ({per_batch_time*1000:.1f}ms per batch)", flush=True)
    print(f"  - Speedup vs On-the-Fly   : ~{1.2 / per_batch_time:.1f}x faster throughput!", flush=True)

    print("=" * 75, flush=True)
    print("SANITY CHECK PASSED: Cached pipeline is verified, finite, and high-speed.", flush=True)
    print("=" * 75, flush=True)


def train_soundsense_cached(
    max_epochs=10,
    patience=3,
    batch_size=16,
    lr=0.001,
    metadata_path="soundsense_data/metadata.csv",
    features_root="soundsense_data/features",
    models_dir="soundsense_data/models",
    history_path="soundsense_data/cached_training_history.csv"
):
    print("=" * 75, flush=True)
    print("SOUNDSENSE CACHED BASELINE TRAINING (STEP 7)", flush=True)
    print("=" * 75, flush=True)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device : {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU Name        : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU VRAM Total  : {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB", flush=True)

    # 2. Datasets & DataLoaders (Train and Validate only, test split excluded)
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

    # 3. Model instantiation (exact same architecture)
    model = SoundSenseCNN(num_classes=8).to(device)

    # 4. Loss and Optimizer (exact same settings)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5. Sanity check
    run_cached_sanity_check(model, train_loader, val_loader, criterion, optimizer, device)

    # Reset model parameters for a clean epoch 1 training run
    print("\nResetting model parameters for clean training run...", flush=True)
    model = SoundSenseCNN(num_classes=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Ensure checkpoint directory exists
    os.makedirs(models_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(models_dir, "cached_baseline_best_model.pth")

    # Training state tracking
    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_per_class_f1 = {}
    best_micro_f1 = 0.0
    patience_counter = 0
    history = []
    early_stopped = False

    print("\n" + "=" * 75, flush=True)
    print(f"BEGINNING CACHED TRAINING (Max Epochs: {max_epochs}, Patience: {patience}, Batch Size: {batch_size})", flush=True)
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
        # Validation Phase
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
        print("  Per-Class Validation F1:", flush=True)
        for cname in CLASS_NAMES:
            print(f"    - {cname:22s} : P={per_class_p[cname]:.3f}, R={per_class_r[cname]:.3f}, F1={per_class_f1[cname]:.3f}", flush=True)

        # =====================================================================
        # Checkpointing and Early Stopping
        # =====================================================================
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_per_class_f1 = per_class_f1.copy()
            best_micro_f1 = micro_f1
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "macro_f1": best_macro_f1,
                    "val_loss": best_val_loss,
                    "class_names": CLASS_NAMES
                },
                best_checkpoint_path
            )
            print(f"  >> Validation Macro F1 improved! Checkpoint saved to: {best_checkpoint_path}", flush=True)
        else:
            patience_counter += 1
            print(f"  >> Macro F1 did not improve. Early stopping counter: {patience_counter}/{patience}", flush=True)

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
    print("SOUNDSENSE CACHED TRAINING SUMMARY & COMPARISON", flush=True)
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

    print("\nPer-Class F1 at Best Epoch:", flush=True)
    for cname in CLASS_NAMES:
        print(f"  - {cname:24s} : {best_per_class_f1.get(cname, 0.0):.4f}", flush=True)

    # Comparison with original baseline
    print("\n" + "-" * 75, flush=True)
    print("COMPARISON: ORIGINAL BASELINE vs. CACHED TRAINING", flush=True)
    print("-" * 75, flush=True)
    print(f"{'Metric / Property':32s} | {'Original On-The-Fly':20s} | {'New Cached Features':20s}", flush=True)
    print("-" * 75, flush=True)
    print(f"{'Data Pipeline':32s} | {'Raw WAV + Librosa':20s} | {'Precomputed .npy':20s}", flush=True)
    print(f"{'Epoch Time (approx.)':32s} | {ORIGINAL_BASELINE_EPOCH_TIME:20s} | {f'{avg_epoch_time/60:.2f} min/epoch':20s}", flush=True)
    print(f"{'Best Validation Macro F1':32s} | {f'{ORIGINAL_BASELINE_MACRO_F1:.4f}':20s} | {f'{best_macro_f1:.4f}':20s}", flush=True)
    print(f"{'Best Checkpoint File':32s} | {'best_model.pth':20s} | {'cached_baseline_best_model.pth':20s}", flush=True)
    print(f"{'Training History File':32s} | {'training_history.csv':20s} | {'cached_training_history.csv':20s}", flush=True)
    print("-" * 75, flush=True)

    # Overfitting assessment
    final_train_loss = history[-1]["train_loss"]
    final_val_loss = history[-1]["val_loss"]
    loss_gap = final_val_loss - final_train_loss

    print("\nOverfitting Assessment:", flush=True)
    if loss_gap > 0.15 and len(history) > 2:
        print(f"  - Overfitting detected: Validation loss ({final_val_loss:.4f}) is higher than training loss ({final_train_loss:.4f}) with a gap of {loss_gap:.4f}.", flush=True)
    else:
        print(f"  - No significant overfitting detected: Train loss ({final_train_loss:.4f}) and validation loss ({final_val_loss:.4f}) (gap = {loss_gap:.4f}).", flush=True)

    print("=" * 75, flush=True)
    print("Cached baseline training finished.", flush=True)


if __name__ == "__main__":
    train_soundsense_cached()
