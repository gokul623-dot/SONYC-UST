"""
train_soundsense.py

Step 7: GPU-Accelerated Training Pipeline for SoundSense Urban Sound Tagging.

Requirements:
1. Device: Auto-select CUDA if available, otherwise CPU.
2. Data: Train (train) and Validation (validate) splits using SoundSenseDataset (batch_size=16).
3. Model: SoundSenseCNN from soundsense_model.py.
4. Loss: nn.BCEWithLogitsLoss() (no pos_weight for baseline).
5. Optimizer: Adam with learning rate = 0.001.
6. Training: Up to 10 epochs with early stopping (patience=3) based on validation Macro F1.
7. Metrics: Per-class Precision, Recall, F1; Macro F1; Micro F1.
8. Checkpoint: Save best model to soundsense_data/models/best_model.pth.
9. History: Save metrics per epoch to soundsense_data/training_history.csv.
10. Sanity check: Run GPU sanity check before beginning full training.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from soundsense_dataset import SoundSenseDataset
from soundsense_model import SoundSenseCNN

# Exact 8 coarse class names in taxonomy order
CLASS_NAMES = [
    "engine",
    "machinery-impact",
    "non-machinery-impact",
    "powered-saw",
    "alert-signal",
    "music",
    "human-voice",
    "dog"
]


def compute_metrics(y_true, y_pred):
    """
    Computes per-class precision, recall, F1, macro F1, and micro F1.
    
    Args:
        y_true (np.ndarray): Binary ground-truth matrix of shape [N, 8].
        y_pred (np.ndarray): Binary prediction matrix of shape [N, 8].
        
    Returns:
        tuple: (per_class_precision, per_class_recall, per_class_f1, macro_f1, micro_f1)
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

    # Macro F1 is the unweighted average across all 8 classes
    macro_f1 = float(np.mean(list(per_class_f1.values())))

    # Micro F1 aggregates all true positives, false positives, and false negatives globally
    total_tp = np.sum((y_true == 1) & (y_pred == 1))
    total_fp = np.sum((y_true == 0) & (y_pred == 1))
    total_fn = np.sum((y_true == 1) & (y_pred == 0))

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return per_class_p, per_class_r, per_class_f1, macro_f1, float(micro_f1)


def run_gpu_sanity_check(model, train_loader, criterion, optimizer, device):
    """
    Pre-flight GPU sanity check:
    - Loads one training batch.
    - Moves it to CUDA (or selected device).
    - Runs forward pass.
    - Calculates BCEWithLogitsLoss.
    - Runs backward().
    - Runs optimizer.step().
    - Confirms model output is on CUDA.
    - Confirms loss is finite.
    """
    print("=" * 75, flush=True)
    print("GPU PRE-TRAINING SANITY CHECK", flush=True)
    print("=" * 75, flush=True)

    model.train()
    train_iter = iter(train_loader)
    
    # 1. Load one training batch
    specs, targets = next(train_iter)
    print(f"  - Loaded Batch Spectrogram Shape : {list(specs.shape)}", flush=True)
    print(f"  - Loaded Batch Target Shape      : {list(targets.shape)}", flush=True)

    # 2. Move to device
    specs = specs.to(device)
    targets = targets.to(device)
    print(f"  - Moved Tensors to Device        : {specs.device}", flush=True)

    # Record parameter snapshot to verify update
    initial_param = list(model.parameters())[0].clone()

    # 3. Forward pass
    optimizer.zero_grad()
    outputs = model(specs)
    print(f"  - Forward Pass Output Shape      : {list(outputs.shape)}", flush=True)
    print(f"  - Model Output Device            : {outputs.device}", flush=True)

    # Confirm output is on CUDA (if CUDA enabled)
    if device.type == "cuda":
        assert outputs.is_cuda, "Sanity Check Error: Model output is NOT on CUDA!"
        print(f"  - CUDA Device Verification       : CONFIRMED (is_cuda=True, device={outputs.device})", flush=True)

    # 4. Calculate BCEWithLogitsLoss
    loss = criterion(outputs, targets)
    loss_val = loss.item()
    is_finite = torch.isfinite(loss).item()
    print(f"  - Calculated BCE Loss            : {loss_val:.4f}", flush=True)
    print(f"  - Loss is Finite                 : {is_finite}", flush=True)
    if not is_finite:
        raise ValueError(f"Sanity Check Error: Loss is NaN or Inf ({loss_val})")

    # 5. Backward pass
    loss.backward()
    print("  - Backward Pass                  : SUCCESS (Gradients computed)", flush=True)

    # 6. Optimizer step
    optimizer.step()
    updated_param = list(model.parameters())[0]
    weight_diff = torch.norm(updated_param - initial_param).item()
    print(f"  - Optimizer Step                 : SUCCESS (Weight Delta Norm = {weight_diff:.6f})", flush=True)
    if weight_diff == 0.0:
        raise RuntimeError("Sanity Check Error: Optimizer did not update model parameters.")

    print("=" * 75, flush=True)
    print("GPU SANITY CHECK PASSED: Device tensors, loss computation, and gradients verified.", flush=True)
    print("=" * 75, flush=True)


def train_soundsense(
    max_epochs=10,
    patience=3,
    batch_size=16,
    lr=0.001,
    models_dir="soundsense_data/models",
    history_path="soundsense_data/training_history.csv"
):
    print("=" * 75, flush=True)
    print("SOUNDSENSE ML GPU TRAINING PIPELINE (STEP 7)", flush=True)
    print("=" * 75, flush=True)

    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device : {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU Name        : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU VRAM Total  : {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB", flush=True)

    # 2. DataLoaders (Train and Validate only, test split excluded)
    print("\nLoading Datasets...", flush=True)
    train_dataset = SoundSenseDataset(split="train")
    val_dataset = SoundSenseDataset(split="validate")

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

    # 3. Model instantiation on selected device
    model = SoundSenseCNN(num_classes=8).to(device)

    # 4. Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5. Sanity Check
    run_gpu_sanity_check(model, train_loader, criterion, optimizer, device)

    # Reset model and optimizer for fresh training from epoch 1
    print("\nResetting model parameters for clean training run...", flush=True)
    model = SoundSenseCNN(num_classes=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Ensure model checkpoint directory exists
    os.makedirs(models_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(models_dir, "best_model.pth")

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
    print(f"BEGINNING TRAINING (Max Epochs: {max_epochs}, Patience: {patience}, Batch Size: {batch_size})", flush=True)
    print("=" * 75, flush=True)

    start_training_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()

        # =====================================================================
        # Training Phase
        # =====================================================================
        model.train()
        train_loss_total = 0.0
        train_batches_count = 0

        print(f"\n>>> Epoch {epoch}/{max_epochs} [Training on {device}]", flush=True)
        batch_log_interval = 100

        for batch_idx, (specs, targets) in enumerate(train_loader, start=1):
            specs = specs.to(device, non_blocking=use_pin_memory)
            targets = targets.to(device, non_blocking=use_pin_memory)

            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_batches_count += 1

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

                # Apply sigmoid only for metric computation
                probs = torch.sigmoid(logits)

                all_val_targets.append(targets.cpu().numpy())
                all_val_probs.append(probs.cpu().numpy())

        avg_val_loss = val_loss_total / val_batches_count

        # Concatenate full validation predictions
        y_true = np.vstack(all_val_targets)
        y_probs = np.vstack(all_val_probs)
        y_pred = (y_probs >= 0.5).astype(int)

        # Compute metrics
        per_class_p, per_class_r, per_class_f1, macro_f1, micro_f1 = compute_metrics(y_true, y_pred)

        epoch_duration = time.time() - epoch_start_time

        # Save history entry
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

        # Write training history CSV after each epoch
        history_df = pd.DataFrame(history)
        history_df.to_csv(history_path, index=False)

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
            improvement = macro_f1 - (best_macro_f1 if best_macro_f1 > 0 else 0)
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
            print(f"  >> Validation Macro F1 improved! Model checkpoint saved to: {best_checkpoint_path}", flush=True)
        else:
            patience_counter += 1
            print(f"  >> Macro F1 did not improve. Early stopping counter: {patience_counter}/{patience}", flush=True)

            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation Macro F1 has not improved for {patience} consecutive epochs.", flush=True)
                early_stopped = True
                break

    total_training_duration = time.time() - start_training_time

    # =========================================================================
    # Final Comprehensive Summary
    # =========================================================================
    avg_epoch_time = total_training_duration / len(history) if history else 0.0

    print("\n" + "=" * 75, flush=True)
    print("SOUNDSENSE TRAINING SUMMARY", flush=True)
    print("=" * 75, flush=True)
    print(f"Device / GPU                      : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Time per Epoch (average)          : {avg_epoch_time:.1f}s ({avg_epoch_time/60:.2f} min)", flush=True)
    print(f"Total Training Duration           : {total_training_duration/60:.2f} minutes", flush=True)
    print(f"Epochs Actually Completed         : {len(history)} of {max_epochs}", flush=True)
    print(f"Best Epoch                        : Epoch {best_epoch}", flush=True)
    print(f"Best Validation Macro F1          : {best_macro_f1:.4f}", flush=True)
    print(f"Best Validation Micro F1          : {best_micro_f1:.4f}", flush=True)
    print(f"Best Validation Loss              : {best_val_loss:.4f}", flush=True)
    print(f"Saved Checkpoint Path             : {best_checkpoint_path}", flush=True)
    print(f"Early Stopping Occurred           : {early_stopped}", flush=True)

    print("\nPer-Class F1 at Best Epoch:", flush=True)
    for cname in CLASS_NAMES:
        print(f"  - {cname:24s} : {best_per_class_f1.get(cname, 0.0):.4f}", flush=True)

    # Overfitting assessment
    if history:
        final_train_loss = history[-1]["train_loss"]
        final_val_loss = history[-1]["val_loss"]
        loss_gap = final_val_loss - final_train_loss

        print("\nOverfitting Assessment:", flush=True)
        if loss_gap > 0.15 and len(history) > 2:
            print(f"  - Signs of overfitting detected: Validation loss ({final_val_loss:.4f}) is noticeably higher than training loss ({final_train_loss:.4f}) with a gap of {loss_gap:.4f}.", flush=True)
        else:
            print(f"  - No significant overfitting detected: Train loss ({final_train_loss:.4f}) and validation loss ({final_val_loss:.4f}) remain well-aligned (gap = {loss_gap:.4f}).", flush=True)

    print("=" * 75, flush=True)
    print("Training pipeline finished.", flush=True)


if __name__ == "__main__":
    train_soundsense()
