"""
precompute_features.py

Precomputes log-mel spectrogram features from raw SONYC-UST audio files and saves
them as compact .npy files organized by split (train, validate, test).

This eliminates the on-the-fly audio loading and STFT bottleneck, accelerating
training epochs from ~11 minutes to seconds.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import librosa

# Exact taxonomy class names in order
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

# Exact feature parameters matching SoundSenseDataset
SAMPLE_RATE = 48000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
EXPECTED_SAMPLES = SAMPLE_RATE * 10
EXPECTED_SHAPE = (N_MELS, 938)


def build_audio_index(audio_root="."):
    """Builds a lookup dictionary mapping audio filename to its folder path."""
    audio_map = {}
    for i in range(19):
        folder = os.path.join(audio_root, f"audio-{i}")
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".wav"):
                    audio_map[fname] = os.path.join(folder, fname)
    return audio_map


def compute_log_mel_spectrogram(wav_path):
    """
    Computes log-mel spectrogram exactly matching soundsense_dataset.py:
    1. Load audio at 48,000 Hz mono
    2. Pad/truncate to exactly 10.0 seconds (480,000 samples)
    3. Compute mel spectrogram (n_fft=2048, hop_length=512, n_mels=128)
    4. Convert to dB using librosa.power_to_db(ref=np.max)
    """
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    if len(y) < EXPECTED_SAMPLES:
        y = np.pad(y, (0, EXPECTED_SAMPLES - len(y)), mode="constant")
    elif len(y) > EXPECTED_SAMPLES:
        y = y[:EXPECTED_SAMPLES]

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel.astype(np.float32)


def get_feature_path(features_root, split, audio_filename):
    """Generates deterministic .npy path for a given split and audio filename."""
    base_name = os.path.splitext(audio_filename)[0] + ".npy"
    return os.path.join(features_root, split, base_name)


def run_small_verification(
    metadata_path="soundsense_data/metadata.csv",
    audio_root=".",
    features_root="soundsense_data/features",
    samples_per_split=2
):
    """
    Extracts, saves, reloads, and verifies a small sample of audio clips (5-10 files)
    before full-scale feature extraction.
    """
    print("=" * 75, flush=True)
    print("STEP 3: SMALL VERIFICATION TEST (PRECOMPUTE FEATURES)", flush=True)
    print("=" * 75, flush=True)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    audio_map = build_audio_index(audio_root)

    # Select samples from each split
    test_samples = []
    for split in ["train", "validate", "test"]:
        sub_df = df[df["split"] == split].head(samples_per_split)
        for _, row in sub_df.iterrows():
            test_samples.append((split, row["audio_filename"]))

    print(f"Selected {len(test_samples)} verification clips across splits:", flush=True)
    for s, f in test_samples:
        print(f"  - [{s:8s}] {f}", flush=True)

    # Ensure directories exist
    for split in ["train", "validate", "test"]:
        os.makedirs(os.path.join(features_root, split), exist_ok=True)

    verification_results = []
    all_passed = True

    print("\nProcessing and verifying sample features...", flush=True)
    for split, fname in test_samples:
        wav_path = audio_map.get(fname)
        if not wav_path or not os.path.exists(wav_path):
            raise FileNotFoundError(f"Source audio {fname} not found on disk.")

        feat_path = get_feature_path(features_root, split, fname)

        # 1. Compute feature
        log_mel = compute_log_mel_spectrogram(wav_path)

        # 2. Save feature to disk
        np.save(feat_path, log_mel)

        # 3. Reload feature
        reloaded = np.load(feat_path)

        # 4. Verification checks
        shape_ok = (reloaded.shape == EXPECTED_SHAPE)
        dtype_ok = (reloaded.dtype == np.float32)
        finite_ok = bool(np.all(np.isfinite(reloaded)))
        exact_match = bool(np.allclose(log_mel, reloaded, atol=1e-6))
        file_size_bytes = os.path.getsize(feat_path)

        passed = shape_ok and dtype_ok and finite_ok and exact_match
        if not passed:
            all_passed = False

        verification_results.append({
            "split": split,
            "filename": fname,
            "feature_path": feat_path,
            "shape": reloaded.shape,
            "dtype": str(reloaded.dtype),
            "size_kb": round(file_size_bytes / 1024, 1),
            "finite": finite_ok,
            "exact_match": exact_match,
            "passed": passed
        })

    print("\n" + "-" * 75, flush=True)
    print(f"{'Split':10s} | {'Audio Filename':14s} | {'Shape':12s} | {'Dtype':9s} | {'Size':8s} | {'Finite':6s} | {'Match':6s}", flush=True)
    print("-" * 75, flush=True)
    for res in verification_results:
        print(
            f"{res['split']:10s} | {res['filename']:14s} | {str(res['shape']):12s} | {res['dtype']:9s} | "
            f"{res['size_kb']} KB | {str(res['finite']):6s} | {str(res['exact_match']):6s}",
            flush=True
        )
    print("-" * 75, flush=True)

    if all_passed:
        print("\nALL VERIFICATION CHECKS PASSED: Feature shape, dtype, and numerical fidelity confirmed.", flush=True)
    else:
        print("\nVERIFICATION FAILED for one or more clips!", flush=True)
        sys.exit(1)

    return all_passed


def precompute_all_features(
    metadata_path="soundsense_data/metadata.csv",
    audio_root=".",
    features_root="soundsense_data/features",
    log_interval=500
):
    """
    Extracts and caches features for all recordings in metadata.csv.
    Safely resumes if interrupted by checking if the feature already exists.
    """
    print("=" * 75, flush=True)
    print("SOUNDSENSE BULK FEATURE EXTRACTION", flush=True)
    print("=" * 75, flush=True)

    df = pd.read_csv(metadata_path)
    total_files = len(df)
    audio_map = build_audio_index(audio_root)

    for split in ["train", "validate", "test"]:
        os.makedirs(os.path.join(features_root, split), exist_ok=True)

    print(f"Total recordings in metadata: {total_files:,}", flush=True)
    for split in ["train", "validate", "test"]:
        cnt = (df["split"] == split).sum()
        print(f"  - {split.capitalize():10s}: {cnt:,} files", flush=True)

    t0 = time.time()
    skipped_count = 0
    computed_count = 0
    failed_files = []

    for idx, row in df.iterrows():
        fname = row["audio_filename"]
        split = row["split"]
        feat_path = get_feature_path(features_root, split, fname)

        # Skip if already exists and valid
        if os.path.exists(feat_path) and os.path.getsize(feat_path) > 1000:
            skipped_count += 1
            continue

        wav_path = audio_map.get(fname)
        if not wav_path or not os.path.exists(wav_path):
            failed_files.append((fname, "WAV not found"))
            continue

        try:
            log_mel = compute_log_mel_spectrogram(wav_path)
            np.save(feat_path, log_mel)
            computed_count += 1
        except Exception as e:
            failed_files.append((fname, str(e)))

        processed_so_far = computed_count + skipped_count
        if processed_so_far % log_interval == 0 or processed_so_far == total_files:
            elapsed = time.time() - t0
            print(f"  Processed {processed_so_far:,}/{total_files:,} ({processed_so_far/total_files*100:5.1f}%) | "
                  f"New: {computed_count:,} | Skipped: {skipped_count:,} | Elapsed: {elapsed:.1f}s", flush=True)

    total_time = time.time() - t0
    
    # Calculate total disk space used by features
    total_bytes = 0
    for root, _, files in os.walk(features_root):
        for f in files:
            if f.endswith(".npy"):
                total_bytes += os.path.getsize(os.path.join(root, f))
    total_gb = total_bytes / (1024 ** 3)
    total_mb = total_bytes / (1024 ** 2)

    print("\n" + "=" * 75, flush=True)
    print("FEATURE EXTRACTION SUMMARY", flush=True)
    print("=" * 75, flush=True)
    print(f"Total Files in Metadata   : {total_files:,}", flush=True)
    print(f"Newly Computed Features   : {computed_count:,}", flush=True)
    print(f"Skipped (Already Cached)  : {skipped_count:,}", flush=True)
    print(f"Failed Files              : {len(failed_files)}", flush=True)
    print(f"Total Processing Time     : {total_time/60:.2f} minutes ({total_time:.1f}s)", flush=True)
    print(f"Total Disk Space Used     : {total_gb:.2f} GB ({total_mb:,.1f} MB)", flush=True)

    # Verification of counts in directories
    print("\nSplit Count Verification:", flush=True)
    for split in ["train", "validate", "test"]:
        expected_cnt = (df["split"] == split).sum()
        actual_cnt = len([f for f in os.listdir(os.path.join(features_root, split)) if f.endswith(".npy")])
        match_str = "MATCH" if expected_cnt == actual_cnt else "MISMATCH"
        print(f"  - [{split:8s}] Expected: {expected_cnt:,} | Cached on disk: {actual_cnt:,} ({match_str})", flush=True)

    # Verification of random sample features
    print("\nRandom Sample Verification:", flush=True)
    import random
    rng = random.Random(42)
    sample_checks_passed = True
    for split in ["train", "validate", "test"]:
        split_dir = os.path.join(features_root, split)
        npy_files = [f for f in os.listdir(split_dir) if f.endswith(".npy")]
        if not npy_files:
            continue
        sampled = rng.sample(npy_files, min(3, len(npy_files)))
        for sf in sampled:
            fp = os.path.join(split_dir, sf)
            arr = np.load(fp)
            s_ok = (arr.shape == EXPECTED_SHAPE)
            d_ok = (arr.dtype == np.float32)
            f_ok = bool(np.all(np.isfinite(arr)))
            if not (s_ok and d_ok and f_ok):
                sample_checks_passed = False
            print(f"  - [{split:8s}] {sf:14s} | Shape={arr.shape} | Dtype={arr.dtype} | Finite={f_ok}", flush=True)

    print(f"\nRandom Sample Integrity Check : {'ALL PASSED' if sample_checks_passed else 'FAILED'}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute SoundSense log-mel spectrogram features")
    parser.add_argument("--verify", action="store_true", help="Run small verification test only")
    parser.add_argument("--samples", type=int, default=3, help="Samples per split for small verification")
    args = parser.parse_args()

    if args.verify:
        run_small_verification(samples_per_split=args.samples)
    else:
        precompute_all_features()

