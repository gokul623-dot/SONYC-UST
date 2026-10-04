"""
prepare_soundsense_dataset.py

Step 3: Preparing the SoundSense ML Dataset from SONYC-UST.

This script:
1. Loads annotations.csv without modifying any original SONYC-UST dataset files.
2. Extracts the 8 coarse presence columns.
3. Groups annotations by unique audio_filename.
4. Computes binary ground-truth labels using the DCASE consensus/majority standard:
   - If an annotator_id == 0 (expert verified ground truth) row exists, use it.
   - Otherwise, take the majority vote of citizen science volunteers (annotator_id > 0, >= 2 out of 3 votes).
5. Preserves the official train, validate, and test splits.
6. Saves the clean metadata to soundsense_data/metadata.csv with standardized column names:
   [audio_filename, split, engine, machinery_impact, non_machinery_impact, powered_saw, alert_signal, music, human_voice, dog]
7. Prints a comprehensive validation summary.
"""

import os
import sys
import pandas as pd
import numpy as np

def prepare_soundsense_dataset(
    annotations_path="annotations.csv",
    output_dir="soundsense_data",
    output_filename="metadata.csv"
):
    print("=" * 70)
    print("SOUNDSENSE ML DATASET PREPARATION")
    print("=" * 70)

    # 1. Validate input file
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found at: {annotations_path}")
        sys.exit(1)

    print(f"Loading annotations from: {annotations_path}")
    df = pd.read_csv(annotations_path)
    print(f"Total annotation rows loaded: {len(df):,}")

    # 2. Define source columns and target standardized names
    column_mapping = {
        "1_engine_presence": "engine",
        "2_machinery-impact_presence": "machinery_impact",
        "3_non-machinery-impact_presence": "non_machinery_impact",
        "4_powered-saw_presence": "powered_saw",
        "5_alert-signal_presence": "alert_signal",
        "6_music_presence": "music",
        "7_human-voice_presence": "human_voice",
        "8_dog_presence": "dog",
    }
    coarse_cols = list(column_mapping.keys())
    target_labels = list(column_mapping.values())

    # Verify coarse columns exist in dataframe
    for col in coarse_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in {annotations_path}")

    # 3. Preserve split mapping per audio file
    print("Mapping audio files to their designated splits...")
    split_map = df.groupby("audio_filename")["split"].first()
    unique_files = sorted(df["audio_filename"].unique())
    print(f"Total unique audio recordings: {len(unique_files):,}")

    # 4. Extract verified ground truth (annotator_id == 0)
    verified_df = df[df["annotator_id"] == 0]
    verified_files = set(verified_df["audio_filename"])
    print(f"Audio files with verified ground truth (annotator_id == 0): {len(verified_files):,}")

    # Verify no duplicate rows exist for verified annotations
    if verified_df.duplicated(subset=["audio_filename"]).any():
        print("Warning: Duplicate verified rows detected. Keeping first.")
        verified_df = verified_df.drop_duplicates(subset=["audio_filename"])
    
    verified_indexed = verified_df.set_index("audio_filename")[coarse_cols]

    # 5. Extract crowd annotations (annotator_id > 0) and compute majority vote
    crowd_df = df[df["annotator_id"] > 0]
    print(f"Processing crowd volunteer annotations (rows: {len(crowd_df):,})...")
    
    # Majority vote: >= 2 out of 3 annotators (mean >= 0.5)
    crowd_mean = crowd_df.groupby("audio_filename")[coarse_cols].mean()
    crowd_majority = (crowd_mean >= 0.5).astype(int)

    # 6. Combine: Start with crowd majority, override with verified ground-truth where available
    clean_labels = crowd_majority.copy()
    clean_labels.update(verified_indexed.astype(int))

    # Rename to clean target column names
    clean_labels = clean_labels.rename(columns=column_mapping)
    clean_labels = clean_labels.astype(int)

    # Attach split column
    clean_labels["split"] = split_map
    clean_labels = clean_labels.reset_index()

    # Reorder columns to exact requested format
    final_cols = ["audio_filename", "split"] + target_labels
    clean_df = clean_labels[final_cols].sort_values("audio_filename").reset_index(drop=True)

    # 7. Output directory and save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    clean_df.to_csv(out_path, index=False)
    print(f"\nClean metadata saved to: {out_path}")

    # 8. Print comprehensive summary
    print("\n" + "=" * 70)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 70)
    
    total_recordings = len(clean_df)
    print(f"Total Unique Recordings: {total_recordings:,}")
    
    # Split counts
    print("\n[Data Splits]")
    split_counts = clean_df["split"].value_counts()
    for s in ["train", "validate", "test"]:
        cnt = split_counts.get(s, 0)
        pct = (cnt / total_recordings) * 100
        print(f"  - {s.capitalize():10s}: {cnt:5d} ({pct:5.1f}%)")

    # Positive count per label
    print("\n[Positive Label Counts & Prevalence]")
    for lbl in target_labels:
        pos_cnt = (clean_df[lbl] == 1).sum()
        pos_pct = (pos_cnt / total_recordings) * 100
        print(f"  - {lbl:22s}: {pos_cnt:5d} positive ({pos_pct:5.1f}%)")

    # Label co-occurrence analysis
    active_labels_per_file = clean_df[target_labels].sum(axis=1)
    zero_label_count = (active_labels_per_file == 0).sum()
    multi_label_count = (active_labels_per_file >= 2).sum()
    single_label_count = (active_labels_per_file == 1).sum()

    print("\n[Label Co-occurrence & Ambience Breakdown]")
    print(f"  - Zero labels (pure background ambience) : {zero_label_count:5d} ({zero_label_count/total_recordings*100:5.1f}%)")
    print(f"  - Exactly one active label               : {single_label_count:5d} ({single_label_count/total_recordings*100:5.1f}%)")
    print(f"  - Multiple active labels (>= 2 classes)  : {multi_label_count:5d} ({multi_label_count/total_recordings*100:5.1f}%)")

    print("\n[Verification Check]")
    null_count = clean_df.isnull().sum().sum()
    print(f"  - Total missing / NaN values: {null_count}")
    print(f"  - Output shape: {clean_df.shape[0]} rows x {clean_df.shape[1]} columns")
    print("=" * 70)
    print("SoundSense ML dataset preparation complete successfully.")

if __name__ == "__main__":
    prepare_soundsense_dataset()
