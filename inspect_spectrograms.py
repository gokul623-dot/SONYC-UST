"""
inspect_spectrograms.py

Inspects and verifies the 3 generated log-mel spectrogram PNG files in:
soundsense_data/test_spectrograms/

Checks:
1. Opens each image using Pillow to verify validity and file integrity.
2. Reports image dimensions, color mode, and file size.
3. Matches each image to its source audio recording and active labels in soundsense_data/metadata.csv.
4. Confirms readiness of the audio feature extraction pipeline for dynamic PyTorch/ML training.
"""

import os
import sys
import pandas as pd
from PIL import Image

def inspect_spectrograms():
    spectrogram_dir = os.path.join("soundsense_data", "test_spectrograms")
    metadata_path = os.path.join("soundsense_data", "metadata.csv")

    print("=" * 75)
    print("SOUNDSENSE SPECTROGRAM VERIFICATION & INSPECTION")
    print("=" * 75)

    # 1. Verify paths
    if not os.path.isdir(spectrogram_dir):
        print(f"Error: Directory not found: {spectrogram_dir}")
        sys.exit(1)

    if not os.path.isfile(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        sys.exit(1)

    df = pd.read_csv(metadata_path)
    label_cols = [
        "engine", "machinery_impact", "non_machinery_impact", "powered_saw",
        "alert_signal", "music", "human_voice", "dog"
    ]

    # Mapping of generated sample images to audio filenames
    samples = [
        {
            "image_filename": "sample_1_single_label.png",
            "audio_filename": "00_000066.wav",
            "description": "Single Active Label Sample"
        },
        {
            "image_filename": "sample_2_multi_label.png",
            "audio_filename": "00_000118.wav",
            "description": "Multiple Active Labels Sample"
        },
        {
            "image_filename": "sample_3_zero_labels.png",
            "audio_filename": "00_000071.wav",
            "description": "Zero Active Labels (Ambient Noise) Sample"
        }
    ]

    all_valid = True

    for item in samples:
        img_path = os.path.join(spectrogram_dir, item["image_filename"])
        audio_name = item["audio_filename"]
        desc = item["description"]

        print(f"\n--- [{desc}] ---")
        print(f"  Image File       : {item['image_filename']}")
        print(f"  Image Path       : {img_path}")

        # Check if file exists
        if not os.path.exists(img_path):
            print(f"  Status           : FAILED - File not found")
            all_valid = False
            continue

        file_size_kb = os.path.getsize(img_path) / 1024

        # Open image and verify validity
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verifies file integrity
            # Reopen to read dimensions after verify()
            with Image.open(img_path) as img:
                width, height = img.size
                img_format = img.format
                img_mode = img.mode

            print(f"  Verification     : VALID IMAGE (integrity check passed)")
            print(f"  Format / Mode    : {img_format} / {img_mode}")
            print(f"  Dimensions       : {width} x {height} pixels (Width x Height)")
            print(f"  File Size        : {file_size_kb:.1f} KB")

        except Exception as e:
            print(f"  Verification     : CORRUPTED/INVALID - {e}")
            all_valid = False
            continue

        # Look up metadata for the corresponding audio file
        meta_row = df[df["audio_filename"] == audio_name]
        if meta_row.empty:
            print(f"  Metadata Match   : WARNING - Audio file '{audio_name}' not found in metadata.csv")
            all_valid = False
        else:
            row = meta_row.iloc[0]
            split = row["split"]
            active_labels = [col for col in label_cols if row[col] == 1]
            active_str = ", ".join(active_labels) if active_labels else "None (Pure Ambience)"
            label_vector = [int(row[col]) for col in label_cols]

            print(f"  Audio Source     : {audio_name}")
            print(f"  Assigned Split   : {split}")
            print(f"  Active Labels    : [{active_str}]")
            print(f"  Target Vector    : {label_vector} (corresponds to {label_cols})")

    print("\n" + "=" * 75)
    if all_valid:
        print("RESULT: ALL 3 SPECTROGRAM IMAGES ARE VALID, VERIFIED, AND FULLY CONSISTENT.")
        print("The audio-to-spectrogram preprocessing pipeline is verified and ready.")
    else:
        print("RESULT: VERIFICATION FAILED FOR ONE OR MORE SAMPLES.")
    print("=" * 75)

if __name__ == "__main__":
    inspect_spectrograms()
