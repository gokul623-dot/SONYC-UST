"""
test_spectrogram.py

Step 4: Testing Audio to Log-Mel Spectrogram Conversion for SoundSense.

This script verifies the audio preprocessing and feature extraction pipeline on
3 representative audio clips from soundsense_data/metadata.csv:
1. Single active label (e.g. Engine only)
2. Multiple active labels (e.g. Engine + Human Voice)
3. Zero active labels (Ambient background noise)

Pipeline parameters:
- Target Sample Rate: 48,000 Hz
- FFT window size (n_fft): 2,048
- Hop length (hop_length): 512
- Mel filter banks (n_mels): 128
- Scale: Decibels (dB) via librosa.power_to_db
"""

import os
import sys

# Check for required audio and visualization libraries
missing_libs = []
try:
    import librosa
    import librosa.display
except ImportError:
    missing_libs.append("librosa")

try:
    import matplotlib.pyplot as plt
except ImportError:
    missing_libs.append("matplotlib")

if missing_libs:
    print("=" * 70)
    print("SOUNDSENSE SPECTROGRAM PIPELINE - DEPENDENCY CHECK")
    print("=" * 70)
    print(f"Error: Missing required Python package(s): {', '.join(missing_libs)}")
    print("\nTo install the missing dependencies without changing the architecture:")
    print(f"    pip install {' '.join(missing_libs)}")
    print("=" * 70)
    sys.exit(1)

import pandas as pd
import numpy as np

def find_audio_path(audio_filename, base_dir="."):
    """Locate the .wav file across audio-0 to audio-18 folders."""
    for i in range(19):
        candidate = os.path.join(base_dir, f"audio-{i}", audio_filename)
        if os.path.exists(candidate):
            return candidate
    return None

def test_spectrogram_pipeline():
    metadata_path = os.path.join("soundsense_data", "metadata.csv")
    output_dir = os.path.join("soundsense_data", "test_spectrograms")
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        sys.exit(1)
        
    df = pd.read_csv(metadata_path)
    label_cols = [
        "engine", "machinery_impact", "non_machinery_impact", "powered_saw",
        "alert_signal", "music", "human_voice", "dog"
    ]
    
    # Calculate active label count per file
    active_counts = df[label_cols].sum(axis=1)
    
    # Select 3 representative files:
    # 1. Single label
    single_label_row = df[active_counts == 1].iloc[0]
    # 2. Multiple labels
    multi_label_row = df[active_counts >= 2].iloc[0]
    # 3. Zero labels (ambience)
    zero_label_row = df[active_counts == 0].iloc[0]
    
    selected_samples = [
        ("Single Label", single_label_row, "sample_1_single_label.png"),
        ("Multiple Labels", multi_label_row, "sample_2_multi_label.png"),
        ("Zero Labels (Ambience)", zero_label_row, "sample_3_zero_labels.png"),
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("SOUNDSENSE AUDIO TO LOG-MEL SPECTROGRAM PIPELINE TEST")
    print("=" * 70)
    
    sr_target = 48000
    n_mels = 128
    n_fft = 2048
    hop_length = 512
    
    for category_name, row, out_name in selected_samples:
        fname = row["audio_filename"]
        wav_path = find_audio_path(fname)
        
        if wav_path is None or not os.path.exists(wav_path):
            print(f"Error: Audio file {fname} not found in audio folders.")
            continue
            
        # Active labels list
        active_labels = [col for col in label_cols if row[col] == 1]
        active_str = ", ".join(active_labels) if active_labels else "None (Ambient Noise)"
        
        # Load audio with librosa
        y, sr = librosa.load(wav_path, sr=sr_target)
        duration = len(y) / sr
        num_samples = len(y)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # Convert to decibels (log-mel)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        print(f"\n--- [{category_name}] ---")
        print(f"  Filename         : {fname}")
        print(f"  Path             : {wav_path}")
        print(f"  Sample Rate      : {sr} Hz")
        print(f"  Duration         : {duration:.2f} seconds")
        print(f"  Total Samples    : {num_samples:,}")
        print(f"  Active Labels    : {active_str}")
        print(f"  Spectrogram Shape: {log_mel_spec.shape} (mels x time frames)")
        print(f"  Value Range (dB) : min={log_mel_spec.min():.1f} dB, max={log_mel_spec.max():.1f} dB")
        
        # Plot and save
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            log_mel_spec,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            cmap="magma"
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(f"{category_name}: {fname}\nActive Labels: [{active_str}]", fontsize=11, fontweight="bold")
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, out_name)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved Image      : {save_path} (Exists: {os.path.exists(save_path)})")
        
    print("\n" + "=" * 70)
    print(f"Test completed. Spectrogram plots saved inside: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    test_spectrogram_pipeline()
