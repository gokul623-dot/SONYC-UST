"""
app.py

SoundSense – Environmental Sound AI
Interactive Streamlit application for multi-label urban sound detection.

Loads the trained class-weighted SoundSenseCNN checkpoint and applies frozen
validation-optimized decision thresholds to detect sound events in user-uploaded
WAV audio recordings.
"""

import os
import io
import json
import numpy as np
import pandas as pd
import librosa
import torch
import streamlit as st

from soundsense_model import SoundSenseCNN

# Exact 8 coarse classes in DCASE taxonomy order
CLASS_KEYS = [
    "engine",
    "machinery_impact",
    "non_machinery_impact",
    "powered_saw",
    "alert_signal",
    "music",
    "human_voice",
    "dog"
]

CLASS_DISPLAY_NAMES = {
    "engine": "Engine",
    "machinery_impact": "Machinery Impact",
    "non_machinery_impact": "Non-Machinery Impact",
    "powered_saw": "Powered Saw",
    "alert_signal": "Alert Signal",
    "music": "Music",
    "human_voice": "Human Voice",
    "dog": "Dog"
}

# Exact preprocessing parameters matching training & cached feature generation
SAMPLE_RATE = 48000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
TARGET_DURATION_SEC = 10.0
EXPECTED_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION_SEC)  # 480,000 samples
EXPECTED_FRAMES = 938  # 1 + 480000 // 512

CHECKPOINT_PATH = os.path.join("soundsense_data", "models", "cached_baseline_best_model.pth")
THRESHOLDS_PATH = os.path.join("soundsense_data", "optimized_thresholds.json")


@st.cache_resource
def load_model():
    """Loads the trained SoundSenseCNN checkpoint in eval mode."""
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at: '{CHECKPOINT_PATH}'. "
            f"Please verify that the trained model file exists."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    model = SoundSenseCNN(num_classes=8).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "N/A")
    return model, device, epoch


@st.cache_data
def load_thresholds():
    """Loads the frozen validation-optimized decision thresholds."""
    if not os.path.exists(THRESHOLDS_PATH):
        raise FileNotFoundError(
            f"Optimized thresholds file not found at: '{THRESHOLDS_PATH}'."
        )

    with open(THRESHOLDS_PATH, "r") as f:
        thresholds = json.load(f)

    # Validate that all 8 classes are present
    missing = [c for c in CLASS_KEYS if c not in thresholds]
    if missing:
        raise ValueError(f"Thresholds file is missing required classes: {missing}")

    return thresholds


def preprocess_audio(file_bytes):
    """
    Preprocesses uploaded WAV audio using the exact training configuration:
    1. Mono channel at 48,000 Hz
    2. Pad (zeros) or truncate to exactly 10.0 seconds (480,000 samples)
    3. Log-mel spectrogram (n_fft=2048, hop_length=512, n_mels=128, power_to_db with ref=np.max)
    4. Formats tensor to shape [1, 1, 128, 938]
    """
    # 1. Load audio with librosa at 48 kHz mono
    try:
        y, orig_sr = librosa.load(io.BytesIO(file_bytes), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to decode audio file. Please ensure it is a valid WAV recording. Error: {str(e)}")

    orig_duration = len(y) / float(SAMPLE_RATE)

    # 2. Pad or truncate to exactly 10 seconds
    if len(y) < EXPECTED_SAMPLES:
        y = np.pad(y, (0, EXPECTED_SAMPLES - len(y)), mode="constant")
    elif len(y) > EXPECTED_SAMPLES:
        y = y[:EXPECTED_SAMPLES]

    # 3. Compute Mel-scaled spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # 4. Decibel scaling
    log_mel = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)

    # 5. Convert to PyTorch tensor [1, 1, 128, 938]
    input_tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 938]

    return input_tensor, orig_duration, log_mel


def generate_context_summary(detected_classes):
    """
    Deterministic rule-based summary generation based strictly on detected classes.
    Does not hallucinate or claim beyond detected sound events.
    """
    if not detected_classes:
        return "No target sound detected above the model thresholds."

    det_set = set(detected_classes)

    # Specific pairs from specifications
    if det_set == {"engine", "human_voice"}:
        return "Engine-related activity and human presence detected."
    if det_set == {"alert_signal", "human_voice"}:
        return "Alert signal and human voice activity detected."
    if det_set == {"engine"}:
        return "Engine or traffic-related activity detected."
    if det_set == {"human_voice"}:
        return "Human voice activity detected."
    if det_set == {"music"}:
        return "Music detected."
    if det_set == {"dog"}:
        return "Dog sound detected."
    if det_set == {"alert_signal"}:
        return "Possible alert or warning signal detected."
    if det_set == {"machinery_impact"}:
        return "Machinery impact or mechanical noise detected."
    if det_set == {"non_machinery_impact"}:
        return "Non-machinery impact or physical contact sound detected."
    if det_set == {"powered_saw"}:
        return "Powered saw or cutting equipment sound detected."

    # General deterministic multi-label combinations
    descriptions = {
        "engine": "engine/traffic activity",
        "machinery_impact": "machinery impact noise",
        "non_machinery_impact": "non-machinery impact sound",
        "powered_saw": "powered saw sound",
        "alert_signal": "alert/warning signal",
        "music": "music playback",
        "human_voice": "human voice presence",
        "dog": "dog sound"
    }

    elements = [descriptions[k] for k in CLASS_KEYS if k in det_set]
    if len(elements) == 2:
        return f"{elements[0].capitalize()} and {elements[1]} detected."
    else:
        joined = ", ".join(elements[:-1]) + f", and {elements[-1]}"
        return f"{joined.capitalize()} detected."


def main():
    st.set_page_config(
        page_title="SoundSense – Environmental Sound AI",
        page_icon="🔊",
        layout="centered"
    )

    st.title("SoundSense – Environmental Sound AI")
    st.markdown(
        "Multi-label urban acoustic sound event classification powered by a custom "
        "2D Convolutional Neural Network trained on SONYC-UST."
    )

    # --- Sidebar: System Configuration & Model Status ---
    with st.sidebar:
        st.header("Model & System Info")
        try:
            model, device, epoch = load_model()
            thresholds = load_thresholds()
            st.success(f"Model loaded (Epoch {epoch})")
            st.caption(f"Hardware Device: `{device}`")
        except Exception as e:
            st.error(f"Initialization Error: {e}")
            st.stop()

        st.subheader("Frozen Decision Thresholds")
        threshold_df = pd.DataFrame([
            {"Class": CLASS_DISPLAY_NAMES[k], "Threshold": f"{thresholds[k]:.2f}"}
            for k in CLASS_KEYS
        ])
        st.table(threshold_df)

        st.caption(
            "Preprocessing: 48 kHz, Mono, 10s Window, 128 Mel Bins, 2048 FFT."
        )

    # --- Step 1: File Upload ---
    st.subheader("1. Upload Audio Recording")
    uploaded_file = st.file_uploader(
        "Select a 10-second WAV file (e.g. from SONYC-UST dataset)",
        type=["wav"],
        help="Upload standard uncompressed WAV audio."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        # Step 2: Audio Player
        st.subheader("2. Audio Playback")
        st.audio(file_bytes, format="audio/wav")

        # Step 3: Analyze Button
        st.subheader("3. Acoustic Analysis")
        analyze_clicked = st.button("Analyze Audio", type="primary", use_container_width=True)

        if analyze_clicked:
            with st.spinner("Preprocessing audio (48 kHz log-mel spectrogram) and running CNN inference..."):
                try:
                    # Preprocess
                    input_tensor, duration, log_mel = preprocess_audio(file_bytes)

                    # Move to model device
                    input_tensor = input_tensor.to(device)

                    # Inference with torch.no_grad()
                    with torch.no_grad():
                        logits = model(input_tensor)
                        probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)  # [8]

                    # Parse results
                    detected_keys = []
                    results_data = []

                    for i, key in enumerate(CLASS_KEYS):
                        score = float(probs[i])
                        thresh = float(thresholds[key])
                        is_detected = score >= thresh

                        if is_detected:
                            detected_keys.append(key)

                        results_data.append({
                            "Class": CLASS_DISPLAY_NAMES[key],
                            "Score": score,
                            "Threshold": thresh,
                            "Status": "Detected" if is_detected else "Not Detected"
                        })

                except Exception as ex:
                    st.error(f"Inference error: {str(ex)}")
                    return

            st.success("Analysis complete!")

            # --- Detection Summary ---
            st.markdown("### Detected Sounds")
            if detected_keys:
                for key in detected_keys:
                    score = float(probs[CLASS_KEYS.index(key)])
                    thresh = float(thresholds[key])
                    st.markdown(
                        f"- **{CLASS_DISPLAY_NAMES[key]}** : Confidence `{score:.2%}` "
                        f"(Threshold: `{thresh:.2f}`)"
                    )
            else:
                st.info("No target sound detected above the model thresholds.")

            # --- Context Summary Section ---
            st.markdown("### Context")
            context_text = generate_context_summary(detected_keys)
            st.info(f"**Acoustic Summary:** {context_text}")

            # --- Full Breakdown Table ---
            st.markdown("### Full Category Breakdown")
            df_display = pd.DataFrame(results_data)
            df_display["Confidence"] = df_display["Score"].apply(lambda s: f"{s:.4f} ({s*100:.1f}%)")
            df_display["Threshold"] = df_display["Threshold"].apply(lambda t: f"{t:.2f}")

            st.dataframe(
                df_display[["Class", "Confidence", "Threshold", "Status"]],
                use_container_width=True,
                hide_index=True
            )

            # --- Spectrogram Technical Expander ---
            with st.expander("Technical Details (Spectrogram & Input Tensor)"):
                st.write(f"- **Original File Duration**: `{duration:.2f}` seconds")
                st.write(f"- **Standardized Input Tensor**: `{list(input_tensor.shape)}` (Channels: 1, Mels: 128, Frames: 938)")
                st.write(f"- **Log-Mel Min / Max dB**: `{log_mel.min():.1f} dB` / `{log_mel.max():.1f} dB`")


if __name__ == "__main__":
    main()
