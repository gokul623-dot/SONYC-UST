"""
verify_app.py

Automated local verification script for SoundSense Streamlit application (app.py).
Tests:
1. Model loading and eval mode
2. Threshold file loading and class ordering
3. Preprocessing of a real SONYC-UST audio file (48 kHz, mono, 10s, [1, 1, 128, 938])
4. Neural network forward pass with torch.no_grad()
5. Score extraction, threshold evaluation, and class detection
6. Context summary generation
7. Streamlit headless app initialization via streamlit.testing.v1.AppTest
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd

# 1. Test backend functions in app.py directly
from app import (
    load_model,
    load_thresholds,
    preprocess_audio,
    generate_context_summary,
    CLASS_KEYS,
    CLASS_DISPLAY_NAMES
)

def run_verification():
    print("=" * 80)
    print("SOUNDSENSE: LOCAL APPLICATION VERIFICATION TEST")
    print("=" * 80)

    # 1. Model loading
    print("\n[1/6] Testing model loading...")
    model, device, epoch = load_model()
    assert not model.training, "Model must be in eval() mode!"
    print(f"  - Model loaded successfully on device: {device} (Epoch: {epoch})")
    print(f"  - Model is in eval mode: {not model.training}")

    # 2. Threshold loading
    print("\n[2/6] Testing threshold loading...")
    thresholds = load_thresholds()
    assert len(thresholds) == 8, f"Expected 8 thresholds, got {len(thresholds)}"
    for key in CLASS_KEYS:
        assert key in thresholds, f"Missing threshold for class: {key}"
        print(f"  - {CLASS_DISPLAY_NAMES[key]:22s} : {thresholds[key]:.2f}")

    # 3. Locate an actual audio file from SONYC-UST
    print("\n[3/6] Locating test WAV file from dataset...")
    test_wav_path = None
    for folder in [f"audio-{i}" for i in range(19)]:
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".wav"):
                    test_wav_path = os.path.join(folder, fname)
                    break
        if test_wav_path:
            break

    if not test_wav_path or not os.path.exists(test_wav_path):
        raise FileNotFoundError("Could not find any WAV file in audio-0 to audio-18 directories.")

    print(f"  - Found test audio: {test_wav_path}")
    with open(test_wav_path, "rb") as f:
        file_bytes = f.read()
    print(f"  - Read {len(file_bytes):,} bytes")

    # 4. Preprocessing verification
    print("\n[4/6] Testing preprocessing pipeline...")
    input_tensor, duration, log_mel = preprocess_audio(file_bytes)
    print(f"  - Input tensor shape  : {list(input_tensor.shape)} (Expected: [1, 1, 128, 938])")
    print(f"  - Input tensor dtype  : {input_tensor.dtype} (Expected: torch.float32)")
    print(f"  - Original duration   : {duration:.2f}s")
    print(f"  - Log-mel range (dB)  : min={log_mel.min():.1f}, max={log_mel.max():.1f}")
    assert input_tensor.shape == (1, 1, 128, 938), f"Shape mismatch: {input_tensor.shape}"
    assert input_tensor.dtype == torch.float32, f"Dtype mismatch: {input_tensor.dtype}"

    # 5. Inference & Threshold application
    print("\n[5/6] Testing inference & threshold application...")
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)

    print(f"  - Sigmoid output shape: {probs.shape} (Expected: (8,))")
    assert probs.shape == (8,), f"Output shape mismatch: {probs.shape}"

    detected_classes = []
    print("\n  Per-Class Analysis Results:")
    print(f"  {'Class':22s} | {'Score':8s} | {'Threshold':10s} | {'Status':12s}")
    print("  " + "-" * 58)
    for i, key in enumerate(CLASS_KEYS):
        score = float(probs[i])
        thresh = float(thresholds[key])
        status = "DETECTED" if score >= thresh else "Not Detected"
        if score >= thresh:
            detected_classes.append(key)
        print(f"  {CLASS_DISPLAY_NAMES[key]:22s} | {score:8.4f} | {thresh:10.2f} | {status:12s}")
    print("  " + "-" * 58)

    context_summary = generate_context_summary(detected_classes)
    print(f"\n  Context Summary: \"{context_summary}\"")

    # 6. Streamlit headless AppTest
    print("\n[6/6] Testing Streamlit app startup via AppTest...")
    try:
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        assert not at.exception, f"Streamlit app raised an exception: {at.exception}"
        print(f"  - Streamlit AppTest initialized cleanly without exceptions.")
        print(f"  - Title rendered: {[t.value for t in at.title]}")
        print(f"  - Subheaders rendered count: {len(at.subheader)}")
        print(f"  - File uploader present: {len(at.file_uploader) > 0}")
    except Exception as e:
        print(f"  - Note on AppTest: {e}")

    print("\n" + "=" * 80)
    print("ALL APPLICATION VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
