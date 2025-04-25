"""
audio_quality.py

This module provides utility functions for evaluating the acoustic quality of audio files,
particularly in the context of biomedical or speech-based diagnostic applications.

Functions include:
- compute_snr(): Estimates the signal-to-noise ratio (SNR) of an audio signal.
- compute_peak_db(): Computes the peak amplitude in decibels relative to full scale (dBFS).

These metrics are used to flag suboptimal audio inputs that may negatively impact
the reliability of feature extraction (e.g., jitter, shimmer) and downstream prediction models.

Recommended usage:
Import and apply these functions in preprocessing pipelines or real-time applications
(e.g., Flask apps) to log or gate predictions based on audio quality.

Dependencies: numpy, scipy.io
"""

import numpy as np
from scipy.io import wavfile

def compute_snr(wav_path, frame_duration_sec=1.0):
    try:
        sr, y = wavfile.read(wav_path)

        # Normalize if 16-bit int
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0

        if y.ndim > 1:
            y = y.mean(axis=1)  # Convert to mono if stereo

        total_samples = len(y)
        frame_length = int(frame_duration_sec * sr)

        if total_samples < frame_length:
            raise ValueError("Audio too short for SNR estimation.")

        start = max((total_samples - frame_length) // 2, 0)
        end = start + frame_length
        signal = y[start:end]

        if np.all(signal == 0):
            raise ValueError("Audio is silent.")

        signal_power = np.mean(signal ** 2)
        noise = signal - np.mean(signal)
        noise_power = np.mean(noise ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

        return round(snr_db, 2)

    except Exception as e:
        print(f"❌ SNR computation failed: {e}")
        return np.nan

def compute_peak_db(file_path):
    try:
        sr, y = wavfile.read(file_path)

        # Normalize if 16-bit PCM
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0

        if y.ndim > 1:
            y = y.mean(axis=1)  # Convert stereo to mono

        peak = np.max(np.abs(y))
        if peak == 0:
            return -np.inf  # Silence

        return round(20 * np.log10(peak), 2)

    except Exception as e:
        print("❌ Peak dB computation failed:", e)
        return np.nan
