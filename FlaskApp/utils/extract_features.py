"""
extract_features.py

Extracts fundamental acoustic and spectral features from WAV audio files for use in diagnostic and
machine learning applications such as autism voice biomarker analysis.

Functions:
- extract_features(): Extracts avg_F1 (formant F1), jitter, shimmer, and mean HNR using Praat via Parselmouth.
- extract_mfcc_features(): Computes MFCCs using librosa and returns the average values across time frames.
- fix_wav_format(): Converts uploaded audio (e.g., MP3, stereo, variable bit rate) to 16-bit PCM mono WAV at 16kHz 
  using pydub for consistent preprocessing.

Typical use case: Called during Flask app inference pipeline to standardize and extract features from new voice recordings.

Dependencies:
- numpy
- librosa
- parselmouth
- pydub
- uuid
- os
- scipy.stats.variation (optional, if variation is used elsewhere)

"""

import librosa
import parselmouth
import numpy as np
from scipy.stats import variation
from pydub import AudioSegment
import uuid
import os


def extract_features_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # F1 (fundamental frequency)
    F1, _, _ = librosa.pyin(y, fmin=50, fmax=400)
    avg_F1 = np.nanmean(F1)

    # Jitter approximation: relative f0 variation
    F1_diff = np.diff(F1)
    jitter_s = np.nanmean(np.abs(F1_diff / F1[1:])) if F1 is not None else 0

    # Shimmer approximation: energy envelope variability
    energy = librosa.feature.rms(y=y)[0]
    shimmer = np.mean(np.abs(np.diff(energy))) if len(energy) > 1 else 0

    # HNR approximation using harmonic-to-noise ratio from autocorrelation
    autocorr = librosa.autocorrelate(y)
    mean_hnr = 10 * np.log10(np.max(autocorr) / (np.mean(autocorr) + 1e-6))

    # MFCCs (mean over time frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)

    # Return both acoustic and MFCCs
    features = [avg_F1, jitter_s, shimmer, mean_hnr]
    return features, mfcc_means

def extract_features(wav_path):
    snd = parselmouth.Sound(wav_path)

    # --- avg_F1 using formants ---
    try:
        formant = snd.to_formant_burg()
        time_points = np.linspace(0, snd.duration, 100)
        f1_values = [formant.get_value_at_time(1, t) for t in time_points]
        avg_f1 = np.nanmean(f1_values)
        print("✅ avg_F1:", avg_f1)
    except Exception as e:
        print("❌ avg_F1 extraction failed:", e)
        avg_f1 = np.nan

    # --- jitter_s and shimmer ---
    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

        jitter_s = parselmouth.praat.call(
            [point_process, snd],
            "Get jitter (local, absolute)", snd, 0.0, 0.0, 0.0001, 0.02, 1.3
        )

        shimmer = parselmouth.praat.call(
            [point_process,snd],
            "Get shimmer (local)", snd, 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
        )

        print("✅ jitter_s:", jitter_s)
        print("✅ shimmer:", shimmer)

    except Exception as e:
        print("❌ Jitter/Shimmer extraction failed:", e)
        jitter_s = np.nan
        shimmer = np.nan

    # --- mean_HNR ---
    try:
        harmonicity = snd.to_harmonicity_cc()
        #mean_hnr = harmonicity.get_mean()
        mean_hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        print("✅ mean_HNR:", mean_hnr)
    except Exception as e:
        print("❌ mean_HNR extraction failed:", e)
        mean_hnr = np.nan

    # --- Final check and return ---
    features = [avg_f1, jitter_s, shimmer, mean_hnr]
    if np.isnan(features).any():
        print("⚠️ Warning: NaN detected in extracted features:", features)

    return features


def extract_mfcc_features(wav_path, n_mfcc=13):


    fixed_wav_path = fix_wav_format(wav_path)
    y, sr = librosa.load(fixed_wav_path, sr=None)
    #y, sr = librosa.load(wav_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
     
    # Reduce across time axis (e.g., take mean for each coefficient)
    mfcc_mean = np.mean(mfcc, axis=1)

    os.remove(fixed_wav_path)
    return mfcc_mean  # shape: (13,)


def fix_wav_format(input_path, output_dir="uploads"):
    """
    Converts input WAV file to 16-bit PCM mono, 16kHz format using pydub.
    Returns path to fixed output file.
    """
    # Ensure upload directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate unique output filename
    temp_id = uuid.uuid4().hex[:8]
    output_path = os.path.join(output_dir, f"fixed_{temp_id}.wav")

    try:
        # Load and convert audio
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # 16-bit PCM
        audio.export(output_path, format="wav")
        return output_path

    except Exception as e:
        print(f"Error fixing WAV format: {e}")
        return None
