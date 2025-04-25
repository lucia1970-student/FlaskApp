"""
logger.py

Handles structured logging of predictions and diagnostics for audio-based classification tasks.

Functions include:
- log_prediction_csv(): Logs basic prediction outputs (e.g., NEAT/SVC predictions, confidence scores, acoustic features).
- log_prediction_diagnostics_csv(): Logs detailed diagnostics (e.g., SNR, peak dB, voice quality flags), 
  and writes flagged entries to a separate file for further review.

Typical use case: Called at the end of a prediction pipeline to record subject-level outcomes and acoustic conditions.

Outputs:
- results/results_log.csv: standard output log
- results/results_diagnostics.csv: detailed diagnostics log
- results/results_flagged.csv: only poor-quality samples (optional but auto-created)

Dependencies: os, csv, datetime
"""


import csv
import os
from datetime import datetime

def log_prediction_csv(subject_id, features, neat_pred, neat_conf, svc_pred, svc_conf,
                       clinical_neat_conf, clinical_svc_conf,
                       filepath='results/results_log.csv'):
    # Ensure results folder exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Get timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Format row
    row = {
        'timestamp': timestamp,
        'subject_id': subject_id,
        'avg_F1': features['avg_F1'],
        'jitter_s': features['jitter_s'],
        'shimmer': features['shimmer'],
        'mean_HNR': features['mean_HNR'],
        'neat_pred': neat_pred,
        'neat_conf': round(neat_conf, 4),
        'clinical_neat_conf': clinical_neat_conf,
        'svc_pred': svc_pred,
        'svc_conf': round(svc_conf, 4),
        'clinical_svc_conf': clinical_svc_conf
    }

    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def log_prediction_diagnostics_csv(
    subject_id,
    age,
    gender,
    features,
    neat_label,
    neat_conf_pct,
    svc_label,
    svc_conf_pct,
    snr_db,
    peak_db,
    snr_msg,
    peak_db_msg,
    voice_quality_flag=None,  # optional, auto-inferred below
    filepath='results/results_diagnostics.csv'
):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Infer quality flag if not provided explicitly
    if voice_quality_flag is None:
        if snr_db is not None and snr_db < 10:
            voice_quality_flag = "Low SNR"
        elif peak_db is not None and peak_db < 60:
            voice_quality_flag = "Low Peak SPL"
        else:
            voice_quality_flag = "OK"

    row = {
        'Timestamp': timestamp,
        'Subject_ID': subject_id,
        'Age': age,
        'Gender': gender,
        'avg_F1': features["avg_F1"],
        'jitter_s': features["jitter_s"],
        'shimmer': features["shimmer"],
        'mean_HNR': features["mean_HNR"],
        'NEAT_Prediction': neat_label,
        'NEAT_Confidence_%': neat_conf_pct,
        'SVC_Prediction': svc_label,
        'SVC_Confidence_%': svc_conf_pct,
        'SNR_dB': snr_db,
        'Peak_dB': peak_db,
        'Voice_Quality_Flag': voice_quality_flag
    }

    # Log to diagnostics file
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # ⚠️ Log to flagged file if needed
    if voice_quality_flag != "OK":
        flagged_path = "results/results_flagged.csv"
        flagged_exists = os.path.isfile(flagged_path)
        with open(flagged_path, 'a', newline='') as f_flag:
            writer = csv.DictWriter(f_flag, fieldnames=row.keys())
            if not flagged_exists:
                writer.writeheader()
            writer.writerow(row)

    return voice_quality_flag
