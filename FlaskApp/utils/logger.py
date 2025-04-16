import csv
import os
from datetime import datetime

def log_prediction_csv(subject_id, features, neat_pred, neat_conf, svc_pred, svc_conf, filepath='results/results_log.csv'):
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
        'neat_conf': neat_conf,
        'svc_pred': svc_pred,
        'svc_conf': svc_conf,
    }

    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
