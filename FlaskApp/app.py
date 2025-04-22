from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import torch
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from utils.extract_features import extract_features, extract_mfcc_features, fix_wav_format
from models.evolved_model import EvolvedNN, winner, config
from sklearn.preprocessing import StandardScaler
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Load models
neat_net = torch.load('models/evolved_neat_model.pt', map_location=torch.device('cpu'))
neat_model = EvolvedNN(neat_net, winner)
neat_model.eval()

svc_model = pickle.load(open('models/svc_model.joblib', 'rb'))
kmeans_model = pickle.load(open('models/kmeans_model.joblib', 'rb'))
cluster_label_map = pickle.load(open('models/cluster_label_map.npy', 'rb'))
scaler = pickle.load(open('models/scaler.joblib', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['audio']
        if file.filename == '':
            return render_template("index.html", error="No file selected.")

        subject_id = request.form.get('subject_id', '').strip() or "unknown"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_id = str(uuid.uuid4())[:8]
        result_filename = f"result_{subject_id}_{timestamp}_{result_id}.csv"
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(raw_path)

        fixed_path = fix_wav_format(raw_path)
        features = extract_features(fixed_path)
        mfcc_means = extract_mfcc_features(fixed_path)

        # Convert features for model
        X_input = np.array([
            features["avg_F1"],
            features["jitter_s"],
            features["shimmer"],
            features["mean_HNR"]
        ], dtype=np.float32)

        X_tensor = torch.tensor([X_input])
        with torch.no_grad():
            output = neat_model(X_tensor)
            probs = torch.softmax(output, dim=1).squeeze()
            neat_conf_aut = probs[1].item()
            neat_conf_nonaut = probs[0].item()
            neat_pred = int(neat_conf_aut > 0.5)
            neat_label = "Autistic" if neat_pred else "Non-Autistic"
            neat_conf_pct = round((neat_conf_aut if neat_pred else neat_conf_nonaut) * 100, 2)
            neat_conf_str = f"Autistic: {round(neat_conf_aut*100, 2)}%, Non-Autistic: {round(neat_conf_nonaut*100, 2)}%"

        # SVC prediction
        cluster_id = np.argmin(np.linalg.norm(kmeans_model.cluster_centers_ - X_input, axis=1))
        mapped_label = cluster_label_map[cluster_id]
        X_scaled = scaler.transform([X_input])
        svc_conf = svc_model.predict_proba(X_scaled)[0]
        svc_pred = int(svc_conf[1] > 0.5)
        svc_label = "Autistic" if svc_pred else "Non-Autistic"
        svc_conf_pct = round((svc_conf[1] if svc_pred else svc_conf[0]) * 100, 2)
        svc_conf_str = f"Autistic: {round(svc_conf[1]*100, 2)}%, Non-Autistic: {round(svc_conf[0]*100, 2)}%"

        # Clinical explanation placeholder
        clinical_neat_conf = "High confidence subject is " + neat_label
        clinical_svc_conf = "High confidence subject is " + svc_label

        # Export CSV result
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        with open(result_path, 'w', newline='') as f:
            writer = pd.ExcelWriter(f)
            summary = pd.DataFrame({
                "Subject ID": [subject_id],
                "Timestamp": [timestamp],
                "NEAT Prediction": [neat_label],
                "NEAT Confidence": [neat_conf_str],
                "SVC Prediction": [svc_label],
                "SVC Confidence": [svc_conf_str]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)

        display_features = zip(
            ["avg_F1", "jitter_s", "shimmer", "mean_HNR"],
            [
                float(features["avg_F1"]),
                float(features["jitter_s"]),
                float(features["shimmer"]),
                float(features["mean_HNR"])
            ])

        return render_template("results.html",
            features=display_features,
            mfccs=mfcc_means,
            neat_pred=neat_pred,
            neat_label=neat_label,
            neat_conf_pct=neat_conf_pct,
            neat_conf_str=neat_conf_str,
            svc_pred=svc_pred,
            svc_label=svc_label,
            svc_conf_pct=svc_conf_pct,
            svc_conf_str=svc_conf_str,
            clinical_neat_conf=clinical_neat_conf,
            clinical_svc_conf=clinical_svc_conf,
            subject_id=subject_id,
            timestamp=timestamp,
            result_file=result_filename)

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
