from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import torch
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from utils.extract_osf_features import extract_osf_features as extract_features
from utils.extract_features import fix_wav_format, extract_mfcc_features
from models.evolved_model import EvolvedNN, config
from sklearn.preprocessing import StandardScaler
import uuid
import neat

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Load the winner genome used during training
with open("models/winner_genome.pkl", "rb") as f:
    winner = pickle.load(f)

print(f"✅ Loaded winner has {len(winner.nodes)} nodes and {len(winner.connections)} connections")

neat_net = neat.nn.FeedForwardNetwork.create(winner, config)
neat_model = EvolvedNN(neat_net, winner)

print("✅ App loading model structure:")
for i, layer in enumerate(neat_model.layers):
    print(f"Layer {i}: {layer}")

state_dict = torch.load("models/evolved_neat_model.pt", map_location=torch.device("cpu"))
neat_model.load_state_dict(state_dict)
neat_model.eval()

svc_model = joblib.load('models/calibrated_svc_model.joblib')

kmeans_model = pickle.load(open('models/kmeans_model.joblib', 'rb'))
cluster_label_map = pickle.load(open('models/cluster_label_map.npy', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['audio']
        if file.filename == '':
            return render_template("index.html", error="No file selected.")

        subject_id = request.form.get('subject_id', '').strip() or "unknown"
        gender = request.form.get('gender', '').strip().lower() or None
        age = request.form.get('age', '').strip() or None
        age = int(age) if age and age.isdigit() else None
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_id = str(uuid.uuid4())[:8]
        result_filename = f"result_{subject_id}_{timestamp}_{result_id}.csv"
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(raw_path)

        fixed_path = fix_wav_format(raw_path)
        features = extract_features(fixed_path, gender=gender, age=age)
        mfcc_means = extract_mfcc_features(fixed_path)

        X_input = np.array([
            features["avg_F1"],
            features["jitter_s"],
            features["shimmer"],
            features["mean_HNR"]
        ], dtype=np.float32)

        X_tensor = torch.tensor([X_input])
        with torch.no_grad():
          output = neat_model(X_tensor).squeeze()

          # Normalize logits before computing confidence delta
          normalized_logits = (output - output.mean()) / output.std()
          delta = normalized_logits[1] - normalized_logits[0]
          calibrated_conf = torch.sigmoid(delta).item()

          # Binary prediction
          neat_pred = int(calibrated_conf > 0.5)
          neat_label = "Autistic" if neat_pred else "Non-Autistic"

          # Confidence percentage (based on calibrated delta)
          neat_conf_pct = round(calibrated_conf * 100, 2) if neat_pred else round((1 - calibrated_conf) * 100, 2)

          # Confidence string
          neat_conf_str = f"Autistic: {round(calibrated_conf * 100, 2)}%, Non-Autistic: {round((1 - calibrated_conf) * 100, 2)}%"


          print("✅ Raw NEAT logits:", output.tolist())
          print("✅ Logit Δ (output[1] - output[0]):", (output[1] - output[0]).item())
          print("✅ Sigmoid(Δ):", torch.sigmoid(output[1] - output[0]).item())


        cluster_id = np.argmin(np.linalg.norm(kmeans_model.cluster_centers_ - X_input, axis=1))
        mapped_label = cluster_label_map[cluster_id]
        X_scaled = scaler.transform([X_input])

        svc_prob = svc_model.predict_proba(X_scaled)[0]
        svc_pred = int(svc_prob[1] > 0.5)
        svc_label = "Autistic" if svc_pred else "Non-Autistic"

        # Correct confidence values based on probabilities
        svc_conf_pct = round((svc_prob[1] if svc_pred else svc_prob[0]) * 100, 2)
        svc_conf_str = f"Autistic: {round(svc_prob[1]*100, 2)}%, Non-Autistic: {round(svc_prob[0]*100, 2)}%"
        clinical_neat_conf = "High confidence subject is " + neat_label
        clinical_svc_conf = "High confidence subject is " + svc_label

        print(f"✅ SVC prob: {svc_prob}, SVC label: {svc_label}, SVC conf: {svc_conf_pct}%")


        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        with open(result_path, 'w', newline='') as f:
            summary = pd.DataFrame({
                "Subject ID": [subject_id],
                "Age": [age],
                "Gender": [gender],
                "Timestamp": [timestamp],
                "NEAT Prediction": [neat_label],
                "NEAT Confidence": [neat_conf_str],
                "SVC Prediction": [svc_label],
                "SVC Confidence": [svc_conf_str],
                "avg_F1": [features["avg_F1"]],
                "jitter_s": [features["jitter_s"]],
                "shimmer": [features["shimmer"]],
                "mean_HNR": [features["mean_HNR"]]
            })
            summary.to_csv(f, index=False)

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
            age=age,
            gender=gender,
            timestamp=timestamp,
            result_file=result_filename)

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)