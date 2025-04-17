from flask import Flask, render_template, request, send_file, abort
import os
import neat
import torch
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from models.evolved_model import EvolvedNN, winner, config
from utils.extract_features import extract_features_from_audio
from utils.extract_osf_features import extract_osf_features
from utils.extract_features import fix_wav_format, extract_features, extract_mfcc_features
import uuid
from utils.logger import log_prediction_csv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Load NEAT genome and build model
with open('models/evolved_neat_model.pkl', 'rb') as f:
    winner = pickle.load(f)
neat_net = neat.nn.FeedForwardNetwork.create(winner, config)
neat_model = EvolvedNN(neat_net, winner)

# Load SVC model
with open('models/svc_model.joblib', 'rb') as f:
    svc_model = pickle.load(f)

# Load KMeans model
with open('models/kmeans_model.joblib', 'rb') as f:
    kmeans_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Load cluster label map
cluster_label_map = np.load('models/cluster_label_map.npy', allow_pickle=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return render_template("index.html", error="No audio file uploaded.")

        file = request.files['audio']
        if file.filename == '':
            return render_template("index.html", error="No file selected.")

        subject_id = request.form.get('subject_id', '').strip() or "unknown"
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        filename = f"subject_{subject_id}_{timestamp}.wav"

        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(raw_path)

        fixed_path = fix_wav_format(raw_path)
        if not fixed_path:
            return render_template("index.html", error="Audio format could not be processed.")

        features = extract_osf_features(fixed_path, gender="female")
        print("Extracted features:", features)
        #print("Contains NaN?", np.isnan(features).any())

        if np.isnan(np.array(list(features.values()))).any():
            print("⚠️ Warning: NaN detected in extracted features:", features)
            return render_template("index.html", error="Could not extract all features — please try a clearer voice sample.")
        mfcc_means = extract_mfcc_features(fixed_path)

        # Extract feature values and cast safely to float32
        X_np = np.array(list(features.values()), dtype=np.float32)

        # Optional: check for NaNs or infs
        if not np.isfinite(X_np).all():
            print("⚠️ Invalid value in features:", X)

        selected_keys = ["avg_F1", "jitter_s", "shimmer", "mean_HNR"]
        X_input = np.array([features[k] for k in selected_keys], dtype=np.float32)
        X_tensor = torch.tensor([X_input], dtype=torch.float32)

        # Wrap as tensor
        #X_tensor = torch.tensor([X_np], dtype=torch.float32)
        with torch.no_grad():
            output = neat_model(X_tensor)
            print("✅ NEAT raw output:", output)

            confidences = torch.softmax(output, dim=1).squeeze()
            neat_conf = confidences[1].item()  # confidence for 'Autistic'
            neat_pred = int(neat_conf > 0.5)
            neat_label = 'Autistic' if neat_pred else 'Non-Autistic'
            neat_conf_pct = round((neat_conf if neat_pred else 1 - neat_conf) * 100, 2)

        cluster_id = np.argmin(np.linalg.norm(kmeans_model.cluster_centers_ - X_input, axis=1))
        mapped_label = cluster_label_map[cluster_id]

        print("✅ Extracted features (raw):", features)
        print("✅ X_input:", X_input)
        X_scaled = scaler.transform([X_input])
        print("✅ Scaled features :", features)
        print("✅ X_scaled:", X_scaled)

        svc_pred = svc_model.predict(X_scaled)[0]
        idx_autistic = np.where(svc_model.classes_ == 1)[0][0]
        svc_conf = svc_model.predict_proba(X_scaled)[0][idx_autistic]
        svc_label = 'Autistic' if svc_pred else 'Non-Autistic'
        svc_conf_pct = round((svc_conf if svc_pred else 1 - svc_conf) * 100, 2)

        print("✅ SVC predict_proba:", svc_model.predict_proba(X_scaled))
        print("✅ NEAT confidences:", confidences)

        clinical_neat_conf = interpret_clinical_confidence(neat_conf)
        clinical_svc_conf = interpret_clinical_confidence(svc_conf)
        
        log_prediction_csv(subject_id, features, neat_pred, neat_conf, svc_pred, svc_conf, clinical_neat_conf, clinical_svc_conf)
        display_features = zip(
          ["avg_F1", "jitter_s", "shimmer", "mean_HNR"],
            [
              float(features["avg_F1"]),
              float(features["jitter_s"]),
              float(features["shimmer"]),
              float(features["mean_HNR"])
            ])

        # Unique result filename using subject_id and timestamp
        result_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_filename = f"result_{subject_id}_{timestamp}_{result_id}.csv"

        # Full file path (used for saving the CSV)
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        # Save individual CSV report
        df = pd.DataFrame({
          "Feature": ["avg_F1", "jitter_s", "shimmer", "mean_HNR"],
          "Value": [
            float(features["avg_F1"]),
            float(features["jitter_s"]),
            float(features["shimmer"]),
            float(features["mean_HNR"])
          ],
          "NEAT_Prediction": neat_label,
          "NEAT_Confidence (%)": neat_conf_pct,
          "Clinical_NEAT_Confidence": clinical_neat_conf,
          "SVC_Prediction": svc_label,
          "SVC_Confidence (%)": svc_conf_pct,
        "Clinical_SVC_Confidence": clinical_svc_conf
        })

        df.to_csv(result_path, index=False)  # ✅ Write results to CSV

        result_file=result_filename
        
        return render_template("results.html",
            features=display_features,
            mfccs=mfcc_means,
            neat_pred=neat_pred,
            neat_conf=neat_conf,
            neat_label=neat_label,
            neat_conf_pct=neat_conf_pct,
            svc_pred=svc_pred,
            svc_conf=svc_conf,
            svc_label=svc_label,
            svc_conf_pct=svc_conf_pct,
            clinical_neat_conf=clinical_neat_conf,
            clinical_svc_conf=clinical_svc_conf,
            subject_id=subject_id,
            timestamp=timestamp,
            result_file=result_filename)

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)

def interpret_clinical_confidence(prob_autistic: float) -> str:
    if prob_autistic < 0.1:
        return "High confidence subject is Non-Autistic"
    elif prob_autistic < 0.3:
        return "Moderate confidence subject is Non-Autistic"
    elif prob_autistic <= 0.7:
        return "Uncertain — further assessment recommended"
    elif prob_autistic <= 0.9:
        return "Moderate confidence subject is Autistic"
    else:
        return "High confidence subject is Autistic"

if __name__ == "__main__":
    app.run(debug=True)
