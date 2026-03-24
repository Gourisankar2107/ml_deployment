import joblib
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("min_max_scaler.joblib")
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# Model features
model_features = [
    "angle_degrees",
    "angular_velocity_deg_s",
    "range_of_motion_degrees",
    "rms_emg_uv",
    "mav_emg_uv"
]

@app.route("/")
def home():
    return "Flask ML API is running!"

@app.route("/predict", methods=["POST"])
def predict():

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if not all(feature in data for feature in model_features):
        missing = [f for f in model_features if f not in data]
        return jsonify({"error": f"Missing features: {missing}"}), 400

    try:
        input_df = pd.DataFrame([{feature: data[feature] for feature in model_features}])
        scaled_input = scaler.transform(input_df)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    label = int(prediction[0])
    confidence = float(prediction_proba[0][label])

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
