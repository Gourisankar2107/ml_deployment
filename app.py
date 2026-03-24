import joblib
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)

# Load the trained model and scaler
try:
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("min_max_scaler.joblib")
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler file not found.")
    print("Make sure 'random_forest_model.joblib' and 'min_max_scaler.joblib' are in the current directory.")
    exit()
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# Features expected by the model
model_features = [
    "angle_degrees",
    "angular_velocity_deg_s",
    "range_of_motion_degrees",
    "rms_emg_uv",
    "mav_emg_uv"
]

@app.route("/")
def home():
    return "Your Flask API is running! Send POST requests to /predict to get predictions."

@app.route("/predict", methods=["POST"])
def predict():

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Validate input
    if not all(feature in data for feature in model_features):
        missing_features = [f for f in model_features if f not in data]
        return jsonify({
            "error": f"Missing features: {', '.join(missing_features)}"
        }), 400

    try:
        # Convert input to dataframe
        input_df = pd.DataFrame([{feature: data[feature] for feature in model_features}])
    except Exception as e:
        return jsonify({"error": f"Error processing input data: {e}"}), 400

    try:
        # Scale input
        scaled_input = scaler.transform(input_df)
    except Exception as e:
        return jsonify({"error": f"Scaling error: {e}"}), 500

    # Prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    prediction_label = int(prediction[0])
    confidence = float(prediction_proba[0][prediction_label])

    return jsonify({
        "prediction": prediction_label,
        "confidence": confidence
    })


# Correct main function
if __name__ == "__main__":
    app.run(debug=True)
