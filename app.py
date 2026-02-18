from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import random

# =========================
# Flask App Init
# =========================
app = Flask(__name__)

# =========================
# Load ML Model
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "wind_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ wind_model.pkl not found")

model = joblib.load(MODEL_PATH)

print("✅ Model loaded successfully")
print("🔢 Model expects", model.n_features_in_, "features")

# =========================
# Home Page
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# Prediction Page
# =========================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_text = None
    error = None

    if request.method == "POST":
        try:
            # Read all 6 inputs
            wind_speed = float(request.form["wind_speed"])
            wind_direction = float(request.form["wind_direction"])
            air_density = float(request.form["air_density"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            blade_length = float(request.form["blade_length"])

            # Create feature array (MUST match training order)
            features = np.array([[
                wind_speed,
                wind_direction,
                air_density,
                temperature,
                humidity,
                blade_length
            ]])

            prediction = model.predict(features)[0]
            prediction_text = f"Predicted Wind Energy Output: {prediction:.2f} kW"

        except Exception as e:
            error = f"Prediction error: {str(e)}"

    return render_template(
        "prediction.html",
        prediction_text=prediction_text,
        error=error
    )

# =========================
# Simulated Live Wind Data (NO API KEY)
# =========================
@app.route("/live-weather")
def live_weather():
    """
    Simulated live wind data for demo purposes
    """

    return jsonify({
        "wind_speed": round(random.uniform(3.0, 15.0), 2),   # m/s
        "wind_direction": random.randint(0, 360),           # degrees
        "temperature": round(random.uniform(10, 40), 1),    # °C
        "humidity": random.randint(30, 80)                  # %
    })

# =========================
# Run App
# =========================

   
@app.route("/map")
def wind_map():
    return render_template("map.html")

if __name__ == "__main__":
   app.run(debug=True)