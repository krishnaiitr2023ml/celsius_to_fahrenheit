from flask import Flask, render_template, request, jsonify
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf

app = Flask(__name__)

celsius_q = np.array(
    [-40, -10, 0, 8, 10, 15, 22, 50, 20, 38],
    dtype=np.float32
)

fahrenheit_a = np.array(
    [-40.0, 14.0, 32.0, 46.4, 50.0, 59.0, 71.6, 122.0, 68.0, 100.4],
    dtype=np.float32
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.Adam(0.1)
)

model.fit(
    celsius_q,
    fahrenheit_a,
    epochs=800,
    verbose=0
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify(
        status="ok",
        model="Dense(4) -> Dense(4) -> Dense(1)",
        epochs=800
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not data or "celsius" not in data:
            return jsonify(error="Please provide a Celsius value."), 400

        celsius = float(data["celsius"])

        x = tf.constant([[celsius]], dtype=tf.float32)
        prediction = model(x, training=False)
        fahrenheit = float(prediction.numpy()[0][0])

        return jsonify(
            celsius=round(celsius, 2),
            fahrenheit=round(fahrenheit, 2)
        ), 200

    except (TypeError, ValueError):
        return jsonify(error="Please enter a valid numeric Celsius value."), 400

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify(error=f"Prediction failed on server: {type(exc).__name__}"), 500

@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Endpoint not found."), 404

@app.errorhandler(500)
def internal_error(_):
    return jsonify(error="Internal server error."), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
