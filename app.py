from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# ------------------------------------------------------------
# Training data from celsius_to_fahrenheit.ipynb
# ------------------------------------------------------------
celsius_q = np.array(
    [-40, -10, 0, 8, 10, 15, 22, 50, 20, 38],
    dtype=float
)

fahrenheit_a = np.array(
    [-40.0, 14.0, 32.0, 46.4, 50.0, 59.0, 71.6, 122.0, 68.0, 100.4],
    dtype=float
)

# ------------------------------------------------------------
# FINAL MODEL from the notebook
# Dense(4) -> Dense(4) -> Dense(1)
# ------------------------------------------------------------
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([l0, l1, l2])

model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.Adam(0.1)
)

# Same final training setting as the notebook
model.fit(
    celsius_q,
    fahrenheit_a,
    epochs=800,
    verbose=False
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or request.form
        celsius = float(data.get("celsius"))

        prediction = model.predict(
            np.array([[celsius]], dtype=float),
            verbose=0
        )

        fahrenheit = float(prediction[0][0])

        return jsonify({
            "celsius": round(celsius, 2),
            "fahrenheit": round(fahrenheit, 2)
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "Please enter a valid Celsius temperature."
        }), 400


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "Dense(4) -> Dense(4) -> Dense(1)",
        "optimizer": "Adam(0.1)",
        "loss": "mean_squared_error",
        "epochs": 800
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
