########################################################
# Description
########################################################
# script to perform predictions using a saved model
########################################################



########################################################
# Imports & Settings
########################################################
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
########################################################



PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(PROJECT_DIR / "model.bin")))

app = Flask(__name__)


def load_bundle():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run 'uv run python train.py' first."
        )
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
model = bundle["model"]
feature_names = bundle["feature_names"]
original_feature_names = bundle["original_feature_names"]
threshold = bundle["threshold"]

aliases = dict(zip(original_feature_names, feature_names))
aliases.update({name: name for name in feature_names})



@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.post("/predict")
def predict():
    incoming = request.get_json(silent=True)
    if not isinstance(incoming, dict):
        return jsonify({"error": "Send a JSON object of molecular descriptor values"}), 400

    # Keep the request format simple: {"SpMax_L": 3.0, ...}
    canonical = {}
    for name, value in incoming.items():
        mapped_name = aliases.get(name)
        if mapped_name is None:
            return jsonify({"error": f"Unrecognized descriptor: {name}"}), 400
        if mapped_name in canonical:
            return jsonify({"error": f"Descriptor provided twice: {mapped_name}"}), 400
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return jsonify({"error": f"Descriptor {name} must be a finite number"}), 400
        canonical[mapped_name] = float(value)

    missing = [name for name in feature_names if name not in canonical]
    if missing:
        return jsonify({"error": "Missing required descriptors", "missing": missing}), 400

    # Exact training column order matters.
    X = pd.DataFrame([[canonical[name] for name in feature_names]], columns=feature_names)
    probability_rb = float(model.predict_proba(X)[0, 1])
    predicted_label = "RB" if probability_rb >= threshold else "NRB"

    return jsonify({
        "prediction": predicted_label,
        "probability_RB": round(probability_rb, 6),
        "threshold": threshold,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9696, debug=False)