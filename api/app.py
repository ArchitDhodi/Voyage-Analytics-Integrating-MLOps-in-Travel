import argparse
import os
import joblib
from pathlib import Path
from typing import List, Dict, Any
import sys

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from dotenv import load_dotenv

# Some saved gender models (from notebook tuning) reference a DummyPrep class;
# define a no-op here so unpickling succeeds.
class DummyPrep:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

# Price model pipelines (when trained from notebook/scripts) use this feature adder;
# define it here so unpickling works.
def add_features(df):
    out = df.copy()
    out["date_parsed"] = pd.to_datetime(out["date"], errors="coerce")
    out["month"] = out["date_parsed"].dt.month.fillna(0).astype(int)
    out["dayofweek"] = out["date_parsed"].dt.dayofweek.fillna(0).astype(int)
    out["speed"] = out["distance"] / out["time"].replace(0, np.nan)
    out["speed"] = out["speed"].fillna(out["speed"].median())
    return out.drop(columns=["date_parsed"])


ROOT = Path(__file__).resolve().parent.parent
# Ensure scripts/ is on path so pickled pipelines that reference train_* modules can load
sys.path.append(str(ROOT / "scripts"))
load_dotenv(ROOT / ".env", override=True)

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "trained"))
PRICE_MODEL_FILE = MODEL_PATH / "price_model.pkl"
# Prefer the newer TF-IDF gender model, then legacy embeddings
GENDER_MODEL_FILE = MODEL_PATH / "gender_model.pkl"
GENDER_MODEL_FALLBACK = MODEL_PATH / "gender_model_embeddings.pkl"
RECOMMENDER_FILE = MODEL_PATH / "recommender.pkl"

app = Flask(__name__)


def load_model(path: Path):
    if path.exists():
        try:
            model = joblib.load(path)
            app.logger.info("Loaded model: %s", path.name)
            return model
        except Exception as e:
            app.logger.error("Failed to load %s: %s", path, e)
    return None


price_model = load_model(PRICE_MODEL_FILE)
gender_model = load_model(GENDER_MODEL_FILE)
gender_model_name = GENDER_MODEL_FILE.name if gender_model is not None else None
if gender_model is None:
    gender_model = load_model(GENDER_MODEL_FALLBACK)
    if gender_model is not None:
        gender_model_name = GENDER_MODEL_FALLBACK.name
recommender_artifacts = load_model(RECOMMENDER_FILE)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "price_model": bool(price_model), "gender_model": bool(gender_model)}), 200


def validate_price_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    required = ["source", "destination", "flight_type", "time", "distance", "agency", "date"]
    missing = [k for k in required if payload.get(k) in (None, "")]
    if missing:
        return {"error": f"Missing fields: {missing}"}
    return {}


@app.route("/predict", methods=["POST"])
def predict():
    if price_model is None:
        return jsonify({"error": "Price model not loaded. Train first."}), 500
    data = request.get_json() or {}
    err = validate_price_payload(data)
    if err:
        return jsonify(err), 400
    df = pd.DataFrame([data])
    pred = float(price_model.predict(df)[0])
    return jsonify({"predicted_price": round(pred, 2), "model": PRICE_MODEL_FILE.name})


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if price_model is None:
        return jsonify({"error": "Price model not loaded. Train first."}), 500
    data = request.get_json() or {}
    flights: List[Dict[str, Any]] = data.get("flights", [])
    results = []
    for i, f in enumerate(flights):
        err = validate_price_payload(f)
        if err:
            results.append({"index": i, "error": err["error"]})
            continue
        df = pd.DataFrame([f])
        pred = float(price_model.predict(df)[0])
        results.append({"index": i, "predicted_price": round(pred, 2)})
    return jsonify({"results": results, "total": len(flights)})


@app.route("/gender", methods=["POST"])
def gender_predict():
    if gender_model is None:
        return jsonify({"error": "Gender model not loaded. Train first."}), 500
    data = request.get_json() or {}
    df = None
    # If model exposes expected column names, shape input accordingly
    feature_names = getattr(gender_model, "feature_names_in_", None)
    if feature_names is not None:
        row = {}
        for col in feature_names:
            if col == "text":
                name = data.get("name", "")
                company = data.get("company", "")
                row["text"] = f"{name} {company}".strip()
            elif col == "first_name":
                name = str(data.get("name", "")).strip()
                row["first_name"] = name.split()[0].lower() if name else ""
            else:
                row[col] = data.get(col)
        df = pd.DataFrame([row])
    elif "features" in data and isinstance(data.get("features"), list):
        df = pd.DataFrame([data["features"]])
    elif "age" in data:
        # Legacy age/company model
        row = {"age": data.get("age"), "company": data.get("company", "")}
        df = pd.DataFrame([row])
    else:
        return jsonify({"error": "Provide 'name' for prediction (or legacy 'features' / age+company input)."}), 400

    pred = gender_model.predict(df)[0]
    pred_str = str(pred)
    if pred_str in {"0", "1"} and hasattr(gender_model, "classes_"):
        classes = list(gender_model.classes_)
        pred_str = str(classes[int(pred)])

    resp = {"gender_pred": pred_str, "model": gender_model_name or GENDER_MODEL_FILE.name}
    if hasattr(gender_model, "predict_proba"):
        probs = gender_model.predict_proba(df)[0]
        if hasattr(gender_model, "classes_"):
            classes = list(gender_model.classes_)
            if pred_str in classes:
                proba = float(probs[classes.index(pred_str)])
            else:
                proba = float(max(probs))
        else:
            proba = float(max(probs))
        resp["confidence"] = float(proba)
    return jsonify(resp)


@app.route("/recommend", methods=["GET"])
def recommend():
    if not recommender_artifacts:
        return jsonify({"error": "Recommender not trained. Run train_recommender.py"}), 500
    hotels = recommender_artifacts["hotels"]
    tfidf = recommender_artifacts.get("tfidf")
    knn = recommender_artifacts.get("knn")
    legacy_sim = recommender_artifacts.get("similarity")
    idx = int(request.args.get("index", 0))
    n = int(request.args.get("n", 3))
    if idx < 0 or idx >= len(hotels):
        return jsonify({"error": f"index must be between 0 and {len(hotels)-1}"}), 400
    if tfidf is not None and knn is not None:
        # Query nearest neighbors (cosine distance); skip the item itself
        n_neighbors = min(n + 1, len(hotels))
        distances, indices = knn.kneighbors(tfidf[idx], n_neighbors=n_neighbors)
        flat_inds = [i for i in indices[0] if i != idx][:n]
    elif legacy_sim is not None:
        scores = list(enumerate(legacy_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        flat_inds = [i for i, _ in scores[1 : n + 1]]
    else:
        return jsonify({"error": "Recommender artifact missing similarity data. Re-run train_recommender.py."}), 500
    recs = hotels.iloc[flat_inds][["name", "place", "price", "days"]].to_dict(orient="records")
    return jsonify({"recommendations": recs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", 5000)))
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
