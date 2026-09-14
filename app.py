import ssl
import os
import json
import hashlib
import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="public", static_folder="public", static_url_path="")
CORS(app)

# ── Firebase ───────────────────────────────────────────────────────────────────

def get_firestore_client():
    if not firebase_admin._apps:
        creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        if creds_json:
            cred = credentials.Certificate(json.loads(creds_json))
        else:
            key_path = "serviceAccountKey.json"
            if os.path.exists(key_path):
                cred = credentials.Certificate(key_path)
            else:
                raise ValueError(
                    "Firebase credentials not found. "
                    "Set FIREBASE_SERVICE_ACCOUNT env var or provide serviceAccountKey.json"
                )
        firebase_admin.initialize_app(cred)
    return firestore.client()


# ── ML model singleton ─────────────────────────────────────────────────────────

_model = None
_scaler = None

def _verify_hash(path, env_var):
    expected = os.getenv(env_var)
    if not expected:
        return
    with open(path, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != expected:
        raise ValueError(f"Integrity check failed for {os.path.basename(path)}")

def get_model():
    global _model, _scaler
    if _model is None:
        model_path  = "models/decision_tree.pkl"
        scaler_path = "models/scaler.pkl"
        _verify_hash(model_path,  "MODEL_HASH_TREE")
        _verify_hash(scaler_path, "MODEL_HASH_SCALER")
        _model  = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
    return _model, _scaler


# ── Input validation ───────────────────────────────────────────────────────────

def validate_sensor_data(data):
    required = ["urine_output", "urine_flow_rate", "catheter_bag_volume", "remaining_volume"]
    for field in required:
        if data.get(field) is None:
            return None, f"Missing required field: {field}"
    try:
        cleaned = {
            "urine_output":        float(data["urine_output"]),
            "urine_flow_rate":     float(data["urine_flow_rate"]),
            "catheter_bag_volume": float(data["catheter_bag_volume"]),
            "remaining_volume":    float(data["remaining_volume"]),
        }
    except (TypeError, ValueError) as exc:
        return None, f"Invalid data type: {exc}"

    bounds = [
        ("catheter_bag_volume", 0, 800),
        ("remaining_volume",    0, 800),
        ("urine_flow_rate",     0, 100),
        ("urine_output",        0, 5000),
    ]
    for field, lo, hi in bounds:
        if not (lo <= cleaned[field] <= hi):
            return None, f"{field} out of acceptable range ({lo}–{hi})"

    return cleaned, None


# ── API key auth ───────────────────────────────────────────────────────────────

def _check_auth():
    key = os.getenv("API_SECRET_KEY")
    if not key:
        return True  # auth disabled in dev when key not set
    return request.headers.get("X-Api-Key") == key


# ── Firestore helpers ──────────────────────────────────────────────────────────

def save_data_to_firestore(data):
    try:
        db  = get_firestore_client()
        col = db.collection("intellicath_data")

        last_docs = (
            col.order_by("timestamp", direction=firestore.Query.DESCENDING)
               .limit(1)
               .stream()
        )
        last = None
        for doc in last_docs:
            last = doc.to_dict()
            break

        if last:
            no_change = (
                abs(last.get("urine_output",        0) - data["urine_output"])        <= 2   and
                abs(last.get("urine_flow_rate",     0) - data["urine_flow_rate"])     <= 0.1 and
                abs(last.get("catheter_bag_volume", 0) - data["catheter_bag_volume"]) <= 2
            )
            if no_change:
                logger.debug("No significant change — skipping insert.")
                return True

        data["timestamp"] = firestore.SERVER_TIMESTAMP
        col.add(data)
        logger.info("Record saved to Firestore.")
        return True

    except Exception as exc:
        logger.error("Firestore save failed: %s", exc)
        return False


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("landing.html")


@app.route("/predict-post", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401

    raw = request.get_json(silent=True)
    if not raw:
        return jsonify({"error": "No data received"}), 400

    cleaned, err = validate_sensor_data(raw)
    if err:
        return jsonify({"error": err}), 400

    try:
        model, scaler = get_model()
        features = [[cleaned["remaining_volume"], cleaned["urine_flow_rate"]]]
        minutes  = model.predict(scaler.transform(features))[0]
        hours    = int(minutes // 60)
        mins     = int(minutes % 60)
        predicted_time = f"{hours:02} hours and {mins:02} minutes"
        logger.info("Predicted time: %s", predicted_time)

        actual_time = (
            datetime.now().strftime("%H:%M")
            if cleaned["catheter_bag_volume"] >= 800 else None
        )

        device_id = request.headers.get("X-Device-Id")
        record = {**cleaned, "predicted_time": predicted_time, "actual_time": actual_time}
        if device_id:
            record["device_id"] = device_id

        save_data_to_firestore(record)

        return jsonify({"status": "success", "predicted_time": predicted_time, "actual_time": actual_time})

    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        return jsonify({"error": "An internal error occurred. Please try again."}), 500


@app.route("/api/data", methods=["GET"])
def get_data():
    if not _check_auth():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        db        = get_firestore_client()
        col       = db.collection("intellicath_data")
        device_id = request.args.get("device")

        query = col.order_by("timestamp", direction=firestore.Query.DESCENDING)
        if device_id:
            query = query.where("device_id", "==", device_id)

        for doc in query.limit(1).stream():
            data = doc.to_dict()
            if data.get("timestamp"):
                data["timestamp"] = str(data["timestamp"])
            data["id"] = doc.id
            return jsonify(data)

        return jsonify({"status": "no_data", "message": "No data available"})

    except Exception as exc:
        logger.error("Data fetch error: %s", exc, exc_info=True)
        return jsonify({"error": "An internal error occurred. Please try again."}), 500


# ── Server startup ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ssl_cert = os.getenv("SSL_CERT_PATH", "localhost.pem")
    ssl_key  = os.getenv("SSL_KEY_PATH",  "localhost-key.pem")
    debug    = os.getenv("FLASK_DEBUG", "False") == "True"
    host     = os.getenv("FLASK_HOST", "0.0.0.0")
    port     = int(os.getenv("FLASK_PORT", 5001))

    if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile=ssl_cert, keyfile=ssl_key)
        logger.info("Starting with SSL on port %d", port)
        app.run(debug=debug, host=host, port=port, ssl_context=context)
    else:
        logger.warning("SSL certificates not found — starting without SSL.")
        app.run(debug=debug, host=host, port=port)
