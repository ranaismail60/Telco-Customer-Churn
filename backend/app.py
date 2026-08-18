"""
Phase 7 — Backend.
Flask API that loads model.pkl + scaler.pkl + feature_columns.pkl,
exposes POST /predict, and returns a risk score, risk level, and
(if available) the top reasons behind the prediction.

Run locally:
    cd backend
    pip install -r requirements.txt
    python app.py
    # -> http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

FRONTEND_DIR = ROOT / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)  # allow the frontend to call this API

model = joblib.load(MODEL_DIR / "model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

try:
    explainer = shap.TreeExplainer(model)
    HAS_SHAP = True
except Exception:
    try:
        X_sample = pd.read_csv(ROOT / "data" / "processed" / "X_train.csv").head(100)
        explainer = shap.LinearExplainer(model, X_sample)
        HAS_SHAP = True
    except Exception:
        HAS_SHAP = False


def get_tenure_group(tenure: float) -> str:
    if tenure <= 12:
        return "0-12"
    elif tenure <= 24:
        return "13-24"
    elif tenure <= 48:
        return "25-48"
    else:
        return "49+"


def build_feature_row(payload: dict) -> pd.DataFrame:
    """
    Turn a raw form submission into a single-row DataFrame matching the
    exact columns the model was trained on (same one-hot columns, same
    order), filling anything missing with 0.
    """
    row = {col: 0 for col in feature_columns}

    # Extract basic inputs
    tenure = float(payload.get("tenure", 0))
    monthly_charges = float(payload.get("monthly_charges", 0.0))
    total_charges = float(payload.get("total_charges", 0.0))
    contract_ord = int(payload.get("contract_ordinal", 0))

    # Compute derived features
    tenure_grp = get_tenure_group(tenure)
    
    addon_fields = ["online_security", "online_backup", "device_protection",
                    "tech_support", "streaming_tv", "streaming_movies"]
    total_services = sum(1 for f in addon_fields if str(payload.get(f, "")).lower() in ["yes", "1", "true"])

    avg_monthly_spend_ratio = total_charges / tenure if tenure > 0 else monthly_charges

    tech_support_val = str(payload.get("tech_support", "No"))
    combined_risk_flag = 1 if (tenure < 12 and contract_ord == 0 and tech_support_val.lower() in ["no", "0"]) else 0

    payment_method_val = str(payload.get("payment_method", ""))
    payment_risk_flag = 1 if payment_method_val in ["Electronic check", "Mailed check"] else 0

    # Fill direct passthrough / numeric fields
    computed_values = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_ordinal": contract_ord,
        "senior_citizen": int(payload.get("senior_citizen", 0)),
        "partner": int(payload.get("partner", 0)),
        "dependents": int(payload.get("dependents", 0)),
        "phone_service": int(payload.get("phone_service", 0)),
        "paperless_billing": int(payload.get("paperless_billing", 0)),
        "total_services": total_services,
        "avg_monthly_spend_ratio": avg_monthly_spend_ratio,
        "combined_risk_flag": combined_risk_flag,
        "payment_risk_flag": payment_risk_flag,
    }

    for key, val in computed_values.items():
        if key in row:
            row[key] = val

    # One-hot categorical fields
    onehot_prefixes = {
        "gender": payload.get("gender"),
        "multiple_lines": payload.get("multiple_lines"),
        "internet_service": payload.get("internet_service"),
        "online_security": payload.get("online_security"),
        "online_backup": payload.get("online_backup"),
        "device_protection": payload.get("device_protection"),
        "tech_support": payload.get("tech_support"),
        "streaming_tv": payload.get("streaming_tv"),
        "streaming_movies": payload.get("streaming_movies"),
        "payment_method": payload.get("payment_method"),
        "tenure_group": tenure_grp,
    }
    for prefix, value in onehot_prefixes.items():
        if value is None:
            continue
        col_name = f"{prefix}_{value}"
        if col_name in row:
            row[col_name] = 1

    df_row = pd.DataFrame([row], columns=feature_columns)
    return df_row


def risk_level(prob: float) -> str:
    if prob >= 0.66:
        return "High"
    elif prob >= 0.33:
        return "Medium"
    return "Low"


@app.route("/", methods=["GET"])
def index():
    return app.send_static_file("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({
            "message": "This endpoint requires a POST request with JSON customer data. To test interactively, open http://localhost:5000 in your browser."
        })

    payload = request.get_json(force=True)

    raw_row = build_feature_row(payload)
    scaled_row = pd.DataFrame(
        scaler.transform(raw_row), columns=raw_row.columns
    )

    prob = float(model.predict_proba(scaled_row)[0][1])
    level = risk_level(prob)

    reasons = []
    if HAS_SHAP:
        try:
            shap_values = explainer.shap_values(scaled_row)
            sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            contributions = sorted(
                zip(scaled_row.columns, sv), key=lambda x: abs(x[1]), reverse=True
            )[:3]
            for feat, val in contributions:
                direction = "increases" if val > 0 else "decreases"
                reasons.append(f"{feat.replace('_', ' ')} {direction} risk")
        except Exception as e:
            reasons = [f"(explanation unavailable: {e})"]

    return jsonify({
        "churn_probability": round(prob, 4),
        "risk_level": level,
        "top_reasons": reasons,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
