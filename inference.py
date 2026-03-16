"""
ChurnIQ — SageMaker Inference Handler
=======================================
SageMaker calls these 4 functions automatically when serving predictions:
  model_fn()      → called ONCE on startup  — loads the model from disk
  input_fn()      → called per request      — parses incoming JSON
  predict_fn()    → called per request      — runs prediction
  output_fn()     → called per request      — formats the response

This file must be named inference.py and live inside the model.tar.gz
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "xgboost==2.1.4", "imbalanced-learn>=0.12.0", "-q", "--no-cache-dir"])

import os
import json
import pickle
import numpy as np
import pandas as pd

# ── Feature Engineering (must match train.py exactly) ─────────
def engineer_features(data):
    df = data.copy()
    df["tenure"]         = pd.to_numeric(df["tenure"],         errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)
    df["TotalCharges"]   = pd.to_numeric(df["TotalCharges"],   errors="coerce").fillna(0)
    df["SeniorCitizen"]  = pd.to_numeric(df["SeniorCitizen"],  errors="coerce").fillna(0)

    df["ChargePerTenure"]    = df["MonthlyCharges"] / (df["tenure"] + 1)
    service_cols = ["PhoneService","MultipleLines","InternetService","OnlineSecurity",
                    "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    df["TotalServices"]      = df[service_cols].apply(
        lambda row: sum(1 for v in row if str(v).strip() not in
                        ["No","No phone service","No internet service"]), axis=1)
    df["HasProtection"]      = ((df["OnlineSecurity"]=="Yes") | (df["DeviceProtection"]=="Yes") | (df["TechSupport"]=="Yes")).astype(int)
    df["IsLongTermCustomer"] = (df["tenure"] > 24).astype(int)
    df["HighMonthlyCharge"]  = (df["MonthlyCharges"] > 64.76).astype(int)
    df["AutoPayment"]        = df["PaymentMethod"].isin(["Bank transfer (automatic)","Credit card (automatic)"]).astype(int)
    df["FiberNoSecurity"]    = ((df["InternetService"]=="Fiber optic") & (df["OnlineSecurity"]=="No")).astype(int)
    df["HighRiskCombo"]      = ((df["Contract"]=="Month-to-month") & (df["PaymentMethod"]=="Electronic check")).astype(int)
    return df

FEATURE_COLS = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges",
    "ChargePerTenure","TotalServices","HasProtection","IsLongTermCustomer",
    "HighMonthlyCharge","AutoPayment","FiberNoSecurity","HighRiskCombo"
]


# ── 1. model_fn — Load model from disk (called once at startup) ──
def model_fn(model_dir):
    """
    SageMaker calls this when the endpoint starts.
    model_dir = the folder where your model.tar.gz was extracted.
    Return a dict with everything needed for prediction.
    """
    print(f"Loading model from: {model_dir}")

    with open(os.path.join(model_dir, "churn_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(model_dir, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)

    with open(os.path.join(model_dir, "threshold.json"), "r") as f:
        threshold = json.load(f).get("threshold", 0.5)

    print(f"✅ Model loaded | Threshold: {threshold:.2f}")
    return {
        "model":        model,
        "preprocessor": preprocessor,
        "threshold":    threshold
    }


# ── 2. input_fn — Parse incoming request ──────────────────────
def input_fn(request_body, content_type):
    """
    Parses the raw HTTP request body.
    Expects JSON like: {"tenure": 12, "MonthlyCharges": 70.5, ...}
    """
    if content_type == "application/json":
        data = json.loads(request_body)
        # Handle both single dict and list of dicts (batch)
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported content type: {content_type}")


# ── 3. predict_fn — Run prediction ────────────────────────────
def predict_fn(input_df, model_assets):
    """
    Receives the parsed DataFrame and the loaded model assets.
    Returns a list of prediction dicts.
    """
    model        = model_assets["model"]
    preprocessor = model_assets["preprocessor"]
    threshold    = model_assets["threshold"]

    # Engineer features
    df_eng = engineer_features(input_df)
    X      = preprocessor.transform(df_eng[FEATURE_COLS])

    # Predict
    probs  = model.predict_proba(X)[:, 1]
    preds  = (probs >= threshold).astype(int)

    results = []
    for prob, pred in zip(probs, preds):
        results.append({
            "churn":       int(pred),
            "probability": round(float(prob) * 100, 2),
            "risk_level":  "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low",
            "threshold":   round(float(threshold), 2)
        })
    return results


# ── 4. output_fn — Format response ────────────────────────────
def output_fn(predictions, accept="application/json"):
    """
    Converts predictions to JSON response.
    Single prediction → returns dict
    Multiple predictions → returns list
    """
    if accept == "application/json":
        result = predictions[0] if len(predictions) == 1 else predictions
        return json.dumps(result), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")
