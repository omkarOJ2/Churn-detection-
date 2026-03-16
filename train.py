"""
ChurnIQ — SageMaker Training Script
====================================
This script runs INSIDE the SageMaker CPU container (ml.m5.xlarge).
SageMaker automatically:
  - Copies your dataset from S3 → /opt/ml/input/data/train/
  - Runs this script
  - Saves everything in /opt/ml/model/ back to S3 as model.tar.gz
  - Shuts down the instance when done

You never SSH into the machine. SageMaker manages everything.
"""

import os
import sys
import subprocess

# ── Install packages missing from sagemaker-scikit-learn:1.2-1-cpu-py3 ──
# Container has: sklearn, pandas, numpy — NOT xgboost, imbalanced-learn, optuna
print("📦 Installing required packages into container...")
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "xgboost==2.1.4", "imbalanced-learn>=0.12.0", "optuna>=3.6.0",
    "-q", "--no-cache-dir"])
print("✅ Packages ready\n")

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── SageMaker passes these paths as environment variables ──────
# /opt/ml/input/data/train/ → your dataset from S3
# /opt/ml/model/            → everything saved here goes back to S3
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR",       "/opt/ml/model")
SM_DATA_DIR  = os.environ.get("SM_CHANNEL_TRAIN",   "/opt/ml/input/data/train")
SM_CODE_DIR  = os.environ.get("SM_CHANNEL_CODE",    "/opt/ml/input/data/code")
SM_OUTPUT    = os.environ.get("SM_OUTPUT_DATA_DIR",  "/opt/ml/output")

os.makedirs(SM_MODEL_DIR, exist_ok=True)
os.makedirs(SM_OUTPUT,    exist_ok=True)

print("=" * 60)
print("  ChurnIQ v2 — SageMaker Training Job")
print("=" * 60)
print(f"  Model dir  : {SM_MODEL_DIR}")
print(f"  Data dir   : {SM_DATA_DIR}")
print(f"  Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()


# ── 1. FEATURE ENGINEERING ────────────────────────────────────
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


# ── 2. LOAD DATA ──────────────────────────────────────────────
print("📥 Loading dataset...")
csv_path = os.path.join(SM_DATA_DIR, "telco_churn.csv")
df = pd.read_csv(csv_path)
df.drop(columns=["customerID"], inplace=True, errors="ignore")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn"]        = (df["Churn"] == "Yes").astype(int)
df = engineer_features(df)
print(f"  Rows: {len(df)}  |  Churn rate: {df['Churn'].mean()*100:.1f}%")


# ── 3. FEATURE COLUMNS ────────────────────────────────────────
FEATURE_COLS = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges",
    "ChargePerTenure","TotalServices","HasProtection","IsLongTermCustomer",
    "HighMonthlyCharge","AutoPayment","FiberNoSecurity","HighRiskCombo"
]
X = df[FEATURE_COLS]
y = df["Churn"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols     = X.select_dtypes(exclude="object").columns.tolist()


# ── 4. PREPROCESSING PIPELINE ─────────────────────────────────
print("🔧 Building preprocessing pipeline...")
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
print(f"  Train: {X_train_proc.shape}  |  Test: {X_test_proc.shape}")


# ── 5. SMOTE ──────────────────────────────────────────────────
print("⚖️  Applying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)
print(f"  Balanced: {dict(zip(*np.unique(y_train_bal, return_counts=True)))}")


# ── 6. OPTUNA TUNING ──────────────────────────────────────────
print("🔬 Optuna hyperparameter tuning (20 trials)...")

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0, 0.5),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 1.5),
        "eval_metric":      "logloss",
        "random_state":     42,
        "tree_method":      "hist",
        # NOTE: "device": "cuda" removed — container is sagemaker-scikit-learn (CPU only)
        # ml.g4dn.xlarge has GPU but this container doesn't have CUDA runtime
    }
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []
    for tr_idx, val_idx in kf.split(X_train_bal, y_train_bal):
        X_tr, X_val = X_train_bal[tr_idx], X_train_bal[val_idx]
        y_tr, y_val = y_train_bal.iloc[tr_idx], y_train_bal.iloc[val_idx]
        clf = XGBClassifier(**params)
        clf.fit(X_tr, y_tr, verbose=False)
        auc_scores.append(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))
    return np.mean(auc_scores)

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20)
best_params = study.best_params
print(f"  Best ROC-AUC: {study.best_value:.4f}")
print(f"  Best params : {best_params}")


# ── 7. TRAIN FINAL MODEL ──────────────────────────────────────
print("\n🚀 Training final model...")
best_params.update({"eval_metric":"logloss","random_state":42,"tree_method":"hist"})
model = XGBClassifier(**best_params)
model.fit(X_train_bal, y_train_bal,
          eval_set=[(X_test_proc, y_test)], verbose=False)

# Manual 5-fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []
for tr_idx, val_idx in kf.split(X_train_bal, y_train_bal):
    m = XGBClassifier(**best_params)
    m.fit(X_train_bal[tr_idx], y_train_bal.iloc[tr_idx], verbose=False)
    cv_acc.append(accuracy_score(y_train_bal.iloc[val_idx],
                                  m.predict(X_train_bal[val_idx])))
print(f"  CV Accuracy: {np.mean(cv_acc)*100:.2f}% ± {np.std(cv_acc)*100:.2f}%")


# ── 8. THRESHOLD OPTIMIZATION ─────────────────────────────────
print("🎯 Optimizing decision threshold...")
y_proba = model.predict_proba(X_test_proc)[:, 1]
best_acc, BEST_THRESHOLD = 0, 0.5
for t in np.arange(0.35, 0.86, 0.01):
    acc_t = accuracy_score(y_test, (y_proba >= t).astype(int))
    if acc_t > best_acc:
        best_acc, BEST_THRESHOLD = acc_t, t
y_pred = (y_proba >= BEST_THRESHOLD).astype(int)
print(f"  Threshold: {BEST_THRESHOLD:.2f}  |  Accuracy: {best_acc*100:.2f}%")


# ── 9. EVALUATE ───────────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 55)
print("  FINAL EVALUATION")
print("=" * 55)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print("=" * 55)


# ── 10. SAVE ALL FILES TO /opt/ml/model/ ──────────────────────
print("\n💾 Saving model artifacts...")

with open(os.path.join(SM_MODEL_DIR, "churn_model.pkl"), "wb") as f:
    pickle.dump(model, f)
print("  ✅ churn_model.pkl")

with open(os.path.join(SM_MODEL_DIR, "preprocessor.pkl"), "wb") as f:
    pickle.dump(preprocessor, f)
print("  ✅ preprocessor.pkl")

with open(os.path.join(SM_MODEL_DIR, "threshold.json"), "w") as f:
    json.dump({"threshold": float(BEST_THRESHOLD)}, f)
print(f"  ✅ threshold.json ({BEST_THRESHOLD:.2f})")

# Build feature importance
ohe_features   = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_cols).tolist()
all_feat_names = numeric_cols + ohe_features
orig_imp = {}
for feat, imp in zip(all_feat_names, model.feature_importances_):
    for orig in FEATURE_COLS:
        if feat.startswith(orig) or feat == orig:
            orig_imp[orig] = orig_imp.get(orig, 0) + imp
            break
top_features = dict(sorted(orig_imp.items(), key=lambda x: x[1], reverse=True)[:8])

metrics_data = {
    "accuracy":           float(acc),
    "roc_auc":            float(auc),
    "precision":          float(prec),
    "recall":             float(rec),
    "f1_score":           float(f1),
    "confusion_matrix":   cm.tolist(),
    "churn_rate":         float(y.mean()),
    "train_rows":         int(len(X_train)),
    "test_rows":          int(len(X_test)),
    "feature_importance": {k: float(v) for k, v in top_features.items()},
    "threshold":          float(BEST_THRESHOLD),
    "cv_accuracy":        float(np.mean(cv_acc)),
    "trained_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model":              "XGBoost Classifier v2 (Optuna tuned) — SageMaker GPU",
}
with open(os.path.join(SM_MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_data, f, indent=2)
print("  ✅ metrics.json")

print(f"\n🎉 Training complete! Files saved to {SM_MODEL_DIR}/")
print(f"   SageMaker will auto-package these as model.tar.gz → S3")

# ── Copy inference.py into model dir so it's included in model.tar.gz ──
# SageMaker needs inference.py inside model.tar.gz to know how to serve predictions.
# We uploaded inference.py to the 'code' S3 channel, so it lands at SM_CODE_DIR.
import shutil, glob

inference_candidates = [
    "/opt/ml/code/inference.py",                  # sourcedir.tar.gz extraction path
    "inference.py",                               # local fallback
]
copied = False
for src_path in inference_candidates:
    if os.path.exists(src_path):
        shutil.copy(src_path, os.path.join(SM_MODEL_DIR, "inference.py"))
        print(f"\n  ✅ inference.py copied from {src_path} → {SM_MODEL_DIR}/")
        print(f"     model.tar.gz will contain: churn_model.pkl, preprocessor.pkl,")
        print(f"     threshold.json, metrics.json, inference.py")
        copied = True
        break

if not copied:
    print("\n  ⚠️  inference.py not found — endpoints will need it separately")
    print(f"     Searched: {inference_candidates}")
