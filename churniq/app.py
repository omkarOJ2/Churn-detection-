from flask import Flask, render_template, request, jsonify
import os, json, warnings
from datetime import datetime

warnings.filterwarnings("ignore")
app = Flask(__name__)

# ── SageMaker config ─────────────────────────────────────────
USE_SAGEMAKER        = os.environ.get("USE_SAGEMAKER", "true").lower() == "true"
REALTIME_ENDPOINT    = os.environ.get("REALTIME_ENDPOINT", "churniq-realtime")
SERVERLESS_ENDPOINT  = os.environ.get("SERVERLESS_ENDPOINT", "churniq-serverless")
AWS_REGION           = os.environ.get("AWS_REGION", "ap-south-1")
S3_BUCKET            = os.environ.get("S3_BUCKET", "chrun-detection")
S3_MODEL_KEY         = os.environ.get("S3_MODEL_KEY", "training-output/churniq-train-1/output/model.tar.gz")
## ── Load metrics from S3 ─────────────────────────────────────
def load_metrics():
    try:
        import tarfile, io
        s3  = boto3.client("s3", region_name=AWS_REGION)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY)
        tar = tarfile.open(fileobj=io.BytesIO(obj["Body"].read()))
        f   = tar.extractfile("metrics.json")
        print("✅ Metrics loaded from S3")
        return json.loads(f.read().decode())
    except Exception as e:
        print(f"⚠️ Could not load metrics from S3: {e} — using defaults")
        return {}

METRICS = load_metrics()

# ── SageMaker clients ─────────────────────────────────────────
sm_runtime = None
sm_client  = None
try:
    import boto3
    sm_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    sm_client  = boto3.client("sagemaker", region_name=AWS_REGION)
    print(f"🌐 SageMaker Mode | Region: {AWS_REGION}")
    print(f"   Realtime:   {REALTIME_ENDPOINT}")
    print(f"   Serverless: {SERVERLESS_ENDPOINT}")
except Exception as e:
    print(f"⚠️ boto3 error: {e}")

def get_active_endpoint():
    """Try realtime first — if InService use it, else fall back to serverless."""
    if sm_client:
        try:
            resp = sm_client.describe_endpoint(EndpointName=REALTIME_ENDPOINT)
            if resp["EndpointStatus"] == "InService":
                print(f"⚡ Using realtime endpoint: {REALTIME_ENDPOINT}")
                return REALTIME_ENDPOINT, "realtime"
        except Exception:
            pass
    print(f"☁️ Using serverless endpoint: {SERVERLESS_ENDPOINT}")
    return SERVERLESS_ENDPOINT, "serverless"

def invoke_with_retry(endpoint, payload, retries=2):
    """Invoke endpoint with retry — handles serverless cold start timeouts."""
    import time
    last_error = None
    for attempt in range(retries):
        try:
            r = sm_runtime.invoke_endpoint(
                EndpointName=endpoint,
                ContentType="application/json",
                Body=json.dumps(payload)
            )
            return json.loads(r["Body"].read().decode())
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(3)
    raise last_error

# ── Page routes ──────────────────────────────────────────────
@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=METRICS)

@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/about")
def about():
    return render_template("about.html")

# ── API: single prediction ────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        if not sm_runtime:
            return jsonify({"error": "SageMaker client not available."}), 500

        endpoint, endpoint_type = get_active_endpoint()
        result = invoke_with_retry(endpoint, data)
        if "timestamp" not in result:
            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["endpoint_type"] = endpoint_type
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# # ── API: batch prediction ─────────────────────────────────────
# @app.route("/api/batch_predict", methods=["POST"])
# def batch_predict():
#     try:
#         import pandas as pd
#         file = request.files.get("file")
#         if not file:
#             return jsonify({"error": "No file uploaded"}), 400
#         if not sm_runtime:
#             return jsonify({"error": "SageMaker client not available."}), 500

#         endpoint, endpoint_type = get_active_endpoint()
#         df = pd.read_csv(file)
#         results = []
#         for i, row in df.iterrows():
#             try:
#                 pred = invoke_with_retry(endpoint, row.to_dict())
#                 results.append({
#                     "row": i + 1,
#                     "churn": pred.get("churn", 0),
#                     "probability": pred.get("probability", 0),
#                     "risk_level": pred.get("risk_level", "Unknown"),
#                     "endpoint_type": endpoint_type
#                 })
#             except Exception:
#                 results.append({"row": i+1, "churn": 0, "probability": 0, "risk_level": "Error"})

#         churners = sum(r["churn"] for r in results)
#         avg_prob = round(sum(r["probability"] for r in results) / len(results), 2) if results else 0
#         summary  = {"total": len(results), "churners": churners,
#                     "retention": len(results) - churners, "avg_probability": avg_prob}
#         return jsonify({"results": results, "summary": summary, "endpoint_type": endpoint_type})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# ── API: metrics ─────────────────────────────────────────────
@app.route("/api/metrics")
def get_metrics():
    return jsonify(METRICS)

# ── API: mode ─────────────────────────────────────────────────
@app.route("/api/mode")
def get_mode():
    endpoint, endpoint_type = get_active_endpoint() if sm_runtime else (SERVERLESS_ENDPOINT, "serverless")
    return jsonify({
        "sagemaker_mode": sm_runtime is not None,
        "active_endpoint": endpoint,
        "endpoint_type": endpoint_type,
        "realtime_endpoint": REALTIME_ENDPOINT,
        "serverless_endpoint": SERVERLESS_ENDPOINT,
        "region": AWS_REGION
    })

# ── API: health check ────────────────────────────────────────
@app.route("/api/health")
def health_check():
    status = {"app": "ok", "sagemaker": sm_runtime is not None, "region": AWS_REGION}
    if sm_client:
        for name, ep in [("realtime", REALTIME_ENDPOINT), ("serverless", SERVERLESS_ENDPOINT)]:
            try:
                r = sm_client.describe_endpoint(EndpointName=ep)
                status[name] = r["EndpointStatus"]
            except Exception:
                status[name] = "NotFound"
    return jsonify(status)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
