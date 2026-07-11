"""SageMaker script-mode inference handler for the port-window AE model.

Serves the ``port_window_xgb_v1`` artifact on a SageMaker (serverless) endpoint.
It mirrors ``ml/endpoint-runtime/local_model_runtime.py::_predict_port_window`` so
the endpoint returns the same ``inference-response.v1`` the backend already
consumes: the backend sends the full ``inference-request.v1`` (its ``modelInput``
is the assembled event-window feature vector) and gets risk level/score back.

Runs on the SageMaker scikit-learn framework container; ``requirements.txt``
pins ``xgboost==3.3.0`` to match the booster's saved format ([3,3,0]).
"""
import json
import math
import os
import time

import xgboost as xgb


def _softmax(values):
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def _classify(score, warn, danger):
    if score >= danger:
        return "DANGER"
    if score >= warn:
        return "CAUTION"
    return "NORMAL"


def _align_probs(level, probs):
    """Ensure the reported class-probabilities agree with the chosen level."""
    values = {k: float(probs.get(k, 0.0)) for k in ("NORMAL", "CAUTION", "DANGER")}
    top = max(values.values()) if values else 0.0
    if values.get(level, 0.0) >= top:
        return {k: round(v, 3) for k, v in values.items()}
    values[level] = top + 1e-3
    total = sum(values.values()) or 1.0
    return {k: round(v / total, 3) for k, v in values.items()}


# --- SageMaker serving contract -------------------------------------------
def model_fn(model_dir):
    with open(os.path.join(model_dir, "model_artifact.json"), "r", encoding="utf-8") as fh:
        artifact = json.load(fh)
    meta = artifact["model"]["xgboost"]
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, meta["modelPath"]))
    return {"artifact": artifact, "booster": booster}


def input_fn(request_body, content_type="application/json"):
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    return json.loads(request_body)


def predict_fn(request, model):
    started = time.time()
    artifact = model["artifact"]
    booster = model["booster"]
    meta = artifact["model"]["xgboost"]
    feat_names = meta["featureNames"]

    model_input = request.get("modelInput", {})
    features = {k: float(v) for k, v in model_input.items()}
    missing = [n for n in feat_names if n not in features]
    if missing:
        raise ValueError("missing modelInput features: " + ", ".join(missing))

    row = [[features[n] for n in feat_names]]
    matrix = xgb.DMatrix(row, feature_names=feat_names)
    margins = list(booster.predict(matrix, output_margin=True)[0])

    temperature = float(meta.get("temperature", 1.0)) or 1.0
    probs = _softmax([m / temperature for m in margins])
    class_order = meta.get("classOrder", ["NORMAL", "CAUTION", "DANGER"])
    class_probs = {lvl: round(float(probs[i]), 3) for i, lvl in enumerate(class_order)}

    weights = artifact["model"].get("severityWeights", [0.0, 0.5, 1.0])
    severity = sum(float(weights[i]) * float(probs[i]) for i in range(len(probs)))

    policy = artifact["policy"]
    warn = float(policy["warnThreshold"])
    danger = float(policy["dangerThreshold"])
    level = _classify(severity, warn, danger)
    class_probs = _align_probs(level, class_probs)

    return {
        "schemaVersion": artifact.get("schemaVersion", "1.0.0"),
        "requestId": request.get("requestId", ""),
        "traceId": request.get("traceId", ""),
        "riskLevel": level,
        "riskScore": round(severity, 3),
        "classProbabilities": class_probs,
        "modelVersion": artifact.get("modelVersion", "ae-port-window-xgb"),
        "riskSignals": {
            "selectedVariant": "port_window_xgb",
            "severity": round(severity, 6),
            "pNormal": class_probs["NORMAL"],
            "pCaution": class_probs["CAUTION"],
            "pDanger": class_probs["DANGER"],
        },
        "inferenceMs": round((time.time() - started) * 1000, 2),
    }


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
