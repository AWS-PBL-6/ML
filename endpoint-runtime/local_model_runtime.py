"""Pure-Python local runtime for the proxy breakage model artifact."""
from __future__ import annotations

import json
import math
import os
import sys
import time

SCHEMA_VERSION = "1.0.0"
RAW_RUNTIME_FEATURES = (
    "peakAmplitude",
    "energy",
    "durationMs",
    "riseTimeMs",
    "dominantFrequencyHz",
    "temperatureC",
    "humidityPct",
)
_XGBOOST_MODULE = None
_XGBOOST_LOAD_FAILED = False
_XGBOOST_BOOSTERS: dict[str, object] = {}


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def load_artifact(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def _ensure_optional_runtime_deps() -> None:
    vendor_dir = os.path.join(_repo_root(), ".vendor_ml")
    if os.path.isdir(vendor_dir) and vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    mpl_dir = os.path.join(_repo_root(), ".mplconfig")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)


def _load_xgboost_module():
    global _XGBOOST_MODULE, _XGBOOST_LOAD_FAILED
    if _XGBOOST_MODULE is not None:
        return _XGBOOST_MODULE
    if _XGBOOST_LOAD_FAILED:
        return None
    try:
        _ensure_optional_runtime_deps()
        import xgboost  # type: ignore
    except Exception:
        _XGBOOST_LOAD_FAILED = True
        return None
    _XGBOOST_MODULE = xgboost
    return _XGBOOST_MODULE


def _augment_features(model_input: dict) -> dict:
    features = {key: float(value) for key, value in model_input.items()}
    if all(name in features for name in RAW_RUNTIME_FEATURES):
        duration = max(float(features["durationMs"]), 1e-6)
        features.setdefault("logEnergy", math.log1p(float(features["energy"])))
        features.setdefault("energyPerDuration", float(features["energy"]) / duration)
        features.setdefault("riseRatio", float(features["riseTimeMs"]) / duration)
        features.setdefault("freqXRise", float(features["dominantFrequencyHz"]) * float(features["riseTimeMs"]))
        features.setdefault("tempXHumidity", float(features["temperatureC"]) * float(features["humidityPct"]))
    return features


def _standardize_vector(logreg: dict, features: dict, clip: bool = True) -> list[float]:
    names = logreg["featureNames"]
    means = logreg["means"]
    scales = logreg["scales"]
    minima = logreg["minima"]
    maxima = logreg["maxima"]
    values = []
    for idx, name in enumerate(names):
        raw = float(features[name])
        if clip:
            raw = min(max(raw, float(minima[idx])), float(maxima[idx]))
        scale = scales[idx] or 1.0
        values.append((raw - float(means[idx])) / scale)
    return values


def score_logreg(logreg: dict, features: dict) -> float:
    z = float(logreg["intercept"])
    names = logreg["featureNames"]
    weights = logreg["weights"]
    vector = _standardize_vector(logreg, features, clip=True)
    for idx, _ in enumerate(names):
        z += float(weights[idx]) * vector[idx]
    return _sigmoid(z)


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean_knn(query: list[float], examples: list[dict], top_k: int) -> tuple[float, list[tuple[float, dict]]]:
    ranked = sorted(((_euclidean(query, example["vector"]), example) for example in examples), key=lambda item: item[0])
    if not ranked:
        return 0.0, []
    k = min(top_k, len(ranked))
    chosen = ranked[:k]
    return sum(distance for distance, _ in chosen) / k, chosen


def _hybrid_components(model: dict, features: dict, top_k: int = 3) -> dict:
    logreg = model["logreg"]
    vector = _standardize_vector(logreg, features, clip=True)
    breakage_score = score_logreg(logreg, features)

    examples = model["coreExamples"]
    danger_examples = [example for example in examples if int(example["damageType"]) == 2]
    normal_examples = [example for example in examples if int(example["damageType"]) == 0]

    similarity_k = int(model.get("similarityK", 5))
    danger_distance, danger_ranked = _mean_knn(vector, danger_examples, similarity_k)
    normal_distance, normal_ranked = _mean_knn(vector, normal_examples, similarity_k)
    danger_closeness = 1.0 / (1.0 + danger_distance)
    normal_closeness = 1.0 / (1.0 + normal_distance)
    closeness_total = max(danger_closeness + normal_closeness, 1e-9)
    danger_similarity = danger_closeness / closeness_total
    normal_similarity = normal_closeness / closeness_total

    anomaly_p95 = float(model.get("anomalyP95", 0.0))
    anomaly_p99 = float(model.get("anomalyP99", max(anomaly_p95 + 1e-6, 1e-6)))
    if normal_distance <= anomaly_p95:
        anomaly_score = 0.0
    elif normal_distance >= anomaly_p99:
        anomaly_score = 1.0
    else:
        anomaly_score = (normal_distance - anomaly_p95) / max(anomaly_p99 - anomaly_p95, 1e-6)

    manifold_distance, _ = _mean_knn(vector, examples, similarity_k)
    ood_p95 = float(model.get("oodP95", 0.0))
    ood_p99 = float(model.get("oodP99", max(ood_p95 + 1e-6, 1e-6)))
    if manifold_distance <= ood_p95:
        ood_score = 0.0
    elif manifold_distance >= ood_p99:
        ood_score = 1.0
    else:
        ood_score = (manifold_distance - ood_p95) / max(ood_p99 - ood_p95, 1e-6)

    raw = float(model.get("weightBreakage", 0.6)) * breakage_score + float(
        model.get("weightDangerSimilarity", 0.4)
    ) * danger_similarity
    hybrid_score = max(0.0, min(1.0, raw * (1.0 - float(model.get("oodPenalty", 0.3)) * ood_score)))

    return {
        "breakageScore": breakage_score,
        "dangerSimilarity": danger_similarity,
        "normalSimilarity": normal_similarity,
        "normalDistance": normal_distance,
        "anomalyScore": anomaly_score,
        "oodScore": ood_score,
        "manifoldDistance": manifold_distance,
        "hybridScore": hybrid_score,
        "topDangerExamples": [
            {
                "exampleId": example["exampleId"],
                "damageType": example["damageType"],
                "distance": round(distance, 6),
                "features": example["features"],
            }
            for distance, example in danger_ranked[:top_k]
        ],
        "topNormalExamples": [
            {
                "exampleId": example["exampleId"],
                "damageType": example["damageType"],
                "distance": round(distance, 6),
                "features": example["features"],
            }
            for distance, example in normal_ranked[:top_k]
        ],
    }


def _xgboost_candidate_meta(artifact: dict) -> dict | None:
    candidate_models = artifact.get("candidateModels") or {}
    xgboost_meta = candidate_models.get("xgboost")
    if not isinstance(xgboost_meta, dict):
        return None
    if xgboost_meta.get("status") != "ok":
        return None
    return xgboost_meta


def _load_xgboost_booster(artifact: dict, artifact_path: str):
    xgboost_meta = _xgboost_candidate_meta(artifact)
    if xgboost_meta is None:
        return None
    model_path = xgboost_meta.get("modelPath")
    if not model_path:
        return None
    resolved_path = os.path.join(os.path.dirname(os.path.abspath(artifact_path)), model_path)
    if resolved_path in _XGBOOST_BOOSTERS:
        return _XGBOOST_BOOSTERS[resolved_path]
    xgboost = _load_xgboost_module()
    if xgboost is None:
        return None
    booster = xgboost.Booster()
    booster.load_model(resolved_path)
    _XGBOOST_BOOSTERS[resolved_path] = booster
    return booster


def score_xgboost_candidate(artifact: dict, artifact_path: str, features: dict) -> float | None:
    xgboost_meta = _xgboost_candidate_meta(artifact)
    if xgboost_meta is None:
        return None
    booster = _load_xgboost_booster(artifact, artifact_path)
    if booster is None:
        return None
    feature_names = xgboost_meta["featureNames"]
    row = [[float(features[name]) for name in feature_names]]
    xgboost = _load_xgboost_module()
    if xgboost is None:
        return None
    matrix = xgboost.DMatrix(row, feature_names=feature_names)
    return float(booster.predict(matrix)[0])


def score_model(model: dict, features: dict) -> float:
    if model.get("modelType") == "hybrid_proxy_breakage_v2":
        return _hybrid_components(model, features, top_k=1)["hybridScore"]
    logreg = model
    return score_logreg(logreg, features)


def explain_request(request: dict, artifact_path: str, top_k: int = 3) -> dict:
    artifact = load_artifact(artifact_path)
    model = artifact["model"]
    features = _augment_features(request.get("modelInput", {}))
    if model.get("modelType") != "hybrid_proxy_breakage_v2":
        return {"hybridAvailable": False}
    details = _hybrid_components(model, features, top_k=top_k)
    xgboost_score = score_xgboost_candidate(artifact, artifact_path, features)
    variant = os.environ.get("LOCAL_ML_MODEL_VARIANT", "").strip() or (
        artifact.get("runtime", {}) or {}
    ).get("defaultVariant", "hybrid")
    details["xgboostScore"] = xgboost_score
    details["selectedVariant"] = variant if xgboost_score is not None or variant == "hybrid" else "hybrid"
    details["selectedScore"] = (
        xgboost_score if details["selectedVariant"] == "xgboost_candidate" and xgboost_score is not None else details["hybridScore"]
    )
    return details


def policy_probabilities(score: float, warn_threshold: float, danger_threshold: float) -> dict:
    warn = max(1e-6, float(warn_threshold))
    danger = max(warn + 1e-6, float(danger_threshold))
    centre = (warn + danger) / 2.0
    half_width = max((danger - warn) / 2.0, 1e-6)

    normal_raw = max(0.0, (warn - score) / warn)
    caution_raw = max(0.0, 1.0 - abs(score - centre) / half_width) if warn <= score <= danger else 0.0
    danger_raw = max(0.0, (score - warn) / max(1.0 - warn, 1e-6))
    probs = _softmax([normal_raw * 4.0, caution_raw * 4.0, danger_raw * 4.0])
    return {
        "NORMAL": round(probs[0], 3),
        "CAUTION": round(probs[1], 3),
        "DANGER": round(probs[2], 3),
    }


def classify(score: float, warn_threshold: float, danger_threshold: float) -> str:
    if score >= danger_threshold:
        return "DANGER"
    if score >= warn_threshold:
        return "CAUTION"
    return "NORMAL"


def _align_probs(level: str, probs: dict) -> dict:
    values = {key: float(probs.get(key, 0.0)) for key in ("NORMAL", "CAUTION", "DANGER")}
    top = max(values.values()) if values else 0.0
    if values.get(level, 0.0) >= top:
        return {key: round(value, 3) for key, value in values.items()}
    values[level] = top + 1e-3
    total = sum(values.values()) or 1.0
    return {key: round(value / total, 3) for key, value in values.items()}


def _selected_variant(artifact: dict) -> str:
    requested = os.environ.get("LOCAL_ML_MODEL_VARIANT", "").strip()
    if requested:
        return requested
    runtime_cfg = artifact.get("runtime") or {}
    return runtime_cfg.get("defaultVariant", "hybrid")


def predict_request(request: dict, artifact_path: str) -> dict:
    started = time.time()
    artifact = load_artifact(artifact_path)
    model = artifact["model"]
    model_input = request.get("modelInput", {})
    features = _augment_features(model_input)

    feature_spec = artifact.get("featureSpec") or {}
    raw_required = feature_spec.get("runtimeFeatureNames") or RAW_RUNTIME_FEATURES
    missing = [name for name in raw_required if name not in model_input]
    if missing:
        raise ValueError(f"missing modelInput features: {', '.join(missing)}")

    hybrid_details = _hybrid_components(model, features, top_k=3) if model.get("modelType") == "hybrid_proxy_breakage_v2" else None
    hybrid_score = hybrid_details["hybridScore"] if hybrid_details is not None else score_model(model, features)
    xgboost_score = score_xgboost_candidate(artifact, artifact_path, features)
    variant = _selected_variant(artifact)
    if variant == "xgboost_candidate" and xgboost_score is not None:
        score = xgboost_score
        model_version = f"{artifact.get('modelVersion', 'ae-proxy-breakage-local')}+xgboost"
    else:
        variant = "hybrid"
        score = hybrid_score
        model_version = artifact.get("modelVersion", "ae-proxy-breakage-local")
    policy = artifact["policy"]
    warn_threshold = float(policy["warnThreshold"])
    danger_threshold = float(policy["dangerThreshold"])
    probs = policy_probabilities(score, warn_threshold, danger_threshold)
    level = classify(score, warn_threshold, danger_threshold)

    details = hybrid_details
    if model.get("modelType") == "hybrid_proxy_breakage_v2" and details is not None:
        if details["oodScore"] >= float(policy.get("oodBlockDangerThreshold", 0.85)) and score < float(
            policy.get("highConfidenceDanger", 0.75)
        ):
            level = "CAUTION"
        elif details["oodScore"] >= float(policy.get("oodCautionThreshold", 0.60)) and level == "NORMAL":
            level = "CAUTION"
        if details["anomalyScore"] >= float(policy.get("anomalyBlockNormalThreshold", 0.85)) and level == "NORMAL":
            level = "CAUTION"
        elif details["anomalyScore"] >= float(policy.get("anomalyCautionThreshold", 0.60)) and level == "NORMAL":
            level = "CAUTION"
    probs = _align_probs(level, probs)

    return {
        "schemaVersion": artifact.get("schemaVersion", SCHEMA_VERSION),
        "requestId": request.get("requestId", ""),
        "traceId": request.get("traceId", ""),
        "riskLevel": level,
        "riskScore": round(score, 3),
        "classProbabilities": probs,
        "modelVersion": model_version,
        "riskSignals": {
            "selectedVariant": variant,
            "hybridScore": round(hybrid_score, 6),
            "xgboostScore": round(xgboost_score, 6) if xgboost_score is not None else None,
            "oodScore": round(details["oodScore"], 6) if details is not None else None,
            "anomalyScore": round(details["anomalyScore"], 6) if details is not None else None,
        },
        "inferenceMs": round((time.time() - started) * 1000, 2),
    }
