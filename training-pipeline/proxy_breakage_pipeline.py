#!/usr/bin/env python3
"""Train the AE proxy breakage model and emit deployable artifacts.

This pipeline keeps the model objective narrow:
  - core labels: 0 (No Damage) vs 2 (Fiber Breakage)
  - shadow labels: 1 (Matrix Cracking), 3 (Delamination)

The trained model is a dependency-free weighted logistic regression over the
runtime-compatible feature subset. A CatBoost comparison hook is included when
the optional dependency is available, but the emitted artifact stays portable
and is always served by the local pure-Python runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "ae-proxy-breakage-hybrid-v2"
CORE_LABELS = (0, 2)
SHADOW_LABELS = (1, 3)
SOURCE_TO_RUNTIME = {
    "Peak_Amplitude_dB": "peakAmplitude",
    "Energy_Release_J": "energy",
    "Signal_Duration_ms": "durationMs",
    "Rise_Time_ms": "riseTimeMs",
    "Frequency_Spectrum_Hz": "dominantFrequencyHz",
    "Temperature_C": "temperatureC",
    "Humidity_%": "humidityPct",
}
RUNTIME_FEATURES = tuple(SOURCE_TO_RUNTIME.values())
DERIVED_FEATURES = {
    "logEnergy": "log(1 + energy)",
    "energyPerDuration": "energy / durationMs",
    "riseRatio": "riseTimeMs / durationMs",
    "freqXRise": "dominantFrequencyHz * riseTimeMs",
    "tempXHumidity": "temperatureC * humidityPct",
}
FEATURE_SETS = {
    "baseline": list(RUNTIME_FEATURES),
    "engineered": [*RUNTIME_FEATURES, *DERIVED_FEATURES.keys()],
}


@dataclass
class PreparedRow:
    damage_type: int
    features: dict


@dataclass(frozen=True)
class HybridParams:
    weight_breakage: float
    weight_danger_similarity: float
    ood_penalty: float


@dataclass
class StandardizedLogReg:
    feature_names: list[str]
    means: list[float]
    scales: list[float]
    minima: list[float]
    maxima: list[float]
    weights: list[float]
    intercept: float

    def standardize_vector(self, feature_map: dict, clip: bool = True) -> list[float]:
        values = []
        for idx, name in enumerate(self.feature_names):
            raw = float(feature_map[name])
            if clip:
                raw = min(max(raw, self.minima[idx]), self.maxima[idx])
            scale = self.scales[idx] or 1.0
            values.append((raw - self.means[idx]) / scale)
        return values

    def predict_score(self, feature_map: dict) -> float:
        z = self.intercept
        for weight, value in zip(self.weights, self.standardize_vector(feature_map, clip=True)):
            z += weight * value
        return _sigmoid(z)

    def to_dict(self) -> dict:
        return {
            "modelType": "weighted_logistic_regression",
            "featureNames": self.feature_names,
            "means": self.means,
            "scales": self.scales,
            "minima": self.minima,
            "maxima": self.maxima,
            "weights": self.weights,
            "intercept": self.intercept,
        }


@dataclass
class HybridEngine:
    logreg: StandardizedLogReg
    core_examples: list[dict]
    similarity_k: int
    ood_p95: float
    ood_p99: float
    anomaly_p95: float
    anomaly_p99: float
    weight_breakage: float = 0.6
    weight_danger_similarity: float = 0.4
    ood_penalty: float = 0.3

    def _examples_for_damage(self, damage_type: int) -> list[dict]:
        return [example for example in self.core_examples if example["damageType"] == damage_type]

    def _mean_knn(self, query_vector: list[float], examples: list[dict], top_k: int) -> tuple[float, list[tuple[float, dict]]]:
        if not examples:
            return 0.0, []
        ranked = sorted(
            ((_euclidean(query_vector, example["vector"]), example) for example in examples),
            key=lambda item: item[0],
        )
        k = min(top_k, len(ranked))
        chosen = ranked[:k]
        return sum(distance for distance, _ in chosen) / k, chosen

    def _ood_score(self, query_vector: list[float]) -> tuple[float, float]:
        mean_distance, _ = self._mean_knn(query_vector, self.core_examples, self.similarity_k)
        if mean_distance <= self.ood_p95:
            return 0.0, mean_distance
        if mean_distance >= self.ood_p99:
            return 1.0, mean_distance
        span = max(self.ood_p99 - self.ood_p95, 1e-6)
        return (mean_distance - self.ood_p95) / span, mean_distance

    def nearest_examples(self, feature_map: dict, damage_type: int, top_k: int = 3) -> list[dict]:
        query_vector = self.logreg.standardize_vector(feature_map, clip=True)
        _, ranked = self._mean_knn(query_vector, self._examples_for_damage(damage_type), top_k)
        return [
            {
                "exampleId": example["exampleId"],
                "damageType": example["damageType"],
                "distance": round(distance, 6),
                "features": example["features"],
            }
            for distance, example in ranked
        ]

    def predict(self, feature_map: dict, top_k: int = 3) -> dict:
        breakage_score = self.logreg.predict_score(feature_map)
        query_vector = self.logreg.standardize_vector(feature_map, clip=True)

        danger_examples = self._examples_for_damage(2)
        normal_examples = self._examples_for_damage(0)
        danger_distance, danger_ranked = self._mean_knn(query_vector, danger_examples, self.similarity_k)
        normal_distance, normal_ranked = self._mean_knn(query_vector, normal_examples, self.similarity_k)

        danger_closeness = 1.0 / (1.0 + danger_distance)
        normal_closeness = 1.0 / (1.0 + normal_distance)
        closeness_total = max(danger_closeness + normal_closeness, 1e-9)
        danger_similarity = danger_closeness / closeness_total
        normal_similarity = normal_closeness / closeness_total

        if normal_distance <= self.anomaly_p95:
            anomaly_score = 0.0
        elif normal_distance >= self.anomaly_p99:
            anomaly_score = 1.0
        else:
            anomaly_score = (normal_distance - self.anomaly_p95) / max(self.anomaly_p99 - self.anomaly_p95, 1e-6)

        ood_score, manifold_distance = self._ood_score(query_vector)
        hybrid_raw = (
            self.weight_breakage * breakage_score
            + self.weight_danger_similarity * danger_similarity
        )
        hybrid_score = max(0.0, min(1.0, hybrid_raw * (1.0 - self.ood_penalty * ood_score)))

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

    def to_dict(self) -> dict:
        return {
            "modelType": "hybrid_proxy_breakage_v2",
            "logreg": self.logreg.to_dict(),
            "coreExamples": self.core_examples,
            "similarityK": self.similarity_k,
            "oodP95": self.ood_p95,
            "oodP99": self.ood_p99,
            "anomalyP95": self.anomaly_p95,
            "anomalyP99": self.anomaly_p99,
            "weightBreakage": self.weight_breakage,
            "weightDangerSimilarity": self.weight_danger_similarity,
            "oodPenalty": self.ood_penalty,
        }


def augment_feature_map(base_features: dict) -> dict:
    features = {name: float(value) for name, value in base_features.items()}
    duration = max(float(features["durationMs"]), 1e-6)
    features["logEnergy"] = math.log1p(float(features["energy"]))
    features["energyPerDuration"] = float(features["energy"]) / duration
    features["riseRatio"] = float(features["riseTimeMs"]) / duration
    features["freqXRise"] = float(features["dominantFrequencyHz"]) * float(features["riseTimeMs"])
    features["tempXHumidity"] = float(features["temperatureC"]) * float(features["humidityPct"])
    return features


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _ema(values: list[float], alpha: float) -> float:
    if not values:
        return 0.0
    acc = values[0]
    for value in values[1:]:
        acc = alpha * value + (1.0 - alpha) * acc
    return acc


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _std(values: Iterable[float], mean_value: float) -> float:
    values = list(values)
    variance = sum((x - mean_value) ** 2 for x in values) / len(values)
    return math.sqrt(variance) or 1.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty list")
    ordered = sorted(values)
    if q <= 0:
        return ordered[0]
    if q >= 1:
        return ordered[-1]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def load_rows(csv_path: str) -> list[PreparedRow]:
    rows: list[PreparedRow] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            base_features = {
                runtime_name: float(raw[source_name])
                for source_name, runtime_name in SOURCE_TO_RUNTIME.items()
            }
            rows.append(PreparedRow(damage_type=int(raw["Damage_Type"]), features=augment_feature_map(base_features)))
    return rows


def split_rows(rows: list[PreparedRow]) -> tuple[list[PreparedRow], list[PreparedRow]]:
    core = [row for row in rows if row.damage_type in CORE_LABELS]
    shadow = [row for row in rows if row.damage_type in SHADOW_LABELS]
    return core, shadow


def write_dataset_csv(path: str, rows: list[PreparedRow], feature_names: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["damageType", *feature_names])
        for row in rows:
            writer.writerow([row.damage_type, *(row.features[name] for name in feature_names)])


def stratified_folds(rows: list[PreparedRow], folds: int, seed: int) -> list[list[PreparedRow]]:
    rng = random.Random(seed)
    neg = [row for row in rows if row.damage_type == 0]
    pos = [row for row in rows if row.damage_type == 2]
    rng.shuffle(neg)
    rng.shuffle(pos)
    buckets = [[] for _ in range(folds)]
    for idx, row in enumerate(pos):
        buckets[idx % folds].append(row)
    for idx, row in enumerate(neg):
        buckets[idx % folds].append(row)
    return buckets


def fit_logreg(train_rows: list[PreparedRow], feature_names: list[str], epochs: int = 900) -> StandardizedLogReg:
    means = []
    scales = []
    minima = []
    maxima = []
    for name in feature_names:
        values = [row.features[name] for row in train_rows]
        m = _mean(values)
        means.append(m)
        scales.append(_std(values, m))
        minima.append(min(values))
        maxima.append(max(values))

    n_train = len(train_rows)
    n_pos = sum(1 for row in train_rows if row.damage_type == 2)
    n_neg = n_train - n_pos
    w_pos = n_train / (2.0 * n_pos)
    w_neg = n_train / (2.0 * n_neg)

    intercept = 0.0
    weights = [0.0 for _ in feature_names]
    lr = 0.05
    l2 = 1e-3
    for _ in range(epochs):
        grad_b = 0.0
        grad_w = [0.0 for _ in feature_names]
        for row in train_rows:
            y = 1.0 if row.damage_type == 2 else 0.0
            x = []
            for idx, name in enumerate(feature_names):
                scale = scales[idx] or 1.0
                x.append((row.features[name] - means[idx]) / scale)
            z = intercept + sum(w * v for w, v in zip(weights, x))
            p = _sigmoid(z)
            sample_weight = w_pos if y == 1.0 else w_neg
            err = (p - y) * sample_weight
            grad_b += err
            for idx, value in enumerate(x):
                grad_w[idx] += err * value

        intercept -= lr * (grad_b / n_train)
        for idx in range(len(weights)):
            weights[idx] -= lr * ((grad_w[idx] / n_train) + l2 * weights[idx])

    return StandardizedLogReg(
        feature_names=feature_names,
        means=means,
        scales=scales,
        minima=minima,
        maxima=maxima,
        weights=weights,
        intercept=intercept,
    )


def fit_hybrid_engine(
    train_rows: list[PreparedRow],
    feature_names: list[str],
    params: HybridParams | None = None,
    epochs: int = 900,
) -> HybridEngine:
    params = params or HybridParams(weight_breakage=0.6, weight_danger_similarity=0.4, ood_penalty=0.3)
    logreg = fit_logreg(train_rows, feature_names, epochs=epochs)
    core_examples = []
    for idx, row in enumerate(train_rows):
        core_examples.append(
            {
                "exampleId": f"core-{idx:05d}",
                "damageType": row.damage_type,
                "features": {name: round(float(row.features[name]), 6) for name in feature_names},
                "vector": [round(value, 8) for value in logreg.standardize_vector(row.features, clip=False)],
            }
        )

    similarity_k = min(5, max(1, len(core_examples) - 1))
    leave_one_out = []
    for idx, example in enumerate(core_examples):
        others = core_examples[:idx] + core_examples[idx + 1 :]
        if not others:
            continue
        ranked = sorted(_euclidean(example["vector"], other["vector"]) for other in others)
        k = min(similarity_k, len(ranked))
        leave_one_out.append(sum(ranked[:k]) / k)

    if leave_one_out:
        ood_p95 = _percentile(leave_one_out, 0.95)
        ood_p99 = _percentile(leave_one_out, 0.99)
        if ood_p99 <= ood_p95:
            ood_p99 = ood_p95 + 1e-6
    else:
        ood_p95 = 0.0
        ood_p99 = 1e-6

    normal_examples = [example for example in core_examples if example["damageType"] == 0]
    normal_leave_one_out = []
    for idx, example in enumerate(normal_examples):
        others = normal_examples[:idx] + normal_examples[idx + 1 :]
        if not others:
            continue
        ranked = sorted(_euclidean(example["vector"], other["vector"]) for other in others)
        k = min(similarity_k, len(ranked))
        normal_leave_one_out.append(sum(ranked[:k]) / k)

    if normal_leave_one_out:
        anomaly_p95 = _percentile(normal_leave_one_out, 0.95)
        anomaly_p99 = _percentile(normal_leave_one_out, 0.99)
        if anomaly_p99 <= anomaly_p95:
            anomaly_p99 = anomaly_p95 + 1e-6
    else:
        anomaly_p95 = 0.0
        anomaly_p99 = 1e-6

    return HybridEngine(
        logreg=logreg,
        core_examples=core_examples,
        similarity_k=similarity_k,
        ood_p95=ood_p95,
        ood_p99=ood_p99,
        anomaly_p95=anomaly_p95,
        anomaly_p99=anomaly_p99,
        weight_breakage=params.weight_breakage,
        weight_danger_similarity=params.weight_danger_similarity,
        ood_penalty=params.ood_penalty,
    )


def metrics_for_threshold(y_true: list[int], scores: list[float], threshold: float) -> dict:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(y_true, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc(y_true: list[int], scores: list[float]) -> float:
    ranked = sorted(zip(scores, y_true), key=lambda item: item[0], reverse=True)
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    tp = 0
    fp = 0
    points = [(0.0, 0.0)]
    prev_score = None
    for score, y in ranked:
        if prev_score is not None and score != prev_score:
            points.append((fp / negatives, tp / positives))
        if y == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    points.append((fp / negatives, tp / positives))

    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        area += (x2 - x1) * (y1 + y2) / 2.0
    return area


def pr_auc(y_true: list[int], scores: list[float]) -> float:
    ranked = sorted(zip(scores, y_true), key=lambda item: item[0], reverse=True)
    positives = sum(y_true)
    if positives == 0:
        return 0.0

    tp = 0
    fp = 0
    points = [(0.0, 1.0)]
    for _, y in ranked:
        if y == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        points.append((recall, precision))

    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        area += (x2 - x1) * (y1 + y2) / 2.0
    return area


def choose_danger_threshold(
    y_true: list[int], scores: list[float], target_recall: float
) -> tuple[float, dict]:
    candidates = sorted({0.0, 1.0, *[round(score, 6) for score in scores]})
    best_pass = None
    best_fallback = None
    for threshold in candidates:
        metric = metrics_for_threshold(y_true, scores, threshold)
        pass_key = (metric["f1"], metric["precision"], -metric["fp"])
        fallback_key = (metric["recall"], metric["f1"], metric["precision"], -metric["fp"])

        if metric["recall"] >= target_recall:
            if best_pass is None or pass_key > best_pass[0]:
                best_pass = (pass_key, threshold, metric)
        if best_fallback is None or fallback_key > best_fallback[0]:
            best_fallback = (fallback_key, threshold, metric)

    chosen = best_pass or best_fallback
    return chosen[1], chosen[2]


def summarize_scores(scores: list[float], warn_threshold: float, danger_threshold: float) -> dict:
    return {
        "count": len(scores),
        "mean": _mean(scores) if scores else 0.0,
        "min": min(scores) if scores else 0.0,
        "p50": _percentile(scores, 0.50) if scores else 0.0,
        "p75": _percentile(scores, 0.75) if scores else 0.0,
        "p95": _percentile(scores, 0.95) if scores else 0.0,
        "max": max(scores) if scores else 0.0,
        "warnOrAboveRatio": (
            sum(1 for score in scores if score >= warn_threshold) / len(scores) if scores else 0.0
        ),
        "dangerOrAboveRatio": (
            sum(1 for score in scores if score >= danger_threshold) / len(scores) if scores else 0.0
        ),
    }


def classify_score(score: float, warn_threshold: float, danger_threshold: float) -> str:
    if score >= danger_threshold:
        return "DANGER"
    if score >= warn_threshold:
        return "CAUTION"
    return "NORMAL"


def apply_static_ood_policy(score: float, ood_score: float, policy: dict) -> str:
    level = classify_score(score, policy["warnThreshold"], policy["dangerThreshold"])
    if ood_score >= policy["oodBlockDangerThreshold"] and score < policy["highConfidenceDanger"]:
        return "CAUTION"
    if ood_score >= policy["oodCautionThreshold"] and level == "NORMAL":
        return "CAUTION"
    return level


def evaluate_static_policy(
    core_records: list[dict],
    shadow_records: list[dict],
    warn_threshold: float,
    danger_threshold: float,
    danger_recall_target: float,
) -> dict:
    candidates = []
    for caution_threshold in (0.50, 0.60, 0.70):
        for block_threshold in (0.75, 0.85, 0.95):
            if block_threshold <= caution_threshold:
                continue
            for high_conf_offset in (0.10, 0.15, 0.20):
                high_conf = min(0.95, max(danger_threshold + high_conf_offset, danger_threshold + 1e-6))
                policy = {
                    "warnThreshold": warn_threshold,
                    "dangerThreshold": danger_threshold,
                    "oodCautionThreshold": caution_threshold,
                    "oodBlockDangerThreshold": block_threshold,
                    "highConfidenceDanger": high_conf,
                }
                core_levels = [apply_static_ood_policy(row["score"], row["oodScore"], policy) for row in core_records]
                shadow_levels = [apply_static_ood_policy(row["score"], row["oodScore"], policy) for row in shadow_records]

                positives = [level for row, level in zip(core_records, core_levels) if row["label"] == 1]
                negatives = [level for row, level in zip(core_records, core_levels) if row["label"] == 0]
                pos_danger_recall = (
                    sum(1 for level in positives if level == "DANGER") / len(positives) if positives else 0.0
                )
                neg_danger_ratio = (
                    sum(1 for level in negatives if level == "DANGER") / len(negatives) if negatives else 0.0
                )
                shadow_danger_ratio = (
                    sum(1 for level in shadow_levels if level == "DANGER") / len(shadow_levels) if shadow_levels else 0.0
                )
                shadow_caution_or_above = (
                    sum(1 for level in shadow_levels if level in {"CAUTION", "DANGER"}) / len(shadow_levels)
                    if shadow_levels else 0.0
                )
                key = (
                    1 if pos_danger_recall >= danger_recall_target else 0,
                    -shadow_danger_ratio,
                    -neg_danger_ratio,
                    shadow_caution_or_above,
                    pos_danger_recall,
                )
                candidates.append(
                    {
                        "sortKey": key,
                        "policy": policy,
                        "metrics": {
                            "positiveDangerRecall": pos_danger_recall,
                            "negativeDangerRatio": neg_danger_ratio,
                            "shadowDangerRatio": shadow_danger_ratio,
                            "shadowCautionOrAboveRatio": shadow_caution_or_above,
                        },
                    }
                )
    best = max(candidates, key=lambda item: item["sortKey"])
    tuned = dict(best["policy"])
    tuned.update(best["metrics"])
    return tuned


def evaluate_feature_candidate(
    core_rows: list[PreparedRow],
    shadow_rows: list[PreparedRow],
    feature_set_name: str,
    feature_names: list[str],
    folds: int,
    seed: int,
    danger_recall_target: float,
    warn_negative_quantile: float,
    params: HybridParams,
    quick_epochs: int = 250,
) -> dict:
    fold_rows = stratified_folds(core_rows, folds, seed)
    core_y: list[int] = []
    core_scores: list[float] = []
    core_records: list[dict] = []
    shadow_records: list[dict] = []
    fold_metrics: list[dict] = []

    for idx in range(folds):
        test_rows = fold_rows[idx]
        train_rows = [row for fold_idx, fold in enumerate(fold_rows) if fold_idx != idx for row in fold]
        engine = fit_hybrid_engine(train_rows, feature_names, params=params, epochs=quick_epochs)

        test_scores = []
        test_y = []
        for row in test_rows:
            pred = engine.predict(row.features, top_k=1)
            label = 1 if row.damage_type == 2 else 0
            core_records.append({"label": label, "score": pred["hybridScore"], "oodScore": pred["oodScore"]})
            core_scores.append(pred["hybridScore"])
            core_y.append(label)
            test_scores.append(pred["hybridScore"])
            test_y.append(label)

        for row in shadow_rows:
            pred = engine.predict(row.features, top_k=1)
            shadow_records.append(
                {
                    "damageType": row.damage_type,
                    "score": pred["hybridScore"],
                    "oodScore": pred["oodScore"],
                }
            )

        fold_metric = metrics_for_threshold(test_y, test_scores, 0.5)
        fold_metric["rocAuc"] = roc_auc(test_y, test_scores)
        fold_metric["prAuc"] = pr_auc(test_y, test_scores)
        fold_metric["fold"] = idx
        fold_metrics.append(fold_metric)

    metrics_05 = metrics_for_threshold(core_y, core_scores, 0.5)
    metrics_05["rocAuc"] = roc_auc(core_y, core_scores)
    metrics_05["prAuc"] = pr_auc(core_y, core_scores)

    negative_scores = [score for score, y in zip(core_scores, core_y) if y == 0]
    warn_threshold = _percentile(negative_scores, warn_negative_quantile)
    danger_threshold, danger_metrics = choose_danger_threshold(core_y, core_scores, target_recall=danger_recall_target)
    if warn_threshold >= danger_threshold:
        warn_threshold = max(0.0, danger_threshold - 0.1)

    shadow_summary = {}
    for label in SHADOW_LABELS:
        label_scores = [row["score"] for row in shadow_records if row["damageType"] == label]
        shadow_summary[str(label)] = summarize_scores(label_scores, warn_threshold, danger_threshold)

    static_policy = evaluate_static_policy(
        core_records=core_records,
        shadow_records=shadow_records,
        warn_threshold=warn_threshold,
        danger_threshold=danger_threshold,
        danger_recall_target=danger_recall_target,
    )
    shadow_danger_ratio = _mean(
        [shadow_summary[str(label)]["dangerOrAboveRatio"] for label in SHADOW_LABELS]
    )
    sort_key = (
        1 if danger_metrics["recall"] >= danger_recall_target else 0,
        -shadow_danger_ratio,
        danger_metrics["f1"],
        metrics_05["prAuc"],
        metrics_05["rocAuc"],
    )

    return {
        "featureSetName": feature_set_name,
        "featureNames": feature_names,
        "hybridParams": {
            "weightBreakage": params.weight_breakage,
            "weightDangerSimilarity": params.weight_danger_similarity,
            "oodPenalty": params.ood_penalty,
        },
        "sortKey": sort_key,
        "coreMetricsAt05": metrics_05,
        "foldMetrics": fold_metrics,
        "warnThreshold": warn_threshold,
        "dangerThreshold": danger_threshold,
        "dangerMetrics": danger_metrics,
        "shadowSummary": shadow_summary,
        "staticPolicy": static_policy,
    }


def tune_feature_and_hybrid_config(
    core_rows: list[PreparedRow],
    shadow_rows: list[PreparedRow],
    folds: int,
    seed: int,
    danger_recall_target: float,
    warn_negative_quantile: float,
) -> tuple[dict, list[dict]]:
    params_grid = [
        HybridParams(weight_breakage=weight, weight_danger_similarity=1.0 - weight, ood_penalty=ood_penalty)
        for weight in (0.50, 0.60, 0.70)
        for ood_penalty in (0.00, 0.30)
    ]
    candidates: list[dict] = []
    for feature_set_name, feature_names in FEATURE_SETS.items():
        for params in params_grid:
            candidates.append(
                evaluate_feature_candidate(
                    core_rows=core_rows,
                    shadow_rows=shadow_rows,
                    feature_set_name=feature_set_name,
                    feature_names=feature_names,
                    folds=folds,
                    seed=seed,
                    danger_recall_target=danger_recall_target,
                    warn_negative_quantile=warn_negative_quantile,
                    params=params,
                )
            )
    best = max(candidates, key=lambda item: item["sortKey"])
    return best, candidates


def apply_temporal_policy_offline(
    score: float,
    base_level: str,
    history_scores: list[float],
    policy: dict,
) -> str:
    window_size = max(int(policy.get("windowSize", 5)), 1)
    scores = [score, *history_scores[: max(window_size - 1, 0)]]
    warn = float(policy["warnThreshold"])
    danger = float(policy["dangerThreshold"])
    warn_release = float(policy["warnRelease"])
    danger_release = float(policy["dangerRelease"])
    high_conf_danger = float(policy["highConfidenceDanger"])
    alpha = float(policy.get("emaAlpha", 0.5))
    caution_hits = int(policy.get("cautionMinWarnHits", 2))
    danger_hits = int(policy.get("dangerMinDangerHits", 3))

    over_warn = sum(1 for value in scores if value >= warn)
    over_danger = sum(1 for value in scores if value >= danger)
    ema_score = _ema(scores, alpha)
    has_history = bool(history_scores)
    danger_confirmed = (
        over_danger >= danger_hits
        or (has_history and over_danger >= 2 and ema_score >= danger)
    )

    if score >= high_conf_danger:
        return "DANGER"
    if score >= danger:
        return "DANGER" if danger_confirmed else "CAUTION"
    if score >= warn:
        return "CAUTION"
    if over_warn >= caution_hits or ema_score >= warn:
        return "CAUTION"
    if base_level == "CAUTION" and score >= warn_release:
        return "CAUTION"
    if base_level == "DANGER" and score >= danger_release:
        return "CAUTION"
    if base_level == "CAUTION":
        return "CAUTION"
    return "NORMAL"


def tune_temporal_policy(policy: dict) -> dict:
    candidates = []
    warn = float(policy["warnThreshold"])
    danger = float(policy["dangerThreshold"])
    for caution_hits in (2, 3):
        for danger_hits in (2, 3):
            for alpha in (0.4, 0.5, 0.6):
                for warn_release in (max(0.0, warn - 0.10), max(0.0, warn - 0.05)):
                    for danger_release in (max(warn, danger - 0.15), max(warn, danger - 0.10)):
                        candidate = {
                            **policy,
                            "cautionMinWarnHits": caution_hits,
                            "dangerMinDangerHits": danger_hits,
                            "emaAlpha": alpha,
                            "warnRelease": warn_release,
                            "dangerRelease": danger_release,
                        }
                        scenarios = [
                            ("stable_normal", [warn * 0.20, warn * 0.28, warn * 0.18, warn * 0.22], "NORMAL"),
                            ("warn_spikes", [warn * 0.20, warn + 0.01, warn * 0.25, warn + 0.02], "CAUTION"),
                            ("single_danger_spike", [danger + 0.03], "CAUTION"),
                            ("danger_escalation", [danger + 0.03, danger + 0.04, danger + 0.05], "DANGER"),
                            ("recover_to_normal", [warn + 0.02, warn * 0.10, warn * 0.08, warn * 0.05], "NORMAL"),
                        ]
                        penalty = 0.0
                        for _, scores, expected_last in scenarios:
                            history: list[float] = []
                            last_level = "NORMAL"
                            for score in scores:
                                base_level = classify_score(score, warn, danger)
                                last_level = apply_temporal_policy_offline(score, base_level, history, candidate)
                                history = [score, *history]
                            if last_level != expected_last:
                                penalty += 10.0
                                if expected_last == "CAUTION" and last_level == "DANGER":
                                    penalty += 5.0
                        candidates.append((penalty, candidate))
    best = min(candidates, key=lambda item: item[0])[1]
    return {
        "windowSize": 5,
        "cautionMinWarnHits": int(best["cautionMinWarnHits"]),
        "dangerMinDangerHits": int(best["dangerMinDangerHits"]),
        "emaAlpha": float(best["emaAlpha"]),
        "warnRelease": float(best["warnRelease"]),
        "dangerRelease": float(best["dangerRelease"]),
    }


def estimate_demo_rms(feature_map: dict, core_rows: list[PreparedRow]) -> float:
    amp_values = [row.features["peakAmplitude"] for row in core_rows]
    density_values = [row.features["energy"] / max(row.features["durationMs"], 1e-6) for row in core_rows]
    amp_norm = (feature_map["peakAmplitude"] - min(amp_values)) / max(max(amp_values) - min(amp_values), 1e-6)
    density_value = feature_map["energy"] / max(feature_map["durationMs"], 1e-6)
    density_norm = (density_value - min(density_values)) / max(max(density_values) - min(density_values), 1e-6)
    rms = 0.04 + 0.40 * amp_norm + 0.45 * density_norm
    return round(min(0.98, max(0.02, rms)), 3)


def build_demo_profiles(
    core_rows: list[PreparedRow],
    engine: HybridEngine,
    policy: dict,
) -> dict:
    scored = []
    for row in core_rows:
        predicted = engine.predict(row.features, top_k=1)
        base_level = apply_static_ood_policy(predicted["hybridScore"], predicted["oodScore"], policy)
        final_level = apply_temporal_policy_offline(predicted["hybridScore"], base_level, [], {**policy, **policy["trend"]})
        scored.append((row, predicted, base_level, final_level))

    normal_candidates = [item for item in scored if item[0].damage_type == 0 and item[3] == "NORMAL"]
    normal_row = (
        min(normal_candidates, key=lambda item: item[1]["hybridScore"])[0]
        if normal_candidates
        else min((item for item in scored if item[0].damage_type == 0), key=lambda item: item[1]["hybridScore"])[0]
    )
    danger_candidates = [item for item in scored if item[0].damage_type == 2 and item[3] == "DANGER"]
    danger_row = (
        max(danger_candidates, key=lambda item: item[1]["hybridScore"])[0]
        if danger_candidates
        else max((item for item in scored if item[0].damage_type == 2), key=lambda item: item[1]["hybridScore"])[0]
    )
    target = (policy["warnThreshold"] + policy["dangerThreshold"]) / 2.0
    caution_candidates = [item for item in scored if item[3] == "CAUTION"]
    caution_row = (
        min(caution_candidates, key=lambda item: abs(item[1]["hybridScore"] - target))[0]
        if caution_candidates
        else min(scored, key=lambda item: abs(item[1]["hybridScore"] - target))[0]
    )

    def _profile(name: str, row: PreparedRow, tension_kn: float) -> dict:
        raw = {key: row.features[key] for key in RUNTIME_FEATURES}
        predicted = engine.predict(row.features, top_k=1)
        base_level = apply_static_ood_policy(predicted["hybridScore"], predicted["oodScore"], policy)
        final_level = apply_temporal_policy_offline(predicted["hybridScore"], base_level, [], {**policy, **policy["trend"]})
        return {
            "sourceDamageType": row.damage_type,
            "expectedRiskLevel": final_level,
            "expectedRiskScore": round(predicted["hybridScore"], 6),
            "features": {
                "rms": estimate_demo_rms(raw, core_rows),
                "peakAmplitude": round(raw["peakAmplitude"], 6),
                "energy": round(raw["energy"], 6),
                "dominantFrequencyHz": round(raw["dominantFrequencyHz"], 6),
                "riseTimeMs": round(raw["riseTimeMs"], 6),
                "durationMs": round(raw["durationMs"], 6),
            },
            "context": {
                "materialType": "synthetic-rope-proxy",
                "tensionEstimateKn": tension_kn,
                "temperatureC": round(raw["temperatureC"], 6),
                "humidityPct": round(raw["humidityPct"], 6),
            },
        }

    return {
        "normal": _profile("normal", normal_row, 70.0),
        "caution": _profile("caution", caution_row, 140.0),
        "danger": _profile("danger", danger_row, 220.0),
    }


def run_core_cv(
    core_rows: list[PreparedRow],
    feature_names: list[str],
    folds: int,
    seed: int,
) -> tuple[list[int], list[float], list[dict]]:
    fold_rows = stratified_folds(core_rows, folds, seed)
    all_y: list[int] = []
    all_scores: list[float] = []
    fold_metrics: list[dict] = []
    for idx in range(folds):
        test_rows = fold_rows[idx]
        train_rows = [row for fold_idx, fold in enumerate(fold_rows) if fold_idx != idx for row in fold]
        engine = fit_hybrid_engine(train_rows, feature_names)
        test_scores = [engine.predict(row.features, top_k=1)["hybridScore"] for row in test_rows]
        test_y = [1 if row.damage_type == 2 else 0 for row in test_rows]
        all_scores.extend(test_scores)
        all_y.extend(test_y)
        fold_metric = metrics_for_threshold(test_y, test_scores, 0.5)
        fold_metric["rocAuc"] = roc_auc(test_y, test_scores)
        fold_metric["prAuc"] = pr_auc(test_y, test_scores)
        fold_metric["fold"] = idx
        fold_metrics.append(fold_metric)
    return all_y, all_scores, fold_metrics


def run_optional_booster(core_rows: list[PreparedRow], feature_names: list[str], folds: int, seed: int, booster: str) -> dict:
    try:
        if booster == "catboost":
            from catboost import CatBoostClassifier  # type: ignore
        elif booster == "lightgbm":
            from lightgbm import LGBMClassifier  # type: ignore
        elif booster == "xgboost":
            from xgboost import XGBClassifier  # type: ignore
        else:  # pragma: no cover - defensive
            raise ValueError(f"unsupported booster {booster}")
    except Exception as exc:  # pragma: no cover - environment-dependent path
        return {"status": "unavailable", "reason": str(exc)}

    fold_rows = stratified_folds(core_rows, folds, seed)
    all_y: list[int] = []
    all_scores: list[float] = []
    for idx in range(folds):
        test_rows = fold_rows[idx]
        train_rows = [row for fold_idx, fold in enumerate(fold_rows) if fold_idx != idx for row in fold]
        x_train = [[row.features[name] for name in feature_names] for row in train_rows]
        y_train = [1 if row.damage_type == 2 else 0 for row in train_rows]
        x_test = [[row.features[name] for name in feature_names] for row in test_rows]
        y_test = [1 if row.damage_type == 2 else 0 for row in test_rows]
        pos_weight = len(y_train) / max(1, sum(y_train))

        if booster == "catboost":
            model = CatBoostClassifier(
                depth=4,
                learning_rate=0.05,
                iterations=250,
                loss_function="Logloss",
                eval_metric="F1",
                verbose=False,
                class_weights=[1.0, pos_weight],
            )
        elif booster == "lightgbm":
            model = LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                objective="binary",
                scale_pos_weight=pos_weight,
                verbosity=-1,
            )
        else:
            model = XGBClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=4,
                subsample=1.0,
                colsample_bytree=1.0,
                objective="binary:logistic",
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
            )

        model.fit(x_train, y_train)
        scores = [float(probs[1]) for probs in model.predict_proba(x_test)]
        all_scores.extend(scores)
        all_y.extend(y_test)

    return {
        "status": "ok",
        "metricsAt05": {
            **metrics_for_threshold(all_y, all_scores, 0.5),
            "rocAuc": roc_auc(all_y, all_scores),
            "prAuc": pr_auc(all_y, all_scores),
        },
    }


def run_optional_booster_comparisons(core_rows: list[PreparedRow], feature_names: list[str], folds: int, seed: int) -> dict:
    return {
        "catboost": run_optional_booster(core_rows, feature_names, folds, seed, "catboost"),
        "lightgbm": run_optional_booster(core_rows, feature_names, folds, seed, "lightgbm"),
        "xgboost": run_optional_booster(core_rows, feature_names, folds, seed, "xgboost"),
    }


def train_xgboost_candidate_model(
    core_rows: list[PreparedRow],
    feature_names: list[str],
    outdir: str,
    seed: int,
    metrics_at_05: dict | None = None,
) -> dict:
    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent path
        return {"status": "unavailable", "reason": str(exc)}

    x_train = [[row.features[name] for name in feature_names] for row in core_rows]
    y_train = [1 if row.damage_type == 2 else 0 for row in core_rows]
    pos_weight = len(y_train) / max(1, sum(y_train))
    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(x_train, y_train)
    model_path = os.path.join(outdir, "xgboost_candidate_model.json")
    model.get_booster().save_model(model_path)
    return {
        "status": "ok",
        "modelType": "xgboost_binary_classifier",
        "modelPath": os.path.basename(model_path),
        "featureNames": feature_names,
        "metricsAt05": metrics_at_05 or {},
    }


def build_policy_probabilities(score: float, warn_threshold: float, danger_threshold: float) -> dict:
    warn = max(1e-6, warn_threshold)
    danger = max(warn + 1e-6, danger_threshold)
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


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")


def write_markdown_report(path: str, summary: dict) -> None:
    core_counts = summary["coreClassCounts"]
    shadow_counts = summary["shadowClassCounts"]
    core_metrics = summary["coreCv"]["metricsAt05"]
    danger_metrics = summary["dangerThreshold"]["metrics"]
    warn_threshold = summary["warnThreshold"]["value"]
    danger_threshold = summary["dangerThreshold"]["value"]
    shadow = summary["shadowSummary"]
    selected = summary["selectedConfig"]
    boosters = summary["boosterComparisons"]

    lines = [
        "# Proxy Breakage Training Report",
        "",
        f"- Generated at: `{summary['generatedAt']}`",
        f"- Source CSV: `{summary['sourceCsv']}`",
        f"- Model version: `{summary['modelVersion']}`",
        f"- Selected feature set: `{selected['featureSetName']}`",
        f"- Features: `{', '.join(summary['featureNames'])}`",
        f"- Hybrid weights: `breakage={selected['hybridParams']['weightBreakage']:.2f}`, `similarity={selected['hybridParams']['weightDangerSimilarity']:.2f}`, `oodPenalty={selected['hybridParams']['oodPenalty']:.2f}`",
        "",
        "## Core dataset",
        "",
        f"- `Damage_Type=0`: `{core_counts['0']}` rows",
        f"- `Damage_Type=2`: `{core_counts['2']}` rows",
        "",
        "## Shadow dataset",
        "",
        f"- `Damage_Type=1`: `{shadow_counts['1']}` rows",
        f"- `Damage_Type=3`: `{shadow_counts['3']}` rows",
        "",
        "## Core CV metrics @ threshold 0.5",
        "",
        f"- Accuracy: `{core_metrics['accuracy']:.4f}`",
        f"- Precision: `{core_metrics['precision']:.4f}`",
        f"- Recall: `{core_metrics['recall']:.4f}`",
        f"- F1: `{core_metrics['f1']:.4f}`",
        f"- FNR: `{core_metrics['fnr']:.4f}`",
        f"- ROC-AUC: `{core_metrics['rocAuc']:.4f}`",
        f"- PR-AUC: `{core_metrics['prAuc']:.4f}`",
        "",
        "## Threshold policy",
        "",
        f"- `T_warn`: `{warn_threshold:.4f}`",
        f"- `T_danger`: `{danger_threshold:.4f}`",
        f"- `T_warn_release`: `{summary['policy']['warnRelease']:.4f}`",
        f"- `T_danger_release`: `{summary['policy']['dangerRelease']:.4f}`",
        f"- `OOD caution`: `{summary['policy']['oodCautionThreshold']:.2f}`",
        f"- `OOD block danger`: `{summary['policy']['oodBlockDangerThreshold']:.2f}`",
        "- Runtime guardrail: inputs are clipped to the core-training feature range before scoring.",
        "",
        "### Danger threshold metrics",
        "",
        f"- Precision: `{danger_metrics['precision']:.4f}`",
        f"- Recall: `{danger_metrics['recall']:.4f}`",
        f"- F1: `{danger_metrics['f1']:.4f}`",
        f"- FNR: `{danger_metrics['fnr']:.4f}`",
        "",
        "## Shadow score behavior",
        "",
        f"- Matrix Cracking mean score: `{shadow['1']['mean']:.4f}`",
        f"- Matrix Cracking >= warn ratio: `{shadow['1']['warnOrAboveRatio']:.4f}`",
        f"- Matrix Cracking >= danger ratio: `{shadow['1']['dangerOrAboveRatio']:.4f}`",
        f"- Delamination mean score: `{shadow['3']['mean']:.4f}`",
        f"- Delamination >= warn ratio: `{shadow['3']['warnOrAboveRatio']:.4f}`",
        f"- Delamination >= danger ratio: `{shadow['3']['dangerOrAboveRatio']:.4f}`",
        "",
        "## Feature-set search",
        "",
        f"- Candidate count: `{summary['featureSearch']['candidateCount']}`",
        f"- Selected `PR-AUC`: `{selected['coreMetricsAt05']['prAuc']:.4f}`",
        f"- Selected shadow danger mean ratio: `{selected['shadowDangerMeanRatio']:.4f}`",
        "",
        "## Optional booster comparisons",
        "",
    ]
    for booster_name in ("catboost", "lightgbm", "xgboost"):
        booster = boosters[booster_name]
        lines.append(f"- `{booster_name}` status: `{booster['status']}`")
        if booster["status"] == "ok":
            metric = booster["metricsAt05"]
            lines.append(
                f"  precision={metric['precision']:.4f}, recall={metric['recall']:.4f}, f1={metric['f1']:.4f}, roc_auc={metric['rocAuc']:.4f}, pr_auc={metric['prAuc']:.4f}"
            )
        else:
            lines.append(f"  reason={booster['reason']}")

    lines.extend(
        [
            "",
            "## Augmentation roadmap",
            "",
            "- Do not oversample before the core 0-vs-2 baseline is stable.",
            "- First priority is rope real-data capture, not synthetic tabular inflation.",
            "- If augmentation is required later, prefer waveform-level perturbations or conservative feature-space jitter bounded by the real sensor range.",
            "- Keep labels `1` and `3` as shadow sets unless real rope labels support a broader target definition.",
            "",
        ]
    )

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to AE_Damage_Detection_Dataset.csv")
    parser.add_argument(
        "--outdir",
        default=os.path.join("ml", "training-pipeline", "artifacts", "latest"),
        help="Output directory for datasets, metrics, and model artifact",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--danger-recall-target", type=float, default=0.90)
    parser.add_argument("--warn-negative-quantile", type=float, default=0.95)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_rows = load_rows(args.csv)
    core_rows, shadow_rows = split_rows(all_rows)
    selected_config, feature_candidates = tune_feature_and_hybrid_config(
        core_rows=core_rows,
        shadow_rows=shadow_rows,
        folds=args.folds,
        seed=args.seed,
        danger_recall_target=args.danger_recall_target,
        warn_negative_quantile=args.warn_negative_quantile,
    )
    selected_feature_names = list(selected_config["featureNames"])
    selected_params = HybridParams(
        weight_breakage=float(selected_config["hybridParams"]["weightBreakage"]),
        weight_danger_similarity=float(selected_config["hybridParams"]["weightDangerSimilarity"]),
        ood_penalty=float(selected_config["hybridParams"]["oodPenalty"]),
    )
    write_dataset_csv(os.path.join(args.outdir, "prepared_core_dataset.csv"), core_rows, selected_feature_names)
    write_dataset_csv(os.path.join(args.outdir, "prepared_shadow_dataset.csv"), shadow_rows, selected_feature_names)

    warn_threshold = float(selected_config["warnThreshold"])
    danger_threshold = float(selected_config["dangerThreshold"])
    static_policy = selected_config["staticPolicy"]
    temporal_policy = tune_temporal_policy(
        {
            "warnThreshold": warn_threshold,
            "dangerThreshold": danger_threshold,
            "highConfidenceDanger": float(static_policy["highConfidenceDanger"]),
            "warnRelease": max(0.0, warn_threshold - 0.10),
            "dangerRelease": max(warn_threshold, danger_threshold - 0.15),
        }
    )
    warn_release = float(temporal_policy["warnRelease"])
    danger_release = float(temporal_policy["dangerRelease"])

    final_engine = fit_hybrid_engine(core_rows, selected_feature_names, params=selected_params)
    shadow_summary = {}
    for label in SHADOW_LABELS:
        label_rows = [row for row in shadow_rows if row.damage_type == label]
        scores = [final_engine.predict(row.features, top_k=1)["hybridScore"] for row in label_rows]
        shadow_summary[str(label)] = summarize_scores(scores, warn_threshold, danger_threshold)

    booster_comparisons = run_optional_booster_comparisons(
        core_rows,
        feature_names=selected_feature_names,
        folds=args.folds,
        seed=args.seed,
    )
    xgboost_candidate = train_xgboost_candidate_model(
        core_rows=core_rows,
        feature_names=selected_feature_names,
        outdir=args.outdir,
        seed=args.seed,
        metrics_at_05=(booster_comparisons.get("xgboost") or {}).get("metricsAt05"),
    )
    demo_profiles = build_demo_profiles(
        core_rows,
        final_engine,
        {
            "warnThreshold": warn_threshold,
            "dangerThreshold": danger_threshold,
            "highConfidenceDanger": float(static_policy["highConfidenceDanger"]),
            "oodCautionThreshold": float(static_policy["oodCautionThreshold"]),
            "oodBlockDangerThreshold": float(static_policy["oodBlockDangerThreshold"]),
            "trend": {
                "windowSize": int(temporal_policy["windowSize"]),
                "cautionMinWarnHits": int(temporal_policy["cautionMinWarnHits"]),
                "dangerMinDangerHits": int(temporal_policy["dangerMinDangerHits"]),
                "emaAlpha": float(temporal_policy["emaAlpha"]),
            },
            "warnRelease": warn_release,
            "dangerRelease": danger_release,
        },
    )
    shadow_danger_mean_ratio = _mean([shadow_summary[str(label)]["dangerOrAboveRatio"] for label in SHADOW_LABELS])
    artifact = {
        "schemaVersion": SCHEMA_VERSION,
        "modelVersion": MODEL_VERSION,
        "generatedAt": _utcnow(),
        "sourceCsv": os.path.abspath(args.csv),
        "labelPolicy": {
            "coreNegativeDamageType": 0,
            "corePositiveDamageType": 2,
            "shadowDamageTypes": [1, 3],
        },
        "featureSpec": {
            "runtimeFeatureNames": list(RUNTIME_FEATURES),
            "derivedFeatureNames": [name for name in selected_feature_names if name not in RUNTIME_FEATURES],
            "derivedFeatureDescriptions": DERIVED_FEATURES,
            "selectedFeatureSet": selected_config["featureSetName"],
            "modelFeatureNames": selected_feature_names,
            "sourceToRuntime": SOURCE_TO_RUNTIME,
        },
        "runtime": {
            "defaultVariant": "hybrid",
            "availableVariants": [
                "hybrid",
                *(
                    ["xgboost_candidate"]
                    if xgboost_candidate.get("status") == "ok"
                    else []
                ),
            ],
        },
        "policy": {
            "warnThreshold": warn_threshold,
            "dangerThreshold": danger_threshold,
            "warnRelease": warn_release,
            "dangerRelease": danger_release,
            "highConfidenceDanger": float(static_policy["highConfidenceDanger"]),
            "oodCautionThreshold": float(static_policy["oodCautionThreshold"]),
            "oodBlockDangerThreshold": float(static_policy["oodBlockDangerThreshold"]),
            "anomalyCautionThreshold": 0.60,
            "anomalyBlockNormalThreshold": 0.85,
            "trend": {
                "windowSize": int(temporal_policy["windowSize"]),
                "cautionMinWarnHits": int(temporal_policy["cautionMinWarnHits"]),
                "dangerMinDangerHits": int(temporal_policy["dangerMinDangerHits"]),
                "emaAlpha": float(temporal_policy["emaAlpha"]),
            },
            "classProbabilityMethod": "hybrid_policy_mapping",
        },
        "candidateModels": {
            "xgboost": xgboost_candidate,
        },
        "demoProfiles": demo_profiles,
        "model": final_engine.to_dict(),
    }
    write_json(os.path.join(args.outdir, "model_artifact.json"), artifact)

    summary = {
        "schemaVersion": SCHEMA_VERSION,
        "modelVersion": MODEL_VERSION,
        "generatedAt": artifact["generatedAt"],
        "sourceCsv": artifact["sourceCsv"],
        "featureNames": selected_feature_names,
        "coreClassCounts": {"0": sum(1 for row in core_rows if row.damage_type == 0), "2": sum(1 for row in core_rows if row.damage_type == 2)},
        "shadowClassCounts": {"1": sum(1 for row in shadow_rows if row.damage_type == 1), "3": sum(1 for row in shadow_rows if row.damage_type == 3)},
        "coreCv": {
            "folds": args.folds,
            "seed": args.seed,
            "foldMetrics": selected_config["foldMetrics"],
            "metricsAt05": selected_config["coreMetricsAt05"],
        },
        "warnThreshold": {
            "value": warn_threshold,
            "negativeQuantile": args.warn_negative_quantile,
        },
        "dangerThreshold": {
            "value": danger_threshold,
            "recallTarget": args.danger_recall_target,
            "metrics": selected_config["dangerMetrics"],
        },
        "policy": {
            "warnRelease": warn_release,
            "dangerRelease": danger_release,
            "highConfidenceDanger": float(static_policy["highConfidenceDanger"]),
            "oodCautionThreshold": float(static_policy["oodCautionThreshold"]),
            "oodBlockDangerThreshold": float(static_policy["oodBlockDangerThreshold"]),
            "anomalyCautionThreshold": 0.60,
            "anomalyBlockNormalThreshold": 0.85,
            "trend": artifact["policy"]["trend"],
            "exampleProbabilities": {
                "score0.20": build_policy_probabilities(0.20, warn_threshold, danger_threshold),
                "score0.50": build_policy_probabilities(0.50, warn_threshold, danger_threshold),
                "score0.80": build_policy_probabilities(0.80, warn_threshold, danger_threshold),
            },
        },
        "shadowSummary": shadow_summary,
        "selectedConfig": {
            **selected_config,
            "shadowDangerMeanRatio": shadow_danger_mean_ratio,
        },
        "featureSearch": {
            "candidateCount": len(feature_candidates),
            "topCandidates": sorted(
                [
                    {
                        "featureSetName": item["featureSetName"],
                        "hybridParams": item["hybridParams"],
                        "prAuc": item["coreMetricsAt05"]["prAuc"],
                        "rocAuc": item["coreMetricsAt05"]["rocAuc"],
                        "dangerRecall": item["dangerMetrics"]["recall"],
                        "shadowDangerMeanRatio": _mean(
                            [item["shadowSummary"][str(label)]["dangerOrAboveRatio"] for label in SHADOW_LABELS]
                        ),
                        "selectionSortKey": item["sortKey"],
                    }
                    for item in feature_candidates
                ],
                key=lambda item: tuple(item["selectionSortKey"]),
                reverse=True,
            )[:5],
        },
        "boosterComparisons": booster_comparisons,
        "candidateModels": artifact["candidateModels"],
        "demoProfiles": demo_profiles,
        "hybridExample": final_engine.predict(core_rows[0].features),
    }
    write_json(os.path.join(args.outdir, "training_summary.json"), summary)
    write_markdown_report(os.path.join(args.outdir, "training_report.md"), summary)

    print(f"Wrote artifacts to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
