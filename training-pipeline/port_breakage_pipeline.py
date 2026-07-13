"""Port-domain AE breakage training pipeline (production artifact).

port_mooring_eye_ae_dataset → per-session window features (port_window_features)
→ XGBoost 3-class (GroupKFold OOF) → temperature calibration → threshold policy
→ artifact readable by ml/endpoint-runtime/local_model_runtime.py.

The artifact drops straight into the backend via ``LOCAL_ML_MODEL_PATH`` with
no application-api change (same inference-request/response contract).

Usage:
    PYTHONPATH=<repo>/.vendor_ml python3 port_breakage_pipeline.py \
        --src ~/Downloads/port_mooring_eye_ae_dataset.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

from port_window_features import (
    MIN_EVENTS,
    WINDOW_FEATURES,
    WINDOW_SIZE,
    WINDOW_SPEC_VERSION,
    SIGNAL_HIGH,
    SIGNAL_LHF,
    SIGNAL_LOW,
    SIGNAL_MED,
)

CLASS_ORDER = ["NORMAL", "CAUTION", "DANGER"]
LABEL_KO2IDX = {"안전": 0, "주의": 1, "위험": 2}
SEVERITY_WEIGHTS = [0.0, 0.5, 1.0]  # NORMAL, CAUTION, DANGER
MODEL_TYPE = "port_window_xgb_v1"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + "000Z"


# ── window feature table (vectorised twin of port_window_features.window_vector) ──
def build_window_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the training matrix: one window row per event (min 3 events)."""
    df = df.sort_values(["Session_ID", "Time_Minutes"]).reset_index(drop=True)
    out = []
    for sid, g in df.groupby("Session_ID", sort=False):
        g = g.reset_index(drop=True)
        oh_low = (g["AE_Signal_Type"] == SIGNAL_LOW).astype(float)
        oh_med = (g["AE_Signal_Type"] == SIGNAL_MED).astype(float)
        oh_high = (g["AE_Signal_Type"] == SIGNAL_HIGH).astype(float)
        oh_lhf = (g["AE_Signal_Type"] == SIGNAL_LHF).astype(float)
        roll = lambda s: s.rolling(WINDOW_SIZE, min_periods=MIN_EVENTS)
        span = g["Time_Minutes"] - g["Time_Minutes"].shift(WINDOW_SIZE - 1)
        n_in_win = roll(pd.Series(1.0, index=g.index)).sum()
        f = pd.DataFrame({
            "amp_mean": roll(g["Amplitude_dB_AE"]).mean(),
            "amp_max": roll(g["Amplitude_dB_AE"]).max(),
            "snr_mean": roll(g["SNR_dB"]).mean(),
            "snr_max": roll(g["SNR_dB"]).max(),
            "hit_sum": roll(g["Hit_Count"]).sum(),
            "hit_max": roll(g["Hit_Count"]).max(),
            "dur_max": roll(g["Duration_ms"]).max(),
            "n_low": roll(oh_low).sum(),
            "n_med": roll(oh_med).sum(),
            "n_high": roll(oh_high).sum(),
            "n_lhf": roll(oh_lhf).sum(),
            "fhigh_max": roll(g["Freq_High_kHz"]).max(),
            # events per minute over the realised window
            "rate": n_in_win / span.where(span > 1e-6, np.nan),
            "noise": g["Ambient_Noise_dB"],
            "rain": g["Rain_mmh"],
            "wind": g["Wind_mps"],
            "crane": g["Crane_Active"].astype(float),
        })
        # rate falls back to event count when the window spans ~0 minutes
        f["rate"] = f["rate"].fillna(n_in_win)
        f["y"] = g["DamageType"].map(LABEL_KO2IDX)
        f["grp"] = sid
        out.append(f)
    table = pd.concat(out).dropna(subset=WINDOW_FEATURES + ["y"]).reset_index(drop=True)
    return table


# ── temperature scaling on OOF margins (embeddable single scalar) ──
def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(oof_margins: np.ndarray, y: np.ndarray) -> float:
    """Grid + local refine for the temperature T>0 minimising multiclass NLL."""
    best_t, best = 1.0, _nll(_softmax_rows(oof_margins), y)
    grid = np.concatenate([np.linspace(0.5, 3.0, 26), np.linspace(3.2, 6.0, 15)])
    for t in grid:
        nll = _nll(_softmax_rows(oof_margins / t), y)
        if nll < best:
            best, best_t = nll, float(t)
    # local refine around best_t
    for t in np.linspace(max(0.3, best_t - 0.2), best_t + 0.2, 21):
        nll = _nll(_softmax_rows(oof_margins / t), y)
        if nll < best:
            best, best_t = nll, float(t)
    return best_t


def _severity(probs: np.ndarray) -> np.ndarray:
    return probs @ np.array(SEVERITY_WEIGHTS)


def _youden_threshold(severity: np.ndarray, positive: np.ndarray) -> float:
    """Severity cut maximising TPR − FPR for a binary (positive-vs-rest) split."""
    order = np.argsort(severity)
    s = severity[order]
    pos = positive[order].astype(float)
    P = pos.sum() or 1.0
    N = (len(pos) - pos.sum()) or 1.0
    # threshold t classifies severity >= t as positive; sweep candidate cuts
    tp = pos[::-1].cumsum()[::-1]            # positives with severity >= s[i]
    fp = (1 - pos)[::-1].cumsum()[::-1]      # negatives with severity >= s[i]
    j = tp / P - fp / N
    return float(s[int(np.argmax(j))])


def pick_thresholds(severity: np.ndarray, y: np.ndarray, danger_recall: float = 0.97):
    """Choose warn/danger cuts on the OOF severity axis.

    ``dangerThreshold``: highest cut still catching ``danger_recall`` of true
    위험. ``warnThreshold``: Youden-optimal 정상 vs (주의·위험) 분리점 — 주의를
    정상으로 놓치지 않도록 최적화.
    """
    dsev = np.sort(severity[y == 2])
    if len(dsev):
        idx = int((1 - danger_recall) * len(dsev))
        danger_t = float(dsev[min(idx, len(dsev) - 1)])
    else:
        danger_t = 0.6
    warn_t = _youden_threshold(severity, (y >= 1))
    warn_t = min(warn_t, danger_t - 0.05)
    warn_t = max(0.05, warn_t)
    danger_t = max(warn_t + 0.05, danger_t)
    return round(warn_t, 4), round(danger_t, 4)


def _params() -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "verbosity": 0,
    }


NUM_ROUNDS = 300


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=os.path.expanduser("~/Downloads/port_mooring_eye_ae_dataset.csv"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "artifacts", "line-eye-latest"))
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    table = build_window_table(df)
    X = table[WINDOW_FEATURES].to_numpy(dtype=float)
    y = table["y"].to_numpy(dtype=int)
    groups = table["grp"].to_numpy()
    print(f"windows={len(table)}  sessions={table['grp'].nunique()}  "
          f"class dist={np.bincount(y).tolist()}")

    # ── OOF for calibration + thresholds + honest metrics ──
    oof_margin = np.zeros((len(y), 3), dtype=float)
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        dtr = xgb.DMatrix(X[tr], label=y[tr], feature_names=WINDOW_FEATURES)
        dva = xgb.DMatrix(X[va], feature_names=WINDOW_FEATURES)
        booster = xgb.train(_params(), dtr, num_boost_round=NUM_ROUNDS)
        oof_margin[va] = booster.predict(dva, output_margin=True)

    temperature = fit_temperature(oof_margin, y)
    oof_probs = _softmax_rows(oof_margin / temperature)
    sev = _severity(oof_probs)
    warn_t, danger_t = pick_thresholds(sev, y)

    # OOF report using the deployed decision rule (severity thresholds)
    pred = np.where(sev >= danger_t, 2, np.where(sev >= warn_t, 1, 0))
    metrics = {
        "oofAccuracy": round(float(accuracy_score(y, pred)), 4),
        "oofMacroF1": round(float(f1_score(y, pred, average="macro")), 4),
        "dangerRecall": round(float(recall_score(y, pred, labels=[2], average="macro")), 4),
        "dangerPrecision": round(
            float(confusion_matrix(y, pred)[2, 2] / max(1, confusion_matrix(y, pred)[:, 2].sum())), 4
        ),
        "confusion": confusion_matrix(y, pred).tolist(),
        "temperature": round(temperature, 4),
    }
    print("OOF:", json.dumps(metrics, ensure_ascii=False))

    # ── final booster on ALL windows ──
    dall = xgb.DMatrix(X, label=y, feature_names=WINDOW_FEATURES)
    final = xgb.train(_params(), dall, num_boost_round=NUM_ROUNDS)

    os.makedirs(args.out, exist_ok=True)
    booster_name = "port_window_xgb.json"
    final.save_model(os.path.join(args.out, booster_name))
    table.to_csv(os.path.join(args.out, "prepared_window_dataset.csv"), index=False)

    model_version = f"ae-port-window-xgb-v1-{datetime.now().strftime('%Y%m%d')}"
    artifact = {
        "schemaVersion": "1.0.0",
        "modelVersion": model_version,
        "createdAt": _utcnow_iso(),
        "featureSpec": {
            "windowSpecVersion": WINDOW_SPEC_VERSION,
            "runtimeFeatureNames": WINDOW_FEATURES,
            "window": {"size": WINDOW_SIZE, "minEvents": MIN_EVENTS},
            "note": "serving side assembles these via repo.list_events(lineId, limit=window.size)",
        },
        "model": {
            "modelType": MODEL_TYPE,
            "xgboost": {
                "modelPath": booster_name,
                "featureNames": WINDOW_FEATURES,
                "numClass": 3,
                "classOrder": CLASS_ORDER,
                "temperature": round(temperature, 6),
            },
            "severityWeights": SEVERITY_WEIGHTS,
        },
        "policy": {
            "warnThreshold": warn_t,
            "dangerThreshold": danger_t,
            "warnRelease": round(max(0.02, warn_t - 0.1), 4),
            "dangerRelease": round(max(warn_t, danger_t - 0.15), 4),
            "highConfidenceDanger": round(min(0.98, danger_t + 0.15), 4),
            "oodCautionThreshold": 0.60,
            "oodBlockDangerThreshold": 0.85,
            "anomalyCautionThreshold": 0.60,
            "anomalyBlockNormalThreshold": 0.85,
            "trend": {
                "windowSize": 5,
                "cautionMinWarnHits": 2,
                "dangerMinDangerHits": 3,
                "emaAlpha": 0.5,
            },
        },
        "runtime": {"defaultVariant": "port_window_xgb"},
        "metrics": metrics,
        "trainedFrom": os.path.basename(args.src),
    }
    with open(os.path.join(args.out, "model_artifact.json"), "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, ensure_ascii=False, indent=2)

    _write_report(args.out, artifact, table, model_version)
    print(f"artifact → {os.path.join(args.out, 'model_artifact.json')}")
    print(f"booster  → {os.path.join(args.out, booster_name)}")


def _write_report(out_dir: str, artifact: dict, table: pd.DataFrame, model_version: str) -> None:
    m = artifact["metrics"]
    lines = [
        f"# Port-domain AE breakage model — {model_version}",
        "",
        f"- 생성: {artifact['createdAt']}  · 데이터: {artifact['trainedFrom']}",
        f"- 윈도: 직전 {artifact['featureSpec']['window']['size']}개 이벤트 "
        f"(최소 {artifact['featureSpec']['window']['minEvents']}개) · 세션 {table['grp'].nunique()}개 · "
        f"학습 윈도 {len(table)}행",
        f"- 모델: XGBoost 3-class(multi:softprob) + 온도 스케일링 T={m['temperature']}",
        "",
        "## OOF 성능 (GroupKFold 5, 세션 단위 · 배포 결정규칙=severity 임계)",
        "",
        f"| 정확도 | macro-F1 | 위험 재현율 | 위험 정밀도 |",
        f"|---|---|---|---|",
        f"| {m['oofAccuracy']} | {m['oofMacroF1']} | {m['dangerRecall']} | {m['dangerPrecision']} |",
        "",
        f"혼동행렬(행=실제 N/C/D, 열=예측): `{m['confusion']}`",
        "",
        "## 임계 정책",
        "",
        f"- warnThreshold(severity) = {artifact['policy']['warnThreshold']}",
        f"- dangerThreshold(severity) = {artifact['policy']['dangerThreshold']}",
        f"- severity = 0.5·P(주의) + 1.0·P(위험)",
        "",
        "## 배포",
        "",
        "`LOCAL_ML_MODEL_PATH`를 이 디렉토리의 `model_artifact.json`으로 지정하면 백엔드",
        "수정 없이 로컬 추론에 연결된다. 서빙 측은 `repo.list_events(lineId, limit=window.size)`로",
        "직전 이벤트를 모아 `featureSpec.runtimeFeatureNames` 순서의 윈도 벡터를 만든다",
        "(로직: `port_window_features.window_vector`).",
    ]
    with open(os.path.join(out_dir, "training_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
