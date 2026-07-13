"""SageMaker Processing entry point for the line-eye piezo dataset."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

INPUT = Path("/opt/ml/processing/input/data/port_mooring_eye_ae_dataset.csv")
OUTPUT = Path("/opt/ml/processing/output")
REQUIRED = {
    "Session_ID",
    "Time_Minutes",
    "Amplitude_dB_AE",
    "Freq_Low_kHz",
    "Freq_High_kHz",
    "Duration_ms",
    "Hit_Count",
    "SNR_dB",
    "AE_Signal_Type",
    "Ambient_Noise_dB",
    "Rain_mmh",
    "Wind_mps",
    "Crane_Active",
    "DamageType",
}


def main() -> None:
    frame = pd.read_csv(INPUT)
    missing = sorted(REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"missing dataset columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("dataset is empty")

    frame = frame.drop_duplicates().sort_values(
        ["Session_ID", "Time_Minutes"], kind="stable"
    )
    invalid_labels = sorted(set(frame["DamageType"].dropna()) - {"안전", "주의", "위험"})
    if invalid_labels:
        raise ValueError(f"unsupported DamageType values: {invalid_labels}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepared = OUTPUT / "port_mooring_eye_ae_dataset.csv"
    frame.to_csv(prepared, index=False)
    summary = {
        "processedAt": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "sessions": int(frame["Session_ID"].nunique()),
        "classes": {str(k): int(v) for k, v in frame["DamageType"].value_counts().items()},
        "sensorPlacement": "mooring-line ship-side eye/splice",
        "input": os.fspath(INPUT),
        "output": prepared.name,
    }
    (OUTPUT / "processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
