"""SageMaker Training entry point for the line-eye window XGBoost model."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from port_breakage_pipeline import main as train_model


def main() -> None:
    channel = Path("/opt/ml/input/data/training")
    candidates = sorted(channel.rglob("port_mooring_eye_ae_dataset.csv"))
    if not candidates:
        raise FileNotFoundError(f"training dataset not found under {channel}")

    model_dir = Path("/opt/ml/model")
    model_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        "port_breakage_pipeline.py",
        "--src",
        str(candidates[0]),
        "--out",
        str(model_dir),
    ]
    train_model()

    code_dir = model_dir / "code"
    code_dir.mkdir(exist_ok=True)
    source_dir = Path(__file__).resolve().parent
    shutil.copy2(source_dir / "inference.py", code_dir / "inference.py")
    shutil.copy2(source_dir / "requirements.txt", code_dir / "requirements.txt")


if __name__ == "__main__":
    main()
