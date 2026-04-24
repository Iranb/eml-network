from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_combined_field_foundation_smoke_training_produces_finite_loss() -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/train_eml_field_foundation.py"),
        "--steps",
        "2",
        "--batch-size",
        "2",
        "--image-train-size",
        "8",
        "--text-train-size",
        "8",
        "--log-interval",
        "1",
        "--output-dir",
        "/tmp/eml_test_field_foundation_smoke",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    assert "total_loss=" in completed.stdout
    assert "attractor_injection_norm=" in completed.stdout
    assert "active_route_strength_mean=" in completed.stdout
