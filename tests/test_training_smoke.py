from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(script: str, *args: str) -> str:
    command = [sys.executable, str(ROOT / script), *args]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return completed.stdout


def test_image_training_smoke() -> None:
    output = run_script(
        "scripts/train_image_shapes.py",
        "--steps",
        "2",
        "--batch-size",
        "4",
        "--train-size",
        "16",
        "--log-interval",
        "1",
        "--output-dir",
        "/tmp/eml_test_image_smoke",
    )
    assert "loss=" in output
    assert "gate_activation_rate=" in output


def test_text_training_smoke() -> None:
    output = run_script(
        "scripts/train_text_grammar.py",
        "--steps",
        "2",
        "--batch-size",
        "4",
        "--train-size",
        "16",
        "--log-interval",
        "1",
        "--output-dir",
        "/tmp/eml_test_text_smoke",
    )
    assert "loss=" in output
    assert "char_accuracy=" in output


def test_foundation_training_smoke() -> None:
    output = run_script(
        "scripts/train_foundation_core.py",
        "--steps",
        "2",
        "--batch-size",
        "4",
        "--image-train-size",
        "16",
        "--text-train-size",
        "16",
        "--log-interval",
        "1",
        "--output-dir",
        "/tmp/eml_test_foundation_smoke",
    )
    assert "total_loss=" in output
    assert "text_char_accuracy=" in output
