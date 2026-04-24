from __future__ import annotations

import json

import torch

from eml_mnist.diagnostics import collect_eml_diagnostics, flatten_nested_metrics
from eml_mnist.experiment_utils import ExperimentLogger, append_csv, count_parameters
from eml_mnist.reporting import generate_validation_report


def test_flatten_nested_metrics_handles_missing_values() -> None:
    values = {
        "loss": torch.tensor(1.5),
        "nested": {"drive": torch.tensor([1.0, 2.0, 3.0])},
        "name": "run",
    }

    flat = flatten_nested_metrics(values)

    assert flat["loss"] == 1.5
    assert flat["nested.drive_mean"] == 2.0
    assert "nested.drive_std" in flat
    assert flat["name"] == "run"


def test_collect_eml_diagnostics_handles_nested_outputs() -> None:
    outputs = {
        "drive": torch.ones(2, 3),
        "resistance": torch.zeros(2, 3),
        "inner": {
            "energy": torch.randn(2, 3),
            "null_weight": torch.full((2,), 0.25),
            "attractor_states": torch.randn(2, 4, 8),
        },
    }

    diagnostics = collect_eml_diagnostics(outputs)

    assert diagnostics["drive_mean"] == 1.0
    assert diagnostics["resistance_mean"] == 0.0
    assert "energy_mean" in diagnostics
    assert "attractor_diversity" in diagnostics


def test_experiment_logger_writes_files(tmp_path) -> None:
    logger = ExperimentLogger(
        run_id="unit_logger",
        config={
            "mode": "test",
            "task_name": "unit",
            "model_name": "linear",
            "dataset_name": "none",
            "seed": 0,
            "device": "cpu",
        },
        root=tmp_path,
    )
    model = torch.nn.Linear(3, 2)
    info = logger.set_model_info(model)
    logger.log_step({"step": 1, "train_loss": 1.0}, {"drive_mean": 0.1})
    logger.finalize({"best_metric": 0.5, "final_metric": 0.5}, model_info=info)

    assert (logger.run_dir / "config.json").exists()
    assert (logger.run_dir / "history.json").exists()
    assert (logger.run_dir / "metrics.csv").exists()
    assert (logger.run_dir / "diagnostics.csv").exists()
    assert (logger.run_dir / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
    summary = json.loads((logger.run_dir / "summary.json").read_text())
    assert summary["status"] == "COMPLETED"
    assert count_parameters(model)["num_params"] == info["num_params"]


def test_not_run_and_report_generation(tmp_path) -> None:
    ExperimentLogger.not_run(
        run_id="unit_not_run",
        config={
            "mode": "test",
            "task_name": "image_synthetic",
            "model_name": "missing",
            "dataset_name": "none",
            "seed": 0,
            "device": "cpu",
        },
        reason="unit test",
        root=tmp_path / "runs",
    )
    output = generate_validation_report(tmp_path / "runs", tmp_path / "report.md")

    text = output.read_text()
    assert "EML Validation and Ablation Report" in text
    assert "NOT RUN" in text


def test_append_csv_expands_columns(tmp_path) -> None:
    path = tmp_path / "items.csv"
    append_csv(path, {"a": 1})
    append_csv(path, {"b": 2})

    text = path.read_text()
    assert "a" in text
    assert "b" in text
