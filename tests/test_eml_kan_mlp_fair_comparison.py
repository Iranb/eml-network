from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_eml_kan_mlp_fair_comparison.py"
SPEC = importlib.util.spec_from_file_location("run_eml_kan_mlp_fair_comparison", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

_build_model = MODULE._build_model
_evaluate = MODULE._evaluate
_make_splits = MODULE._make_splits
_model_forward = MODULE._model_forward


def _args(**overrides: object) -> Namespace:
    values = {
        "mode": "smoke",
        "train_size": 64,
        "val_size": 32,
        "test_size": 32,
        "input_dim": 8,
        "hidden_width": 8,
        "hidden_layers": 1,
        "max_matched_width": 64,
        "grid_size": 9,
        "grid_range": 2.0,
        "dropout": 0.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_eml_kan_mlp_splits_have_expected_shapes() -> None:
    args = _args()

    train, val, test, info = _make_splits("symbolic_regression", args, seed=3)

    assert train.x.shape == (64, 8)
    assert val.y.shape == (32, 1)
    assert test.group.shape == (32,)
    assert train.task_type == "regression"
    assert "target_mean" in info


def test_shift_classification_metrics_include_group_gap() -> None:
    args = _args()
    train, _, test, _ = _make_splits("shift_classification", args, seed=5)
    model, extras = _build_model("mlp_same_width", train, args)

    out = _model_forward(model, train.x[:7], warmup_eta=1.0)
    metrics = _evaluate(model, test, batch_size=16)

    assert out["output"].shape == (7, 2)
    assert "accuracy" in metrics
    assert "group_accuracy_gap" in metrics
    assert extras["approx_flops_per_sample"] > 0


def test_eml_kan_comparison_model_train_step_is_finite() -> None:
    args = _args()
    train, _, _, _ = _make_splits("localized_regression", args, seed=7)
    model, _ = _build_model("eml_kan", train, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    out = _model_forward(model, train.x[:16], warmup_eta=0.5)
    loss = torch.nn.functional.mse_loss(out["output"], train.y[:16])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert out["output"].shape == (16, 1)
    assert torch.isfinite(loss)
    assert "edge_resistance_mean" in out["diagnostics"]
