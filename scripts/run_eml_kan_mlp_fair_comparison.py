from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_mnist.eml_edge_network import EMLEdgeFunctionNetwork
from eml_mnist.experiment_utils import ExperimentLogger, count_parameters
from eml_mnist.kan_replacement import LinearSplineKANNetwork
from eml_mnist.training import resolve_device, set_seed


KANBEFAIR_URL = "https://github.com/yu-rp/KANbeFair"
KAN_PAPER_URL = "https://arxiv.org/abs/2404.19756"
EML_PAPER_URL = "https://arxiv.org/html/2603.21852v2"


@dataclass(frozen=True)
class SplitData:
    x: torch.Tensor
    y: torch.Tensor
    group: torch.Tensor
    task_type: str


class MLPNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int,
        hidden_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or hidden_width <= 0 or hidden_layers <= 0:
            raise ValueError("invalid MLP dimensions")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_width))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_width
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_width = int(hidden_width)
        self.hidden_layers = int(hidden_layers)

    def forward(self, x: torch.Tensor, warmup_eta: float | torch.Tensor = 1.0) -> Dict[str, torch.Tensor]:
        del warmup_eta
        return {"output": self.net(x), "diagnostics": {}}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KANbeFair-style EML-KAN vs MLP comparison")
    parser.add_argument("--mode", choices=["smoke", "real"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs-root", default="reports/runs")
    parser.add_argument("--output", default="reports/EML_KAN_MLP_FAIR_COMPARISON_REPORT.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["symbolic_regression", "localized_regression", "shift_classification"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["eml_kan", "mlp_same_width", "mlp_param_matched", "spline_kan_reference"],
        default=["eml_kan", "mlp_same_width", "mlp_param_matched"],
    )
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=2048)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--max-matched-width", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=33)
    parser.add_argument("--grid-range", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    return parser


def _default_steps(args: argparse.Namespace) -> int:
    if args.steps > 0:
        return args.steps
    return 80 if args.mode == "smoke" else 3000


def _require_input_dim(input_dim: int) -> None:
    if input_dim < 8:
        raise ValueError("input_dim must be at least 8 for the comparison tasks")


def _randn(size: Iterable[int], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(tuple(size), generator=generator)


def _make_symbolic_regression(size: int, input_dim: int, seed: int) -> SplitData:
    _require_input_dim(input_dim)
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(size, input_dim, generator=generator) * 2.0 - 1.0
    y = (
        torch.sin(math.pi * (x[:, 0] * x[:, 1] + 0.35 * x[:, 2]))
        + 0.45 * x[:, 3].pow(3)
        - torch.exp(-2.5 * x[:, 4].square())
        + 0.25 * x[:, 5]
        + 0.15 * torch.cos(2.0 * math.pi * x[:, 6])
    )
    group = (x[:, 7] > 0.0).long()
    return SplitData(x=x, y=y.unsqueeze(-1), group=group, task_type="regression")


def _make_localized_regression(size: int, input_dim: int, seed: int) -> SplitData:
    _require_input_dim(input_dim)
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(size, input_dim, generator=generator) * 2.0 - 1.0
    bump_a = torch.exp(-35.0 * ((x[:, 0] - 0.35).square() + 0.35 * (x[:, 1] + 0.2).square()))
    bump_b = 0.8 * torch.exp(-45.0 * ((x[:, 2] + 0.45).square() + (x[:, 3] - 0.15).square()))
    valley = 0.55 * torch.exp(-25.0 * ((x[:, 4] - x[:, 5]).square()))
    y = bump_a + bump_b - valley + 0.2 * torch.sin(math.pi * x[:, 6])
    group = (x[:, 7] > 0.0).long()
    return SplitData(x=x, y=y.unsqueeze(-1), group=group, task_type="regression")


def _make_shift_classification(size: int, input_dim: int, seed: int, split: str) -> SplitData:
    _require_input_dim(input_dim)
    generator = torch.Generator().manual_seed(seed)
    stable = _randn((size, 5), generator)
    group = torch.bernoulli(torch.full((size,), 0.5), generator=generator).long()
    clean_logit = (
        1.35 * torch.sin(stable[:, 0])
        + 0.95 * stable[:, 1]
        - 0.75 * stable[:, 2].square()
        + 0.55 * stable[:, 3] * stable[:, 4]
    )
    y = torch.bernoulli(torch.sigmoid(clean_logit), generator=generator).long()
    y_sign = y.float() * 2.0 - 1.0

    if split == "train":
        spurious_strength = 1.25
        group_noise = 0.20
    elif split == "val":
        spurious_strength = -0.25
        group_noise = 0.35
    else:
        spurious_strength = -0.65
        group_noise = 0.45

    x = torch.zeros(size, input_dim)
    corruption = group.float().unsqueeze(1) * group_noise * _randn((size, 5), generator)
    x[:, :5] = stable + corruption
    x[:, 5] = spurious_strength * y_sign + 0.75 * _randn((size,), generator)
    x[:, 6] = group.float() * 2.0 - 1.0
    x[:, 7] = group_noise * group.float() + 0.05 * _randn((size,), generator)
    if input_dim > 8:
        x[:, 8:] = _randn((size, input_dim - 8), generator) * 0.25
    return SplitData(x=x, y=y, group=group, task_type="classification")


def _make_split(task: str, size: int, input_dim: int, seed: int, split: str) -> SplitData:
    if task == "symbolic_regression":
        return _make_symbolic_regression(size, input_dim, seed)
    if task == "localized_regression":
        return _make_localized_regression(size, input_dim, seed)
    if task == "shift_classification":
        return _make_shift_classification(size, input_dim, seed, split)
    raise ValueError(f"unknown task: {task}")


def _standardize_regression(train: SplitData, val: SplitData, test: SplitData) -> tuple[SplitData, SplitData, SplitData, Dict[str, float]]:
    mean = train.y.mean()
    std = train.y.std(unbiased=False).clamp_min(1.0e-6)

    def transform(split: SplitData) -> SplitData:
        return SplitData(x=split.x, y=(split.y - mean) / std, group=split.group, task_type=split.task_type)

    return transform(train), transform(val), transform(test), {"target_mean": float(mean.item()), "target_std": float(std.item())}


def _make_splits(task: str, args: argparse.Namespace, seed: int) -> tuple[SplitData, SplitData, SplitData, Dict[str, float]]:
    train = _make_split(task, args.train_size, args.input_dim, 100_000 + seed, "train")
    val = _make_split(task, args.val_size, args.input_dim, 200_000 + seed, "val")
    test = _make_split(task, args.test_size, args.input_dim, 300_000 + seed, "test")
    if train.task_type == "regression":
        return _standardize_regression(train, val, test)
    return train, val, test, {}


def _output_dim(task_type: str) -> int:
    return 1 if task_type == "regression" else 2


def _widths(input_dim: int, output_dim: int, hidden_width: int, hidden_layers: int) -> list[int]:
    return [input_dim, *([hidden_width] * hidden_layers), output_dim]


def _approx_mlp_flops(input_dim: int, output_dim: int, hidden_width: int, hidden_layers: int) -> int:
    dims = _widths(input_dim, output_dim, hidden_width, hidden_layers)
    flops = 0
    for index in range(len(dims) - 1):
        flops += 2 * dims[index] * dims[index + 1] + dims[index + 1]
    return int(flops)


def _approx_eml_kan_flops(input_dim: int, output_dim: int, hidden_width: int, hidden_layers: int) -> int:
    dims = _widths(input_dim, output_dim, hidden_width, hidden_layers)
    edge_count = sum(dims[index] * dims[index + 1] for index in range(len(dims) - 1))
    sum_cost = sum(dims[index + 1] * max(0, dims[index] - 1) for index in range(len(dims) - 1))
    return int(22 * edge_count + sum_cost)


def _approx_spline_flops(input_dim: int, output_dim: int, hidden_width: int, hidden_layers: int, grid_size: int) -> int:
    dims = _widths(input_dim, output_dim, hidden_width, hidden_layers)
    edge_count = sum(dims[index] * dims[index + 1] for index in range(len(dims) - 1))
    sum_cost = sum(dims[index + 1] * max(0, dims[index] - 1) for index in range(len(dims) - 1))
    return int((4 + grid_size) * edge_count + sum_cost)


def _build_mlp(input_dim: int, output_dim: int, width: int, args: argparse.Namespace) -> MLPNet:
    return MLPNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_width=width,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
    )


def _matched_mlp_width(input_dim: int, output_dim: int, target_params: int, args: argparse.Namespace) -> int:
    best_width = 1
    best_delta = float("inf")
    for width in range(1, args.max_matched_width + 1):
        model = _build_mlp(input_dim, output_dim, width, args)
        params = count_parameters(model)["trainable_params"]
        delta = abs(params - target_params)
        if delta < best_delta:
            best_width = width
            best_delta = float(delta)
    return int(best_width)


def _build_model(model_name: str, train: SplitData, args: argparse.Namespace) -> tuple[nn.Module, Dict[str, Any]]:
    output_dim = _output_dim(train.task_type)
    if model_name == "eml_kan":
        model = EMLEdgeFunctionNetwork(
            _widths(args.input_dim, output_dim, args.hidden_width, args.hidden_layers),
            dropout=args.dropout,
            final_layer_norm=False,
        )
        extras = {
            "comparison_family": "eml_kan",
            "hidden_width": args.hidden_width,
            "approx_flops_per_sample": _approx_eml_kan_flops(args.input_dim, output_dim, args.hidden_width, args.hidden_layers),
        }
        return model, extras

    if model_name == "mlp_same_width":
        model = _build_mlp(args.input_dim, output_dim, args.hidden_width, args)
        extras = {
            "comparison_family": "mlp_same_width",
            "hidden_width": args.hidden_width,
            "approx_flops_per_sample": _approx_mlp_flops(args.input_dim, output_dim, args.hidden_width, args.hidden_layers),
        }
        return model, extras

    if model_name == "mlp_param_matched":
        eml_probe = EMLEdgeFunctionNetwork(
            _widths(args.input_dim, output_dim, args.hidden_width, args.hidden_layers),
            dropout=args.dropout,
            final_layer_norm=False,
        )
        target_params = count_parameters(eml_probe)["trainable_params"]
        width = _matched_mlp_width(args.input_dim, output_dim, target_params, args)
        model = _build_mlp(args.input_dim, output_dim, width, args)
        extras = {
            "comparison_family": "mlp_param_matched",
            "hidden_width": width,
            "target_eml_params": target_params,
            "approx_flops_per_sample": _approx_mlp_flops(args.input_dim, output_dim, width, args.hidden_layers),
        }
        return model, extras

    if model_name == "spline_kan_reference":
        model = LinearSplineKANNetwork(
            _widths(args.input_dim, output_dim, args.hidden_width, args.hidden_layers),
            grid_size=args.grid_size,
            grid_range=args.grid_range,
            dropout=args.dropout,
            hidden_layer_norm=True,
        )
        extras = {
            "comparison_family": "spline_kan_reference",
            "hidden_width": args.hidden_width,
            "grid_size": args.grid_size,
            "grid_range": args.grid_range,
            "approx_flops_per_sample": _approx_spline_flops(
                args.input_dim,
                output_dim,
                args.hidden_width,
                args.hidden_layers,
                args.grid_size,
            ),
        }
        return model, extras

    raise ValueError(f"unknown model: {model_name}")


def _to_device(split: SplitData, device: torch.device) -> SplitData:
    return SplitData(
        x=split.x.to(device),
        y=split.y.to(device),
        group=split.group.to(device),
        task_type=split.task_type,
    )


def _model_forward(model: nn.Module, x: torch.Tensor, warmup_eta: float) -> Dict[str, Any]:
    if isinstance(model, EMLEdgeFunctionNetwork):
        return model(x, warmup_eta=warmup_eta)
    return model(x)  # type: ignore[no-any-return]


def _loss(output: torch.Tensor, target: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "regression":
        return F.mse_loss(output, target)
    return F.cross_entropy(output, target.long())


def _ece(probs: torch.Tensor, target: torch.Tensor, bins: int = 10) -> float:
    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(target)
    total = target.numel()
    value = probs.new_tensor(0.0)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            bin_conf = confidence[mask].mean()
            bin_acc = correct[mask].float().mean()
            value = value + mask.float().mean() * (bin_conf - bin_acc).abs()
    return float(value.detach().cpu().item()) if total > 0 else float("nan")


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().flatten().float().cpu()
    labels = labels.detach().flatten().long().cpu()
    pos = labels == 1
    neg = labels == 0
    pos_count = int(pos.sum().item())
    neg_count = int(neg.sum().item())
    if pos_count == 0 or neg_count == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    rank_sum = ranks[pos].sum()
    auc = (rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc.item())


def _group_metric_gap(values: torch.Tensor, group: torch.Tensor, higher_is_better: bool) -> tuple[float, float, float]:
    group = group.long()
    metrics = []
    for group_id in (0, 1):
        mask = group == group_id
        if mask.any():
            metrics.append(float(values[mask].float().mean().detach().cpu().item()))
        else:
            metrics.append(float("nan"))
    gap = abs(metrics[0] - metrics[1]) if math.isfinite(metrics[0]) and math.isfinite(metrics[1]) else float("nan")
    worst = min(metrics) if higher_is_better else max(metrics)
    return metrics[0], metrics[1], worst, gap


def _diagnostic_sample_score(out: Mapping[str, Any], key: str) -> torch.Tensor | None:
    value = out.get(key)
    if not torch.is_tensor(value) or value.numel() == 0:
        return None
    if value.ndim <= 1:
        return value.detach().float()
    dims = tuple(range(1, value.ndim))
    return value.detach().float().mean(dim=dims)


def _evaluate(model: nn.Module, split: SplitData, batch_size: int) -> Dict[str, Any]:
    model.eval()
    outputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    groups: list[torch.Tensor] = []
    resistance_scores: list[torch.Tensor] = []
    energy_scores: list[torch.Tensor] = []
    diagnostics: Dict[str, float] = {}

    with torch.no_grad():
        for start in range(0, split.x.size(0), batch_size):
            xb = split.x[start : start + batch_size]
            out = _model_forward(model, xb, warmup_eta=1.0)
            output = out["output"]
            outputs.append(output.detach())
            targets.append(split.y[start : start + batch_size].detach())
            groups.append(split.group[start : start + batch_size].detach())
            if not diagnostics:
                for key, value in dict(out.get("diagnostics", {})).items():
                    if torch.is_tensor(value):
                        diagnostics[f"diag_{key}"] = float(value.detach().float().mean().cpu().item())
            resistance = _diagnostic_sample_score(out, "resistance")
            if resistance is not None:
                resistance_scores.append(resistance)
            energy = _diagnostic_sample_score(out, "energy")
            if energy is not None:
                energy_scores.append(energy.abs())

    output = torch.cat(outputs, dim=0)
    target = torch.cat(targets, dim=0)
    group = torch.cat(groups, dim=0)
    if split.task_type == "regression":
        error = (output - target).squeeze(-1)
        mse = error.square()
        mae = error.abs()
        group0_mse, group1_mse, worst_group_mse, group_mse_gap = _group_metric_gap(mse, group, higher_is_better=False)
        metrics: Dict[str, Any] = {
            "mse": float(mse.mean().detach().cpu().item()),
            "rmse": float(torch.sqrt(mse.mean()).detach().cpu().item()),
            "mae": float(mae.mean().detach().cpu().item()),
            "group0_mse": group0_mse,
            "group1_mse": group1_mse,
            "worst_group_mse": worst_group_mse,
            "group_mse_gap": group_mse_gap,
        }
        if resistance_scores:
            resistance = torch.cat(resistance_scores, dim=0)
            threshold = mae.median()
            metrics["resistance_high_error_auc"] = _binary_auc(resistance, (mae >= threshold).long())
        if energy_scores:
            energy = torch.cat(energy_scores, dim=0)
            threshold = mae.median()
            metrics["energy_high_error_auc"] = _binary_auc(energy, (mae >= threshold).long())
        metrics.update(diagnostics)
        return metrics

    target_long = target.long()
    probs = torch.softmax(output, dim=1)
    pred = probs.argmax(dim=1)
    correct = pred.eq(target_long)
    nll = F.cross_entropy(output, target_long, reduction="none")
    one_hot = F.one_hot(target_long, num_classes=output.size(1)).float()
    brier = (probs - one_hot).square().sum(dim=1)
    conf = probs.max(dim=1).values
    group0_acc, group1_acc, worst_group_acc, group_acc_gap = _group_metric_gap(correct.float(), group, higher_is_better=True)
    group0_nll, group1_nll, worst_group_nll, group_nll_gap = _group_metric_gap(nll, group, higher_is_better=False)
    pos_rate = pred.float()
    group0_pos, group1_pos, _, pos_gap = _group_metric_gap(pos_rate, group, higher_is_better=True)
    metrics = {
        "accuracy": float(correct.float().mean().detach().cpu().item()),
        "nll": float(nll.mean().detach().cpu().item()),
        "brier": float(brier.mean().detach().cpu().item()),
        "ece": _ece(probs, target_long),
        "group0_accuracy": group0_acc,
        "group1_accuracy": group1_acc,
        "worst_group_accuracy": worst_group_acc,
        "group_accuracy_gap": group_acc_gap,
        "group0_nll": group0_nll,
        "group1_nll": group1_nll,
        "worst_group_nll": worst_group_nll,
        "group_nll_gap": group_nll_gap,
        "group0_positive_rate": group0_pos,
        "group1_positive_rate": group1_pos,
        "positive_rate_gap": pos_gap,
        "uncertainty_error_auc": _binary_auc(1.0 - conf, (~correct).long()),
    }
    if resistance_scores:
        resistance = torch.cat(resistance_scores, dim=0)
        metrics["resistance_error_auc"] = _binary_auc(resistance, (~correct).long())
        metrics["resistance_mean"] = float(resistance.mean().detach().cpu().item())
    if energy_scores:
        energy = torch.cat(energy_scores, dim=0)
        metrics["energy_error_auc"] = _binary_auc(energy, (~correct).long())
    metrics.update(diagnostics)
    return metrics


def _score_from_metrics(metrics: Mapping[str, Any], task_type: str) -> float:
    if task_type == "regression":
        return -float(metrics["mse"])
    return float(metrics["accuracy"])


def _metric_prefix(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state.items()})


def _run_one(model_name: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    set_seed(seed)
    steps = _default_steps(args)
    train_cpu, val_cpu, test_cpu, target_info = _make_splits(task, args, seed)
    train = _to_device(train_cpu, device)
    val = _to_device(val_cpu, device)
    test = _to_device(test_cpu, device)
    model, model_extras = _build_model(model_name, train, args)
    model = model.to(device)

    run_id = f"eml_kan_mlp_{task}_{model_name}_seed{seed}"
    config = {
        "mode": args.mode,
        "task_name": "eml_kan_mlp_fair_comparison",
        "model_name": model_name,
        "dataset_name": task,
        "seed": seed,
        "device": str(device),
        "task_type": train.task_type,
        "steps": steps,
        "eval_interval": args.eval_interval,
        "batch_size": args.batch_size,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "input_dim": args.input_dim,
        "hidden_width": args.hidden_width,
        "hidden_layers": args.hidden_layers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "comparison_source": KANBEFAIR_URL,
    }
    logger = ExperimentLogger(run_id=run_id, config=config, root=args.runs_root)
    model_info = logger.set_model_info(model, extra={**target_info, **model_extras})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -float("inf")
    best_step = 0
    best_state = _clone_state_dict(model)
    stale_evals = 0
    completed_steps = 0
    early_stop_triggered = False
    final_train_loss = float("nan")
    final_val_metrics: Dict[str, Any] = {}
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        indices = torch.randint(0, train.x.size(0), (args.batch_size,), device=device)
        xb = train.x.index_select(0, indices)
        yb = train.y.index_select(0, indices)
        warmup_eta = min(1.0, step / max(1, args.warmup_steps))
        out = _model_forward(model, xb, warmup_eta=warmup_eta)
        loss = _loss(out["output"], yb, train.task_type)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        final_train_loss = float(loss.detach().cpu().item())
        completed_steps = step

        should_eval = step == 1 or step % args.eval_interval == 0 or step == steps
        if should_eval:
            final_val_metrics = _evaluate(model, val, args.batch_size)
            score = _score_from_metrics(final_val_metrics, train.task_type)
            improved = score > best_score + args.min_delta
            if improved:
                best_score = score
                best_step = step
                best_state = _clone_state_dict(model)
                stale_evals = 0
            else:
                stale_evals += 1
            logger.log_step(
                {
                    "step": step,
                    "train_loss": final_train_loss,
                    "val_score": score,
                    "best_val_score": best_score,
                    "warmup_eta": warmup_eta,
                },
                diagnostics=final_val_metrics,
            )
            if stale_evals >= args.patience:
                early_stop_triggered = True
                break

    _load_state_dict(model, best_state, device)
    best_val_metrics = _evaluate(model, val, args.batch_size)
    test_metrics = _evaluate(model, test, args.batch_size)
    total_time = time.perf_counter() - start_time
    best_val_score = _score_from_metrics(best_val_metrics, train.task_type)
    final_val_score = _score_from_metrics(final_val_metrics or best_val_metrics, train.task_type)
    test_score = _score_from_metrics(test_metrics, train.task_type)
    hit_step_cap = not early_stop_triggered and completed_steps >= steps
    summary = {
        "task_type": train.task_type,
        "hidden_width_effective": model_extras.get("hidden_width"),
        "approx_flops_per_sample": model_extras.get("approx_flops_per_sample"),
        "best_metric": best_val_score,
        "final_metric": test_score,
        "best_val_score": best_val_score,
        "final_val_score": final_val_score,
        "test_score": test_score,
        "final_train_loss": final_train_loss,
        "best_step": best_step,
        "completed_steps": completed_steps,
        "early_stop_triggered": early_stop_triggered,
        "hit_step_cap": hit_step_cap,
        "comparison_complete": early_stop_triggered,
        "total_train_time_sec": total_time,
        **_metric_prefix("best_val", best_val_metrics),
        **_metric_prefix("test", test_metrics),
    }
    logger.add_artifact("task_protocol", _task_protocol(task))
    return logger.finalize(summary=summary, model_info=model_info)


def _task_protocol(task: str) -> str:
    protocols = {
        "symbolic_regression": "KAN-friendly smooth symbolic formula regression; lower MSE is better.",
        "localized_regression": "Local bump/valley regression with sharp regions; lower MSE is better.",
        "shift_classification": "Binary classification with train/test spurious-feature reversal and group-dependent corruption; higher accuracy and lower group gap are better.",
    }
    return protocols.get(task, "unknown")


def _rows(root: Path) -> list[Dict[str, str]]:
    summary = root / "summary.csv"
    if not summary.exists():
        return []
    with summary.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metrics(row: Mapping[str, str]) -> Dict[str, Any]:
    try:
        return json.loads(row.get("metrics_json") or "{}")
    except Exception:
        return {}


def _fmt(value: Any, digits: int = 5) -> str:
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return "nan"
        return f"{numeric:.{digits}f}"
    except Exception:
        return "MISSING"


def _latest_rows(rows: list[Dict[str, str]]) -> list[Dict[str, str]]:
    latest: dict[str, Dict[str, str]] = {}
    for row in rows:
        if row.get("run_id", "").startswith("eml_kan_mlp_"):
            latest[row.get("run_id", "")] = row
    return list(latest.values())


def _group_rows(rows: list[Dict[str, str]]) -> dict[tuple[str, str, str], list[Dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "COMPLETED":
            metrics = _metrics(row)
            task_type = str(metrics.get("task_type") or row.get("task_type") or "")
            if not task_type:
                task_type = "classification" if row.get("dataset_name") == "shift_classification" else "regression"
            grouped[(row.get("dataset_name", ""), task_type, row.get("model_name", ""))].append(row)
    return grouped


def _mean(items: list[Dict[str, str]], key: str, source: str = "metrics") -> float:
    values = []
    for item in items:
        try:
            raw = _metrics(item).get(key) if source == "metrics" else item.get(key)
            values.append(float(raw))
        except Exception:
            pass
    return sum(values) / len(values) if values else float("nan")


def _aggregate_regression(rows: list[Dict[str, str]]) -> list[str]:
    lines = [
        "| task | model | n | mean test MSE | mean test RMSE | mean worst-group MSE | mean group MSE gap | mean params | mean approx FLOPs | capped rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (task, task_type, model_name), items in sorted(_group_rows(rows).items()):
        if task_type != "regression":
            continue
        capped = sum(1 for item in items if str(_metrics(item).get("hit_step_cap", "")).lower() == "true")
        lines.append(
            "| "
            + " | ".join(
                [
                    task,
                    model_name,
                    str(len(items)),
                    _fmt(_mean(items, "test_mse")),
                    _fmt(_mean(items, "test_rmse")),
                    _fmt(_mean(items, "test_worst_group_mse")),
                    _fmt(_mean(items, "test_group_mse_gap")),
                    _fmt(_mean(items, "num_params", source="row"), digits=1),
                    _fmt(_mean(items, "approx_flops_per_sample"), digits=1),
                    str(capped),
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| MISSING | MISSING | 0 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
    return lines


def _aggregate_classification(rows: list[Dict[str, str]]) -> list[str]:
    lines = [
        "| task | model | n | mean test acc | mean NLL | mean ECE | mean worst-group acc | mean acc gap | mean resistance-error AUC | mean params | mean approx FLOPs | capped rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (task, task_type, model_name), items in sorted(_group_rows(rows).items()):
        if task_type != "classification":
            continue
        capped = sum(1 for item in items if str(_metrics(item).get("hit_step_cap", "")).lower() == "true")
        lines.append(
            "| "
            + " | ".join(
                [
                    task,
                    model_name,
                    str(len(items)),
                    _fmt(_mean(items, "test_accuracy")),
                    _fmt(_mean(items, "test_nll")),
                    _fmt(_mean(items, "test_ece")),
                    _fmt(_mean(items, "test_worst_group_accuracy")),
                    _fmt(_mean(items, "test_group_accuracy_gap")),
                    _fmt(_mean(items, "test_resistance_error_auc")),
                    _fmt(_mean(items, "num_params", source="row"), digits=1),
                    _fmt(_mean(items, "approx_flops_per_sample"), digits=1),
                    str(capped),
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| MISSING | MISSING | 0 | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
    return lines


def _run_table(rows: list[Dict[str, str]]) -> list[str]:
    lines = [
        "| run_id | task | model | seed | status | test score | best val score | steps | early stop | capped | params | FLOPs |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: item.get("run_id", "")):
        metrics = _metrics(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("run_id", ""),
                    row.get("dataset_name", ""),
                    row.get("model_name", ""),
                    row.get("seed", ""),
                    row.get("status", ""),
                    _fmt(metrics.get("test_score")),
                    _fmt(metrics.get("best_val_score")),
                    str(metrics.get("completed_steps", "")),
                    str(metrics.get("early_stop_triggered", "")),
                    str(metrics.get("hit_step_cap", "")),
                    row.get("num_params", ""),
                    _fmt(metrics.get("approx_flops_per_sample", ""), digits=1),
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
    return lines


def generate_report(runs_root: str | Path, output: str | Path) -> Path:
    rows = _latest_rows(_rows(Path(runs_root)))
    capped = [row for row in rows if str(_metrics(row).get("hit_step_cap", "")).lower() == "true"]
    complete = [row for row in rows if str(_metrics(row).get("comparison_complete", "")).lower() == "true"]
    lines = [
        "# EML-KAN vs MLP Fair Comparison Report",
        "",
        "## Scope",
        f"- KANbeFair reference code: {KANBEFAIR_URL}",
        f"- KAN paper reference: {KAN_PAPER_URL}",
        f"- EML paper reference: {EML_PAPER_URL}",
        "- This benchmark compares the current KAN-style sEML edge network against MLP baselines on identical generated data and seeds.",
        "- `mlp_same_width` matches hidden width/depth; `mlp_param_matched` widens MLP until its parameter count is closest to EML-KAN.",
        "- `spline_kan_reference` is optional and uses the local degree-1 spline edge implementation, not external pykan.",
        "- Rows that hit the configured step cap are marked capped and should not be used as final model-comparison evidence.",
        f"- Completeness: {len(complete)}/{len(rows)} rows early-stopped; {len(capped)} rows hit the step cap.",
        "",
        "## Protocol Notes",
        "- KANbeFair lesson used here: report parameters/FLOPs, use same data/seeds, include a capacity-matched MLP, and keep architecture-specific operators isolated.",
        "- EML-KAN operator: KAN-style edge matrix with `silu` residual plus stable sEML drive/resistance energy.",
        "- MLP baseline: Linear/GELU stack with the same number of hidden layers.",
        "- Validation score drives early stopping: negative MSE for regression and accuracy for classification.",
        "",
        "## Regression Aggregates",
        *_aggregate_regression(rows),
        "",
        "## Classification Aggregates",
        *_aggregate_classification(rows),
        "",
        "## Runs",
        *_run_table(rows),
        "",
        "## Raw Artifacts",
    ]
    for row in sorted(rows, key=lambda item: item.get("run_id", "")):
        lines.append(f"- `{row.get('run_id', '')}`: `{row.get('run_dir', '')}`")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    Path(args.runs_root).mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        for seed in args.seeds:
            for model_name in args.models:
                _run_one(model_name, task, seed, args, device)
    report = generate_report(args.runs_root, args.output)
    print(json.dumps({"status": "complete", "report": str(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
