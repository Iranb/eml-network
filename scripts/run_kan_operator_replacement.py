from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_mnist.eml_edge_network import EMLEdgeFunctionNetwork
from eml_mnist.experiment_utils import ExperimentLogger
from eml_mnist.kan_replacement import LinearSplineKANNetwork
from eml_mnist.training import resolve_device, set_seed


KAN_PAPER_URL = "https://arxiv.org/abs/2404.19756"
EML_PAPER_URL = "https://arxiv.org/html/2603.21852v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare KAN spline edge operators with sEML replacement edge operators")
    parser.add_argument("--mode", choices=["smoke", "ablation"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs-root", default="reports/runs")
    parser.add_argument("--output", default="reports/KAN_OPERATOR_REPLACEMENT_REPORT.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--tasks", nargs="+", default=["additive_smooth", "local_bumps", "mixed_composition"])
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--input-dim", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=17)
    parser.add_argument("--grid-range", type=float, default=2.5)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    return parser


def _default_steps(args: argparse.Namespace) -> int:
    if args.steps > 0:
        return args.steps
    return 100 if args.mode == "smoke" else 500


def _widths(args: argparse.Namespace) -> list[int]:
    return [args.input_dim, *([args.hidden_width] * args.hidden_layers), 1]


def _make_targets(x: torch.Tensor, task: str) -> torch.Tensor:
    if task == "additive_smooth":
        y = torch.sin(math.pi * x[:, 0]) + 0.5 * x[:, 1].square() - torch.exp(-2.0 * x[:, 2].square()) + 0.25 * x[:, 3]
    elif task == "local_bumps":
        y = (
            torch.exp(-30.0 * (x[:, 0] - 0.35).square())
            + 0.7 * torch.exp(-40.0 * (x[:, 1] + 0.25).square())
            - 0.5 * torch.exp(-20.0 * (x[:, 2].square() + (x[:, 3] - 0.4).square()))
        )
    elif task == "mixed_composition":
        y = torch.sin(math.pi * (x[:, 0] * x[:, 1] + 0.5 * x[:, 2])) + 0.35 * x[:, 3].pow(3)
    else:
        raise ValueError(f"unknown task: {task}")
    return y.unsqueeze(-1)


def _make_dataset(task: str, size: int, input_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(size, input_dim, generator=generator) * 2.0 - 1.0
    y = _make_targets(x, task)
    return x, y


def _standardize_targets(train_y: torch.Tensor, val_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    mean = train_y.mean()
    std = train_y.std(unbiased=False).clamp_min(1.0e-6)
    return (train_y - mean) / std, (val_y - mean) / std, float(mean.item()), float(std.item())


def _build_model(model_name: str, args: argparse.Namespace) -> nn.Module:
    widths = _widths(args)
    if model_name == "spline_kan":
        return LinearSplineKANNetwork(
            widths,
            grid_size=args.grid_size,
            grid_range=args.grid_range,
            hidden_layer_norm=True,
        )
    if model_name == "semL_operator_replacement":
        return EMLEdgeFunctionNetwork(widths, final_layer_norm=False)
    raise ValueError(f"unknown model: {model_name}")


def _model_output(model: nn.Module, x: torch.Tensor, warmup_eta: float) -> Dict[str, Any]:
    if isinstance(model, EMLEdgeFunctionNetwork):
        return model(x, warmup_eta=warmup_eta)
    return model(x)  # type: ignore[no-any-return]


def _diagnostics(out: Dict[str, Any]) -> Dict[str, Any]:
    values = dict(out.get("diagnostics", {}))
    for key in ("drive", "resistance", "energy", "edge_output"):
        value = out.get(key)
        if torch.is_tensor(value) and value.numel() > 0:
            detached = value.detach().float()
            values[f"{key}_mean"] = detached.mean()
            values[f"{key}_std"] = detached.std(unbiased=False)
    return values


def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> tuple[float, Dict[str, Any]]:
    model.eval()
    losses: list[torch.Tensor] = []
    diagnostics: Dict[str, Any] = {}
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            out = _model_output(model, xb, warmup_eta=1.0)
            loss = F.mse_loss(out["output"], yb)
            losses.append(loss.detach() * xb.size(0))
            if not diagnostics:
                diagnostics = _diagnostics(out)
    total = torch.stack(losses).sum() / x.size(0)
    return float(total.detach().cpu().item()), diagnostics


def _run_one(model_name: str, task: str, seed: int, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    set_seed(seed)
    steps = _default_steps(args)
    run_id = f"kan_operator_{task}_{model_name}_seed{seed}"
    config = {
        "mode": args.mode,
        "task_name": "kan_operator_replacement",
        "model_name": model_name,
        "dataset_name": task,
        "seed": seed,
        "device": str(device),
        "steps": steps,
        "batch_size": args.batch_size,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "widths": _widths(args),
        "grid_size": args.grid_size,
        "grid_range": args.grid_range,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    logger = ExperimentLogger(run_id=run_id, config=config, root=args.runs_root)

    train_x, train_y = _make_dataset(task, args.train_size, args.input_dim, seed=10_000 + seed)
    val_x, val_y = _make_dataset(task, args.val_size, args.input_dim, seed=20_000 + seed)
    train_y, val_y, target_mean, target_std = _standardize_targets(train_y, val_y)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    model = _build_model(model_name, args).to(device)
    model_info = logger.set_model_info(model, extra={"target_mean": target_mean, "target_std": target_std})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_mse = float("inf")
    best_step = 0
    final_train_mse = float("inf")
    final_val_mse = float("inf")
    stale_evals = 0
    completed_steps = 0
    early_stop_triggered = False
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        indices = torch.randint(0, train_x.size(0), (args.batch_size,), device=device)
        xb = train_x.index_select(0, indices)
        yb = train_y.index_select(0, indices)
        warmup_eta = min(1.0, step / max(1, args.warmup_steps))
        out = _model_output(model, xb, warmup_eta=warmup_eta)
        loss = F.mse_loss(out["output"], yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_train_mse = float(loss.detach().cpu().item())
        completed_steps = step

        should_eval = step == 1 or step % args.eval_interval == 0 or step == steps
        if should_eval:
            final_val_mse, diagnostics = _evaluate(model, val_x, val_y, args.batch_size)
            improved = final_val_mse < best_val_mse - args.min_delta
            if improved:
                best_val_mse = final_val_mse
                best_step = step
                stale_evals = 0
            else:
                stale_evals += 1
            logger.log_step(
                {
                    "step": step,
                    "train_mse": final_train_mse,
                    "val_mse": final_val_mse,
                    "best_val_mse": best_val_mse,
                    "warmup_eta": warmup_eta,
                },
                diagnostics=diagnostics,
            )
            if stale_evals >= args.patience:
                early_stop_triggered = True
                break

    total_time = time.perf_counter() - start_time
    summary = {
        "best_metric": -best_val_mse,
        "final_metric": -final_val_mse,
        "best_val_mse": best_val_mse,
        "final_val_mse": final_val_mse,
        "final_train_mse": final_train_mse,
        "best_step": best_step,
        "completed_steps": completed_steps,
        "early_stop_triggered": early_stop_triggered,
        "total_train_time_sec": total_time,
    }
    logger.add_artifact("task_formula", task)
    return logger.finalize(summary=summary, model_info=model_info)


def _rows(root: Path) -> list[Dict[str, str]]:
    summary = root / "summary.csv"
    if not summary.exists():
        return []
    with summary.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metrics(row: Dict[str, str]) -> Dict[str, Any]:
    try:
        return json.loads(row.get("metrics_json") or "{}")
    except Exception:
        return {}


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return "MISSING"


def _latest_rows(rows: list[Dict[str, str]]) -> list[Dict[str, str]]:
    latest: dict[str, Dict[str, str]] = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if run_id.startswith("kan_operator_"):
            latest[run_id] = row
    return list(latest.values())


def _aggregate_table(rows: list[Dict[str, str]]) -> list[str]:
    grouped: dict[tuple[str, str], list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "COMPLETED":
            grouped[(row.get("dataset_name", ""), row.get("model_name", ""))].append(row)
    lines = [
        "| task | model | n | mean best val MSE | mean final val MSE | mean final train MSE | mean params |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (task, model_name) in sorted(grouped):
        items = grouped[(task, model_name)]

        def mean_metric(key: str, source: str = "metrics") -> float:
            values = []
            for item in items:
                try:
                    raw = _metrics(item).get(key) if source == "metrics" else item.get(key)
                    values.append(float(raw))
                except Exception:
                    pass
            return sum(values) / len(values) if values else float("nan")

        lines.append(
            "| "
            + " | ".join(
                [
                    task,
                    model_name,
                    str(len(items)),
                    _fmt(mean_metric("best_val_mse")),
                    _fmt(mean_metric("final_val_mse")),
                    _fmt(mean_metric("final_train_mse")),
                    _fmt(mean_metric("num_params", source="row")),
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| MISSING | MISSING | 0 | MISSING | MISSING | MISSING | MISSING |")
    return lines


def _run_table(rows: list[Dict[str, str]]) -> list[str]:
    lines = [
        "| run_id | status | task | model | seed | best val MSE | final val MSE | final train MSE | steps | early stop | params | time sec |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: item.get("run_id", "")):
        metrics = _metrics(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("run_id", ""),
                    row.get("status", ""),
                    row.get("dataset_name", ""),
                    row.get("model_name", ""),
                    row.get("seed", ""),
                    _fmt(metrics.get("best_val_mse")),
                    _fmt(metrics.get("final_val_mse")),
                    _fmt(metrics.get("final_train_mse")),
                    str(metrics.get("completed_steps", "")),
                    str(metrics.get("early_stop_triggered", "")),
                    row.get("num_params", ""),
                    _fmt(row.get("total_train_time_sec", "")),
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
    return lines


def generate_report(runs_root: str | Path, output: str | Path) -> Path:
    rows = _latest_rows(_rows(Path(runs_root)))
    lines = [
        "# KAN Operator Replacement Report",
        "",
        "## Scope",
        "- This report compares a compact spline-KAN edge operator with the current stable sEML edge operator under the same KAN-style topology.",
        f"- KAN reference: {KAN_PAPER_URL}",
        f"- EML reference: {EML_PAPER_URL}",
        "- The spline baseline here is a degree-1 B-spline KAN-style implementation, not the external pykan package.",
        "- Lower MSE is better. `best_metric` in raw summaries is stored as negative MSE for compatibility with existing summary fields.",
        "- Rows that reach the configured step cap without early stop are capped ablations, not final model-comparison evidence under the repository validation rules.",
        "",
        "## Operator Difference",
        "| model | edge operator | drive/resistance split | expected strength | expected weakness |",
        "| --- | --- | --- | --- | --- |",
        "| spline_kan | `silu(x) * base + sum_k w_k B_k(x)` | no | local univariate curve fitting through spline control points | no explicit uncertainty/resistance diagnostic |",
        "| semL_operator_replacement | `silu(x) * base + scale * sEML(a*x+b, softplus(s)*(x-c)^2+floor)` | yes | stable EML drive/resistance diagnostics and bounded energy | less local spline resolution per edge |",
        "",
        "## Aggregate Results",
        *_aggregate_table(rows),
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
    for task in args.tasks:
        for seed in args.seeds:
            _run_one("spline_kan", task, seed, args, device)
            _run_one("semL_operator_replacement", task, seed, args, device)
    report = generate_report(args.runs_root, args.output)
    print(json.dumps({"status": "complete", "report": str(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
