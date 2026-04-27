from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_mnist.agent_risk_toy import AgentRiskToyDataset, agent_risk_collate
from eml_mnist.experiment_utils import ExperimentLogger
from eml_mnist.metrics import expected_calibration_error, pearson_corr
from eml_mnist.primitives import EMLUnit
from eml_mnist.training import set_seed


MODELS = ("linear", "mlp", "eml_action", "eml_supervised_resistance")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent-style utility/risk toy benchmark")
    parser.add_argument("--mode", choices=["smoke", "medium"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--runs-root", default="reports/agent_risk_toy/runs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-evals", type=int, default=2)
    return parser


def _sizes(mode: str) -> dict[str, int]:
    return {"train": 256, "test": 128, "steps": 40} if mode == "smoke" else {"train": 2048, "test": 512, "steps": 300}


class LinearActionScorer(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)
        self.risk = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        scores = self.score(features).squeeze(-1)
        risk = F.softplus(self.risk(features).squeeze(-1))
        return {"scores": scores, "risk_score": risk}


class MLPActionScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.risk = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        scores = self.net(features).squeeze(-1)
        risk = F.softplus(self.risk(features).squeeze(-1))
        return {"scores": scores, "risk_score": risk}


class EMLActionScorer(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.drive = nn.Linear(feature_dim, 1)
        self.resistance = nn.Linear(feature_dim, 1)
        self.eml = EMLUnit(dim=1, init_gamma=0.3, init_lambda=1.0)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        drive = self.drive(features)
        resistance = F.softplus(self.resistance(features))
        energy = self.eml(drive, resistance).squeeze(-1)
        return {
            "scores": energy,
            "risk_score": resistance.squeeze(-1),
            "drive": drive.squeeze(-1),
            "resistance": resistance.squeeze(-1),
            "energy": energy,
        }


def _build_model(name: str, feature_dim: int) -> nn.Module:
    if name == "linear":
        return LinearActionScorer(feature_dim)
    if name == "mlp":
        return MLPActionScorer(feature_dim)
    return EMLActionScorer(feature_dim)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    action_correct = []
    unsafe_selected = []
    utility_selected = []
    reward_selected = []
    risk_pred = []
    risk_target = []
    logits_all = []
    targets_all = []
    approval_scores = []
    approvals = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            target = batch["target_action"].to(device)
            out = model(features)
            scores = out["scores"]
            pred = scores.argmax(dim=-1)
            action_correct.append(pred.eq(target).float().cpu())
            unsafe_selected.append(batch["unsafe"].to(device).gather(1, pred.unsqueeze(1)).squeeze(1).cpu())
            utility_selected.append(batch["utility"].to(device).gather(1, pred.unsqueeze(1)).squeeze(1).cpu())
            reward_selected.append(batch["reward"].to(device).gather(1, pred.unsqueeze(1)).squeeze(1).cpu())
            risk = out["risk_score"]
            risk_pred.append(risk.detach().reshape(-1).cpu())
            risk_target.append(batch["risk_target"].reshape(-1).cpu())
            logits_all.append(scores.cpu())
            targets_all.append(target.cpu())
            approval_scores.append(risk.gather(1, pred.unsqueeze(1)).squeeze(1).cpu())
            approvals.append(batch["approval_target"].cpu())
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    approval_score = torch.cat(approval_scores)
    approval_target = torch.cat(approvals)
    approval_pred = approval_score > approval_score.median()
    approval_positive = approval_target.bool()
    precision = (approval_pred & approval_positive).float().sum() / approval_pred.float().sum().clamp_min(1.0)
    recall = (approval_pred & approval_positive).float().sum() / approval_positive.float().sum().clamp_min(1.0)
    return {
        "action_accuracy": float(torch.cat(action_correct).mean().item()),
        "unsafe_action_rate": float(torch.cat(unsafe_selected).mean().item()),
        "expected_utility": float(torch.cat(utility_selected).mean().item()),
        "risk_weighted_reward": float(torch.cat(reward_selected).mean().item()),
        "approval_precision": float(precision.item()),
        "approval_recall": float(recall.item()),
        "risk_corr": pearson_corr(torch.cat(risk_pred), torch.cat(risk_target)),
        "ece": expected_calibration_error(logits, targets),
    }


def _run_one(args: argparse.Namespace, model_name: str, seed: int) -> None:
    set_seed(seed)
    sizes = _sizes(args.mode)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train = AgentRiskToyDataset(sizes["train"], args.num_actions, args.feature_dim, seed=seed)
    test = AgentRiskToyDataset(sizes["test"], args.num_actions, args.feature_dim, seed=seed + 10_000)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=agent_risk_collate)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=agent_risk_collate)
    model = _build_model(model_name, args.feature_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=1.0e-3)
    logger = ExperimentLogger(
        run_id=f"agent_risk_{model_name}_seed{seed}",
        config={
            "mode": args.mode,
            "task_name": "agent_risk_toy",
            "model_name": model_name,
            "dataset_name": "agent_risk_toy",
            "seed": seed,
            "device": str(device),
        },
        root=args.runs_root,
    )
    start = time.time()
    iterator = iter(loader)
    max_steps = int(args.max_steps or sizes["steps"])
    best_metric = float("-inf")
    best_step = 0
    best_summary: Dict[str, float] | None = None
    stale_evals = 0
    eval_count = 0
    early_stop_triggered = False
    steps_run = 0
    for step in range(1, max_steps + 1):
        steps_run = step
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        features = batch["features"].to(device)
        target = batch["target_action"].to(device)
        out = model(features)
        loss = F.cross_entropy(out["scores"], target)
        if model_name == "eml_supervised_resistance":
            loss = loss + 0.25 * F.mse_loss(out["risk_score"], batch["risk_target"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {"step": step, "train_loss": float(loss.detach().cpu().item()), "wall_clock_time_sec": time.time() - start}
        if step % args.eval_interval == 0 or step == max_steps:
            current = _evaluate(model, test_loader, device)
            eval_count += 1
            metric = float(current.get("risk_weighted_reward", float("nan")))
            if metric > best_metric + 1.0e-8:
                best_metric = metric
                best_step = step
                best_summary = current
                stale_evals = 0
            else:
                stale_evals += 1
            row.update({f"eval_{key}": value for key, value in current.items()})
            if eval_count >= args.early_stop_min_evals and stale_evals >= args.early_stop_patience:
                early_stop_triggered = True
                logger.log_step(row, {})
                break
        logger.log_step(row, {})
    summary = best_summary or _evaluate(model, test_loader, device)
    summary.update(
        {
            "total_train_time_sec": time.time() - start,
            "best_metric": best_metric if best_summary is not None else summary["risk_weighted_reward"],
            "final_metric": summary["action_accuracy"],
            "best_step": best_step,
            "steps_run": steps_run,
            "early_stop_triggered": early_stop_triggered,
        }
    )
    logger.finalize(summary=summary)


def _write_report(runs_root: Path) -> None:
    summary = runs_root / "summary.csv"
    report = Path("reports/AGENT_RISK_TOY_REPORT.md")
    lines = ["# Agent Risk Toy Benchmark", "", "| model | seed | action acc | unsafe rate | reward | risk corr |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    if summary.exists():
        with summary.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                metrics = json.loads(row.get("metrics_json") or "{}")
                lines.append(
                    f"| {row.get('model_name')} | {row.get('seed')} | {metrics.get('action_accuracy', 'MISSING')} | {metrics.get('unsafe_action_rate', 'MISSING')} | {metrics.get('risk_weighted_reward', 'MISSING')} | {metrics.get('risk_corr', 'MISSING')} |"
                )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    for seed in args.seeds:
        for model_name in args.models:
            _run_one(args, model_name, seed)
    _write_report(Path(args.runs_root))


if __name__ == "__main__":
    main()
