from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_mnist.experiment_utils import ExperimentLogger
from eml_mnist.merc_toy_tasks import TOY_MODELS, TOY_TASKS
from eml_mnist.metrics import brier_score, classification_accuracy, expected_calibration_error, negative_log_likelihood, pearson_corr
from eml_mnist.training import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MERC toy nonlinear experiments")
    parser.add_argument("--mode", choices=["smoke", "medium"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--runs-root", default="reports/merc_toy/runs")
    parser.add_argument("--report", default="reports/MERC_TOY_REPORT.md")
    return parser


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _steps(args: argparse.Namespace) -> int:
    if args.steps > 0:
        return args.steps
    return 40 if args.mode == "smoke" else 300


def _summary(rows: list[Dict[str, Any]], output: Path) -> None:
    lines = [
        "# MERC Toy Report",
        "",
        "This report only states observed toy-task results.",
        "",
        "| run_id | task | model | seed | best acc | final acc | ece | evidence corr | conflict corr | steps to 0.9 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["run_id"],
                    row["task"],
                    row["model"],
                    str(row["seed"]),
                    f"{row['best_acc']:.4f}",
                    f"{row['final_acc']:.4f}",
                    f"{row['ece']:.4f}",
                    ("MISSING" if math.isnan(row["evidence_corr"]) else f"{row['evidence_corr']:.4f}"),
                    ("MISSING" if math.isnan(row["conflict_corr"]) else f"{row['conflict_corr']:.4f}"),
                    ("MISSING" if row["steps_to_90"] < 0 else str(row["steps_to_90"])),
                ]
            )
            + " |"
        )
    merc_rows = [row for row in rows if row["model"] in {"merc", "merc_energy"}]
    mlp_rows = [row for row in rows if row["model"] == "mlp"]
    verdict = "MERC neuron design is not yet justified."
    if merc_rows and mlp_rows:
        if max(row["final_acc"] for row in merc_rows) > max(row["final_acc"] for row in mlp_rows):
            verdict = "MERC beats MLP on at least one toy task in this run."
    lines.extend(["", "## Conclusion", "", f"- {verdict}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    device = _device(args.device)
    steps = _steps(args)
    rows: list[Dict[str, Any]] = []
    for seed in args.seeds:
        for task_name, task_fn in TOY_TASKS.items():
            for model_name, model_ctor in TOY_MODELS.items():
                set_seed(seed)
                model = model_ctor(args.input_dim).to(device)
                optimizer = AdamW(model.parameters(), lr=args.lr)
                run_id = f"merc_toy_{task_name}_{model_name}_seed{seed}"
                logger = ExperimentLogger(
                    run_id=run_id,
                    config={
                        "mode": "merc_toy",
                        "task_name": task_name,
                        "model_name": model_name,
                        "seed": seed,
                        "device": str(device),
                        "steps": steps,
                    },
                    root=args.runs_root,
                )
                model_info = logger.set_model_info(model)
                best_acc = 0.0
                steps_to_90 = -1
                start = time.time()
                last_eval: Dict[str, Any] = {}
                for step in range(1, steps + 1):
                    batch = task_fn(args.batch_size, args.input_dim, seed * 10000 + step)
                    x = batch.inputs.to(device)
                    y = batch.labels.to(device)
                    out = model(x)
                    loss = F.cross_entropy(out["logits"], y)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    acc = classification_accuracy(out["logits"].detach(), y)
                    if acc >= 0.9 and steps_to_90 < 0:
                        steps_to_90 = step
                    metrics = {
                        "step": step,
                        "train_loss": float(loss.detach().cpu().item()),
                        "train_accuracy": acc,
                        "wall_clock_time_sec": time.time() - start,
                    }
                    if step % max(1, steps // 4) == 0 or step == steps:
                        eval_batch = task_fn(args.batch_size * 2, args.input_dim, seed * 10000 + 1000 + step)
                        eval_out = model(eval_batch.inputs.to(device))
                        logits = eval_out["logits"].detach().cpu()
                        labels = eval_batch.labels.cpu()
                        eval_acc = classification_accuracy(logits, labels)
                        best_acc = max(best_acc, eval_acc)
                        metrics.update(
                            {
                                "val_accuracy": eval_acc,
                                "val_loss": negative_log_likelihood(logits, labels),
                                "val_ece": expected_calibration_error(logits, labels),
                                "val_brier": brier_score(logits, labels),
                            }
                        )
                        if torch.is_tensor(eval_out.get("support_factors")):
                            metrics["val_support_evidence_corr"] = pearson_corr(
                                eval_out["support_factors"].detach().cpu().mean(dim=-1),
                                eval_batch.evidence_target,
                            )
                        if torch.is_tensor(eval_out.get("conflict_factors")):
                            metrics["val_conflict_corr"] = pearson_corr(
                                eval_out["conflict_factors"].detach().cpu().mean(dim=-1),
                                eval_batch.conflict_target,
                            )
                        logger.log_step(metrics, {})
                        last_eval = metrics
                    else:
                        logger.log_step(metrics, {})
                summary = {
                    "best_metric": best_acc,
                    "final_metric": last_eval.get("val_accuracy", float("nan")),
                    "final_val_accuracy": last_eval.get("val_accuracy", float("nan")),
                    "final_val_ece": last_eval.get("val_ece", float("nan")),
                    "evidence_corr": last_eval.get("val_support_evidence_corr", float("nan")),
                    "conflict_corr": last_eval.get("val_conflict_corr", float("nan")),
                    "steps_to_90": steps_to_90,
                    "total_train_time_sec": time.time() - start,
                }
                logger.finalize(summary=summary, model_info=model_info)
                rows.append(
                    {
                        "run_id": run_id,
                        "task": task_name,
                        "model": model_name,
                        "seed": seed,
                        "best_acc": best_acc,
                        "final_acc": summary["final_val_accuracy"],
                        "ece": summary["final_val_ece"],
                        "evidence_corr": summary["evidence_corr"],
                        "conflict_corr": summary["conflict_corr"],
                        "steps_to_90": steps_to_90,
                    }
                )
    _summary(rows, Path(args.report))
    print(json.dumps({"status": "complete", "report": args.report, "runs_root": args.runs_root}, sort_keys=True))


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
