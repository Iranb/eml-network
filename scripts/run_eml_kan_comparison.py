from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eml_mnist.eml_edge_network import EMLEdgeImageClassifier, EMLEdgeTextLM
from eml_mnist.experiment_utils import ExperimentLogger
from eml_mnist.text_codecs import CharVocabulary
from eml_mnist.training import resolve_device, set_seed
from run_eml_validation_suite import EfficientTextLM, _build_cnn_model, _build_efficient_image, _train_image_model, _train_text_model
from run_text_representation_ablation import LocalCausalConvLM, SmallGRULM


KAN_PAPER_URL = "https://arxiv.org/abs/2404.19756"
EML_PAPER_URL = "https://arxiv.org/html/2603.21852v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KAN-inspired EML edge-function comparisons")
    parser.add_argument("--mode", choices=["smoke", "ablation"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-root", default="reports/runs")
    parser.add_argument("--output", default="reports/EML_KAN_COMPARISON_REPORT.md")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--staged-hardening", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--responsibility-temp-start", type=float, default=2.0)
    parser.add_argument("--responsibility-temp-end", type=float, default=0.8)
    parser.add_argument("--ambiguity-warmup-steps", type=int, default=20)
    parser.add_argument("--null-threshold-start", type=float, default=1.0)
    parser.add_argument("--null-threshold-end", type=float, default=0.0)
    return parser


def _default_steps(args: argparse.Namespace) -> int:
    if args.steps > 0:
        return args.steps
    return 8 if args.mode == "smoke" else 50


def _run_image(
    run_id: str,
    model_name: str,
    factory: Callable[[], nn.Module],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> None:
    set_seed(seed)
    _train_image_model(run_id, model_name, factory(), args, device, seed, warmup_enabled=True)


def _run_text(
    run_id: str,
    model_name: str,
    factory: Callable[[], nn.Module],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> None:
    set_seed(seed)
    _train_text_model(run_id, model_name, factory(), args, device, seed)


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
        return f"{float(value):.4f}"
    except Exception:
        return "MISSING"


def _result_table(rows: list[Dict[str, str]], task_name: str) -> list[str]:
    selected = [row for row in rows if row.get("task_name") == task_name]
    lines = [
        "| run_id | status | model | best | final | loss | steps | early stop | params | time sec | reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    if not selected:
        lines.append("| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
        return lines
    for row in selected:
        metrics = _metrics(row)
        loss = metrics.get("final_train_loss", metrics.get("train_loss", ""))
        early_stop = metrics.get("early_stop_triggered", "")
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("run_id", ""),
                    row.get("status", ""),
                    row.get("model_name", ""),
                    _fmt(row.get("best_metric", "")),
                    _fmt(row.get("final_metric", "")),
                    _fmt(loss),
                    str(metrics.get("completed_steps", "")),
                    str(early_stop),
                    row.get("num_params", ""),
                    _fmt(row.get("total_train_time_sec", "")),
                    row.get("reason", ""),
                ]
            )
            + " |"
        )
    return lines


def _group_best(rows: list[Dict[str, str]], task_name: str) -> list[str]:
    grouped: dict[str, list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "COMPLETED" and row.get("task_name") == task_name:
            grouped[row.get("model_name", "")].append(row)
    lines = [
        "| model | n | best metric | mean final metric | mean loss | mean params |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not grouped:
        lines.append("| MISSING | 0 | MISSING | MISSING | MISSING | MISSING |")
        return lines
    for model_name in sorted(grouped):
        items = grouped[model_name]

        def avg_value(key: str, source: str = "row") -> float:
            values = []
            for item in items:
                try:
                    if source == "metrics":
                        values.append(float(_metrics(item).get(key)))
                    else:
                        values.append(float(item.get(key)))
                except Exception:
                    pass
            return sum(values) / len(values) if values else float("nan")

        best = max(float(item.get("best_metric", "nan")) for item in items)
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    str(len(items)),
                    _fmt(best),
                    _fmt(avg_value("final_metric")),
                    _fmt(avg_value("final_train_loss", "metrics")),
                    _fmt(avg_value("num_params")),
                ]
            )
            + " |"
        )
    return lines


def generate_report(runs_root: str | Path, output: str | Path) -> Path:
    selected = [
        row
        for row in _rows(Path(runs_root))
        if row.get("run_id", "").startswith("kan_compare_") or row.get("run_id", "") == "kan_paper_reference"
    ]
    latest_by_run_id: dict[str, Dict[str, str]] = {}
    for row in selected:
        latest_by_run_id[row.get("run_id", "")] = row
    rows = list(latest_by_run_id.values())
    local_rows = [row for row in rows if row.get("run_id", "").startswith("kan_compare_")]
    early_stopped = [
        row
        for row in local_rows
        if str(_metrics(row).get("early_stop_triggered", "")).lower() == "true"
    ]
    early_stop_note = f"{len(early_stopped)}/{len(local_rows)} local comparison rows early-stopped."
    lines = [
        "# EML KAN-Style Comparison Report",
        "",
        "## Scope",
        "- Local experiments run only EML-native models on existing synthetic image/text validation tasks.",
        f"- KAN is paper-only per request: {KAN_PAPER_URL}",
        f"- EML operator reference: {EML_PAPER_URL}",
        "- This is not evidence that EML beats KAN on KAN's original function-fitting/PDE tasks; task families differ.",
        f"- Early-stop completeness: {early_stop_note}",
        "",
        "## Architecture Translation",
        "- KAN paper structure used here: layers are matrices of learnable univariate edge functions and destination nodes sum incoming edge outputs.",
        "- Local EML implementation: each edge function is `base(silu(x)) + scale * sEML(drive(x), resistance(x))`.",
        "- Direct `exp(x) - log(y)` is not stacked raw; the repository's stable sEML primitive keeps fp32 islands and bounded drive for training stability.",
        "",
        "## Paper-Only KAN Comparator",
        "| source | comparator status | relevant result | local action |",
        "| --- | --- | --- | --- |",
        "| KAN arXiv 2404.19756 | NOT RUN | Reports smaller KANs can match or beat larger MLPs on small AI+Science function-fitting tasks, while Feynman KAN/MLP behavior is comparable on average. | Marked `NOT RUN`; no local KAN experiment was run. |",
        "| EML arXiv 2603.21852v2 | Reference primitive | Defines EML as a universal binary expression-tree operator and reports symbolic-regression proof-of-concept, with harder blind recovery at deeper trees. | Implemented stable sEML edge functions, not raw complex EML trees. |",
        "",
        "## Aggregated Image Results",
        *_group_best(rows, "image_synthetic"),
        "",
        "## Aggregated Text Results",
        *_group_best(rows, "text_synthetic"),
        "",
        "## Image Runs",
        *_result_table(rows, "image_synthetic"),
        "",
        "## Text Runs",
        *_result_table(rows, "text_synthetic"),
        "",
        "## Missing Or Failed",
        "| run_id | status | model | reason |",
        "| --- | --- | --- | --- |",
    ]
    missing = [row for row in rows if row.get("status") != "COMPLETED"]
    if missing:
        for row in missing:
            lines.append(f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('model_name', '')} | {row.get('reason', '')} |")
    else:
        lines.append("| none | none | none | none |")
    lines.extend(["", "## Raw Artifacts"])
    for row in rows:
        lines.append(f"- `{row.get('run_id', '')}`: `{row.get('run_dir', '')}`")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    args.steps = _default_steps(args)
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    device = resolve_device(args.device)
    Path(args.runs_root).mkdir(parents=True, exist_ok=True)

    ExperimentLogger.not_run(
        run_id="kan_paper_reference",
        config={
            "mode": args.mode,
            "task_name": "paper_reference",
            "model_name": "KAN_arxiv_2404_19756",
            "dataset_name": "KAN paper AI+Science tasks",
            "seed": args.seed,
            "device": str(device),
        },
        reason="User requested no local KAN experiment; comparing against reported paper results only.",
        root=args.runs_root,
    )

    image_specs = [
        ("kan_compare_image_cnn_eml", "cnn_eml", lambda: _build_cnn_model(args.image_size), args.seed),
        ("kan_compare_image_efficient_eml", "EfficientEMLImageClassifier", lambda: _build_efficient_image(4, 3), args.seed + 1),
        (
            "kan_compare_image_eml_edge",
            "EMLEdgeImageClassifier_kan_style",
            lambda: EMLEdgeImageClassifier(num_classes=5, input_channels=3, state_dim=32, edge_width=32),
            args.seed + 2,
        ),
    ]
    for run_id, model_name, factory, seed in image_specs:
        _run_image(run_id, model_name, factory, args, device, seed)

    vocab = CharVocabulary()
    text_specs = [
        ("kan_compare_text_local_conv", "LocalCausalConvLM", lambda: LocalCausalConvLM(len(vocab), vocab.pad_id), args.seed + 10),
        ("kan_compare_text_small_gru", "SmallGRULM", lambda: SmallGRULM(len(vocab), vocab.pad_id), args.seed + 11),
        ("kan_compare_text_efficient_eml", "EfficientEMLTextEncoder", lambda: EfficientTextLM(len(vocab), vocab.pad_id), args.seed + 12),
        (
            "kan_compare_text_eml_edge",
            "EMLEdgeTextLM_kan_style",
            lambda: EMLEdgeTextLM(len(vocab), vocab.pad_id, state_dim=32, edge_width=48),
            args.seed + 13,
        ),
    ]
    for run_id, model_name, factory, seed in text_specs:
        _run_text(run_id, model_name, factory, args, device, seed)

    report = generate_report(args.runs_root, args.output)
    print(report)


if __name__ == "__main__":
    main()
