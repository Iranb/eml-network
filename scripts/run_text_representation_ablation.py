from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eml_mnist.experiment_utils import ExperimentLogger
from eml_mnist.eml_edge_network import EMLEdgeTextLM
from eml_mnist.text_backbones import EMLTextBackbone
from eml_mnist.text_codecs import CharVocabulary
from eml_mnist.text_heads import LocalTextGenerationHead
from eml_mnist.training import resolve_device, set_seed
from run_eml_validation_suite import EfficientTextLM, _train_text_model


class LocalCausalConvLM(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, state_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, state_dim, padding_idx=pad_id)
        self.conv1 = nn.Conv1d(state_dim, state_dim, kernel_size=3)
        self.conv2 = nn.Conv1d(state_dim, state_dim, kernel_size=3)
        self.out = nn.Linear(state_dim, vocab_size)
        self.pad_id = pad_id

    def _conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        return conv(F.pad(x, (conv.kernel_size[0] - 1, 0)))

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor, warmup_eta: float = 1.0) -> Dict[str, Any]:
        del warmup_eta
        x = self.embedding(input_ids).transpose(1, 2)
        x = F.gelu(self._conv(x, self.conv1))
        x = F.gelu(self._conv(x, self.conv2)).transpose(1, 2)
        x = torch.where(padding_mask.unsqueeze(-1), x, torch.zeros_like(x))
        logits = self.out(x)
        return {"logits": logits, "sequence_states": x, "diagnostics": {}}


class SmallGRULM(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, state_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, state_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(state_dim, state_dim, batch_first=True)
        self.out = nn.Linear(state_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor, warmup_eta: float = 1.0) -> Dict[str, Any]:
        del warmup_eta
        x = self.embedding(input_ids)
        states, _hidden = self.rnn(x)
        states = torch.where(padding_mask.unsqueeze(-1), states, torch.zeros_like(states))
        return {"logits": self.out(states), "sequence_states": states, "diagnostics": {}}


class OldEMLTextLM(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, state_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = EMLTextBackbone(
            vocab_size=vocab_size,
            embed_dim=24,
            feature_dim=state_dim,
            event_dim=state_dim,
            hidden_dim=hidden_dim,
            bank_dim=state_dim,
            num_layers=1,
            causal_window_size=5,
            dropout=0.0,
            pad_id=pad_id,
        )
        self.head = LocalTextGenerationHead(
            context_dim=state_dim,
            token_dim=state_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
        )

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor, warmup_eta: float = 1.0) -> Dict[str, Any]:
        encoded = self.backbone(input_ids, padding_mask=padding_mask, warmup_eta=warmup_eta)
        head = self.head(
            encoded["pooled_representation"],
            encoded["sequence_features"],
            padding_mask=padding_mask,
            warmup_eta=warmup_eta,
        )
        return {
            **head,
            "sequence_states": encoded["sequence_features"],
            "representation": encoded["pooled_representation"],
            "diagnostics": {},
            "encoder": encoded,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic text representation ablations")
    parser.add_argument("--mode", choices=["smoke", "ablation"], default="smoke")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs-root", default="reports/text_representation_ablation/runs")
    parser.add_argument("--output", default="reports/TEXT_REPRESENTATION_ABLATION_REPORT.md")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
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
    return 4 if args.mode == "smoke" else 80


def _run_args(args: argparse.Namespace, staged: bool) -> argparse.Namespace:
    child = copy.copy(args)
    child.steps = _default_steps(args)
    child.staged_hardening = staged or bool(args.staged_hardening)
    return child


def _safe_text_run(
    run_id: str,
    model_name: str,
    factory: Callable[[], nn.Module],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    staged: bool = False,
) -> None:
    child_args = _run_args(args, staged=staged)
    try:
        set_seed(seed)
        _train_text_model(run_id, model_name, factory(), child_args, device, seed)
    except Exception as exc:
        trace = traceback.format_exc()
        logger = ExperimentLogger(
            run_id=run_id,
            config={
                "mode": args.mode,
                "task_name": "text_synthetic",
                "model_name": model_name,
                "dataset_name": "SyntheticTextEnergyDataset",
                "seed": seed,
                "device": str(device),
                "num_workers": args.num_workers,
            },
            root=args.runs_root,
        )
        logger.set_model_info(extra={"num_params": 0, "trainable_params": 0})
        logger.log_text(trace)
        logger.finalize(summary={"error_trace": trace}, status="FAILED", reason=repr(exc))


def _rows(root: Path) -> list[Dict[str, str]]:
    path = root / "summary.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metrics(row: Dict[str, str]) -> Dict[str, Any]:
    try:
        metrics = json.loads(row.get("metrics_json") or "{}")
    except Exception:
        metrics = {}
    run_dir = row.get("run_dir")
    if run_dir:
        try:
            summary = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
            diagnostics = summary.get("final_diagnostics") or {}
            if isinstance(diagnostics, dict):
                metrics.update({key: value for key, value in diagnostics.items() if key not in metrics})
        except Exception:
            pass
    return metrics


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "MISSING"


def generate_report(runs_root: str | Path, output: str | Path) -> Path:
    rows = _rows(Path(runs_root))
    grouped: dict[str, list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "COMPLETED":
            grouped[row.get("model_name", "")].append(row)
    lines = [
        "# Text Representation Ablation Report",
        "",
        "## Summary",
        f"- Completed runs: {sum(1 for row in rows if row.get('status') == 'COMPLETED')}",
        f"- Failed runs: {sum(1 for row in rows if row.get('status') == 'FAILED')}",
        f"- NOT RUN entries: {sum(1 for row in rows if row.get('status') == 'NOT RUN')}",
        "- Window size `8` is the default efficient text path in this report.",
        "",
        "## Results",
        "| model | n | best token acc | mean token acc | mean loss | bits/token | null weight | update gate | entropy | attractor diversity | corruption corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in sorted(grouped):
        items = grouped[model_name]
        metrics = [_metrics(item) for item in items]

        def avg(key: str) -> float:
            values = []
            for item in metrics:
                try:
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
                    _fmt(avg("final_train_accuracy")),
                    _fmt(avg("final_train_loss")),
                    _fmt(avg("bits_per_char")),
                    _fmt(avg("null_weight_mean")),
                    _fmt(avg("update_gate_mean")),
                    _fmt(avg("responsibility_entropy_mean")),
                    _fmt(avg("attractor_diversity")),
                    _fmt(avg("corruption_resistance_corr")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Missing Or Failed",
            "| run_id | status | model | reason |",
            "| --- | --- | --- | --- |",
        ]
    )
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
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    device = resolve_device(args.device)
    Path(args.runs_root).mkdir(parents=True, exist_ok=True)
    vocab = CharVocabulary()
    for seed in args.seeds:
        specs = [
            ("local_causal_conv", "LocalCausalConvLM", lambda: LocalCausalConvLM(len(vocab), vocab.pad_id), False),
            ("eml_edge_kan_text", "EMLEdgeTextLM_kan_style", lambda: EMLEdgeTextLM(len(vocab), vocab.pad_id), False),
            ("small_gru", "SmallGRULM", lambda: SmallGRULM(len(vocab), vocab.pad_id), False),
            ("old_eml_text_backbone", "EMLTextBackbone", lambda: OldEMLTextLM(len(vocab), vocab.pad_id), False),
            ("efficient_window8", "EfficientEMLTextEncoder_window8", lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, enable_composition=False, enable_attractor=False), False),
            ("efficient_window8_thresholded_null", "EfficientEMLTextEncoder_window8_thresholded_null", lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, responsibility_mode="thresholded_null", enable_composition=False, enable_attractor=False), False),
            ("efficient_window8_precision_identity", "EfficientEMLTextEncoder_window8_precision_identity", lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, precision_old_confidence_init=5.0, enable_composition=False, enable_attractor=False), False),
            ("efficient_window8_chunk", "EfficientEMLTextEncoder_window8_chunk", lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, enable_composition=True, enable_attractor=False), False),
            ("efficient_window8_chunk_attractor", "EfficientEMLTextEncoder_window8_chunk_attractor", lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, enable_composition=True, enable_attractor=True), False),
            ("efficient_window8_staged", "EfficientEMLTextEncoder_window8_staged", lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, responsibility_mode="thresholded_null", enable_composition=True, enable_attractor=True), True),
            ("best_text_config", "EfficientEMLTextEncoder_best_text_config", lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, responsibility_mode="thresholded_null", precision_old_confidence_init=5.0, enable_composition=True, enable_attractor=True), True),
        ]
        for name, model_name, factory, staged in specs:
            _safe_text_run(f"{name}_seed{seed}", model_name, factory, args, device, seed, staged=staged)
    report = generate_report(args.runs_root, args.output)
    print(report)


if __name__ == "__main__":
    main()
