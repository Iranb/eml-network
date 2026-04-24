from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CNN head ablation Markdown report")
    parser.add_argument("--runs-root", default="reports/head_ablation/runs")
    parser.add_argument("--output", default="reports/HEAD_ABLATION_REPORT.md")
    return parser


def _read_rows(root: Path) -> list[Dict[str, Any]]:
    summary_path = root / "summary.csv"
    if not summary_path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run_dir = Path(row.get("run_dir", ""))
            summary_json = run_dir / "summary.json"
            if summary_json.exists():
                try:
                    full = json.loads(summary_json.read_text(encoding="utf-8"))
                    row.update(full)
                except Exception:
                    pass
            rows.append(row)
    return rows


def _num(value: Any) -> float:
    try:
        if value == "" or value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _fmt(value: Any, digits: int = 4) -> str:
    number = _num(value)
    if math.isnan(number):
        return "MISSING"
    return f"{number:.{digits}f}"


def _metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(row.get("metrics_json"), str) and row["metrics_json"]:
        try:
            return json.loads(row["metrics_json"])
        except Exception:
            return {}
    return {}


def _status_table(rows: Iterable[Dict[str, Any]]) -> list[str]:
    lines = ["| run_id | status | experiment | model | dataset | seed | reason |", "| --- | --- | --- | --- | --- | ---: | --- |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_id", "")),
                    str(row.get("status", "")),
                    str(row.get("experiment_type", "")),
                    str(row.get("model_name", "")),
                    str(row.get("dataset_name", "")),
                    str(row.get("seed", "")),
                    str(row.get("reason", "")),
                ]
            )
            + " |"
        )
    return lines


def _result_table(rows: Iterable[Dict[str, Any]], title: str, include_loss_mode: bool = False) -> list[str]:
    lines = [f"### {title}"]
    if include_loss_mode:
        lines.append("| run_id | seed | model | loss mode | test acc | val acc | test loss | ECE | Brier | margin | time sec |")
        lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    else:
        lines.append("| run_id | seed | model | test acc | val acc | test loss | ECE | Brier | margin | time sec |")
        lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        metrics = _metrics(row)
        cells = [
            str(row.get("run_id", "")),
            str(row.get("seed", "")),
            str(row.get("model_name", "")),
        ]
        if include_loss_mode:
            cells.append(str(row.get("loss_mode", "")))
        cells.extend(
            [
                _fmt(metrics.get("test_accuracy")),
                _fmt(metrics.get("val_accuracy")),
                _fmt(metrics.get("test_loss")),
                _fmt(metrics.get("test_ece")),
                _fmt(metrics.get("test_brier")),
                _fmt(metrics.get("test_margin_mean")),
                _fmt(metrics.get("total_train_time_sec")),
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _eml_table(rows: Iterable[Dict[str, Any]]) -> list[str]:
    lines = [
        "| run_id | model | pos drive | hard neg drive | pos resistance | hard neg resistance | uncertainty | ambiguity | noise corr | occlusion corr |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = _metrics(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_id", "")),
                    str(row.get("model_name", "")),
                    _fmt(metrics.get("test_positive_drive_mean")),
                    _fmt(metrics.get("test_hard_negative_drive_mean")),
                    _fmt(metrics.get("test_positive_resistance_mean")),
                    _fmt(metrics.get("test_hard_negative_resistance_mean")),
                    _fmt(metrics.get("test_sample_uncertainty_mean")),
                    _fmt(metrics.get("test_ambiguity_mean")),
                    _fmt(metrics.get("test_resistance_noise_corr")),
                    _fmt(metrics.get("test_resistance_occlusion_corr")),
                ]
            )
            + " |"
        )
    return lines


def _load_correct(row: Dict[str, Any]) -> torch.Tensor | None:
    run_dir = Path(str(row.get("run_dir", "")))
    path = run_dir / "eval_predictions.pt"
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    correct = payload.get("test_correct")
    if torch.is_tensor(correct):
        return correct.bool().reshape(-1)
    return None


def _bootstrap_delta(target: torch.Tensor, baseline: torch.Tensor, samples: int = 1000) -> tuple[float, float, float]:
    if target.numel() != baseline.numel() or target.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    delta = target.float() - baseline.float()
    generator = torch.Generator().manual_seed(1234)
    estimates = []
    for _ in range(samples):
        index = torch.randint(0, delta.numel(), (delta.numel(),), generator=generator)
        estimates.append(delta[index].mean())
    values = torch.stack(estimates)
    return (
        float(delta.mean().item()),
        float(torch.quantile(values, 0.025).item()),
        float(torch.quantile(values, 0.975).item()),
    )


def _ci_table(rows: list[Dict[str, Any]]) -> list[str]:
    grouped: Dict[tuple[str, str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "COMPLETED":
            continue
        key = (
            str(row.get("experiment_type", "")),
            str(row.get("dataset_name", "")),
            str(row.get("seed", "")),
            str(row.get("loss_mode", "")),
        )
        grouped[key][str(row.get("model_name", ""))] = row
    lines = [
        "| experiment | dataset | seed | loss mode | comparison | delta acc | 95% CI low | 95% CI high |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    baselines = ["linear", "mlp", "cosine_prototype"]
    for key, by_model in sorted(grouped.items()):
        target = by_model.get("eml_centered_ambiguity")
        if target is None:
            continue
        target_correct = _load_correct(target)
        if target_correct is None:
            continue
        for baseline_name in baselines:
            baseline = by_model.get(baseline_name)
            if baseline is None:
                continue
            baseline_correct = _load_correct(baseline)
            if baseline_correct is None:
                continue
            mean, low, high = _bootstrap_delta(target_correct, baseline_correct)
            lines.append(
                "| "
                + " | ".join(
                    [
                        key[0],
                        key[1],
                        key[2],
                        key[3] or "ce",
                        f"eml_centered_ambiguity - {baseline_name}",
                        _fmt(mean),
                        _fmt(low),
                        _fmt(high),
                    ]
                )
                + " |"
            )
    if len(lines) == 2:
        lines.append("| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
    return lines


def _best_line(rows: list[Dict[str, Any]], experiment_type: str) -> str:
    completed = [row for row in rows if row.get("status") == "COMPLETED" and row.get("experiment_type") == experiment_type]
    if not completed:
        return "MISSING"
    best = max(completed, key=lambda row: _num(_metrics(row).get("test_accuracy")))
    metrics = _metrics(best)
    return f"{best.get('model_name')} seed={best.get('seed')} test_accuracy={_fmt(metrics.get('test_accuracy'))}"


def _claim_text(rows: list[Dict[str, Any]]) -> str:
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    grouped: Dict[tuple[str, str, str, str], Dict[str, float]] = defaultdict(dict)
    for row in completed:
        key = (
            str(row.get("experiment_type", "")),
            str(row.get("dataset_name", "")),
            str(row.get("seed", "")),
            str(row.get("loss_mode", "")),
        )
        grouped[key][str(row.get("model_name", ""))] = _num(_metrics(row).get("test_accuracy"))
    wins = 0
    comparisons = 0
    for values in grouped.values():
        target = values.get("eml_centered_ambiguity")
        if target is None:
            continue
        for baseline in ("linear", "mlp", "cosine_prototype"):
            if baseline in values:
                comparisons += 1
                wins += int(target > values[baseline])
    if comparisons == 0:
        return "The current artifacts do not contain enough paired runs to support or reject the EML-head claim."
    if wins == comparisons:
        return "In these runs the centered EML prototype head beats all paired ordinary heads, but this still needs longer runs and more seeds."
    if wins == 0:
        return "These runs do not support the claim that the centered EML prototype head is better than ordinary heads."
    return f"The evidence is mixed: centered EML wins {wins}/{comparisons} paired comparisons."


def generate_report(runs_root: str | Path, output: str | Path) -> Path:
    root = Path(runs_root)
    rows = _read_rows(root)
    rows_sorted = sorted(rows, key=lambda row: str(row.get("run_id", "")))
    frozen = [row for row in rows_sorted if row.get("experiment_type") == "frozen_features"]
    e2e = [row for row in rows_sorted if row.get("experiment_type") == "end_to_end"]
    completed = [row for row in rows_sorted if row.get("status") == "COMPLETED"]
    not_run = [row for row in rows_sorted if row.get("status") == "NOT RUN"]
    failed = [row for row in rows_sorted if row.get("status") == "FAILED"]

    lines: list[str] = [
        "# CNN Head Ablation Report",
        "",
        "## 1. Executive Summary",
        f"- Completed runs: {len(completed)}",
        f"- NOT RUN entries: {len(not_run)}",
        f"- Failed runs: {len(failed)}",
        f"- Best frozen-feature result: {_best_line(rows_sorted, 'frozen_features')}",
        f"- Best end-to-end result: {_best_line(rows_sorted, 'end_to_end')}",
        f"- Claim status: {_claim_text(rows_sorted)}",
        "",
        "## 2. Experimental Setup",
        "- Frozen-feature runs train one shared CNN feature extractor per dataset/seed, cache features, then train only the selected head.",
        "- End-to-end runs train the same CNN backbone with one selected head; the EML residual-bank variant is reported separately.",
        "- CE-only and prototype-pairwise settings are separated. Linear and MLP heads are marked NOT RUN for prototype-pairwise because that loss is not applicable.",
        "",
        "## 3. Run Status",
        *_status_table(rows_sorted),
        "",
        "## 4. Frozen Feature Results",
        *_result_table(frozen, "Frozen CNN Features"),
        "",
        "## 5. End-To-End Results",
        *_result_table(e2e, "CNN Plus Head", include_loss_mode=True),
        "",
        "## 6. CE-Only Comparison",
        *_result_table([row for row in e2e if row.get("loss_mode", "ce") == "ce"], "End-To-End CE Only", include_loss_mode=True),
        "",
        "## 7. CE + Pairwise Comparison",
        *_result_table([row for row in e2e if row.get("loss_mode") == "ce_pairwise"], "End-To-End CE + Prototype Pairwise", include_loss_mode=True),
        "",
        "## 8. Calibration Metrics",
        "ECE and Brier score are included in the result tables. Lower is better for both.",
        "",
        "## 9. Hard-Negative Margin Analysis",
        "Margin is positive-logit minus hardest-negative-logit; larger is better.",
        "",
        "## 10. EML Drive/Resistance Analysis",
        *_eml_table([row for row in rows_sorted if str(row.get("model_name", "")).startswith("eml")]),
        "",
        "## 11. Robustness Under Noise/Occlusion",
        "Resistance-noise and resistance-occlusion correlations are reported when synthetic metadata is available. MISSING means the head did not expose resistance or the dataset did not provide the field.",
        "",
        "## 12. Statistical Confidence Intervals",
        *_ci_table(rows_sorted),
        "",
        "## 13. Which Claim Is Supported",
        _claim_text(rows_sorted),
        "",
        "## 14. Raw Artifacts",
    ]
    for row in rows_sorted:
        lines.append(f"- `{row.get('run_id', '')}`: `{row.get('run_dir', '')}`")
    lines.extend(
        [
            "",
            "## 15. Appendix: Commands",
            "- `pytest`",
            "- `python scripts/run_head_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`",
            "- `python scripts/run_cnn_head_end_to_end_ablation.py --dataset synthetic_shape --mode smoke --seeds 0 1`",
            "- `python scripts/generate_head_ablation_report.py`",
        ]
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    output = generate_report(args.runs_root, args.output)
    print(output)


if __name__ == "__main__":
    main()
