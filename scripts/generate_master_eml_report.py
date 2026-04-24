from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


REPORTS = {
    "head": Path("reports/HEAD_ABLATION_REPORT.md"),
    "cnn_head": Path("reports/CNN_HEAD_END_TO_END_REPORT.md"),
    "mechanism": Path("reports/MECHANISM_PROBE_REPORT.md"),
    "image": Path("reports/IMAGE_REPRESENTATION_ABLATION_REPORT.md"),
    "text": Path("reports/TEXT_REPRESENTATION_ABLATION_REPORT.md"),
    "cifar": Path("reports/CIFAR_MEDIUM_REPORT.md"),
}

SUMMARY_ROOTS = {
    "head": Path("reports/head_ablation/runs"),
    "mechanism": Path("reports/mechanism_probes/runs"),
    "image": Path("reports/image_representation_ablation/runs"),
    "text": Path("reports/text_representation_ablation/runs"),
    "cifar": Path("reports/cifar_medium/runs"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the master EML next-step report")
    parser.add_argument("--output", default="reports/EML_MASTER_NEXT_STEP_REPORT.md")
    return parser


def _rows(root: Path) -> list[Dict[str, Any]]:
    path = root / "summary.csv"
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run_dir = Path(row.get("run_dir", ""))
            summary_json = run_dir / "summary.json"
            if summary_json.exists():
                try:
                    row.update(json.loads(summary_json.read_text(encoding="utf-8")))
                except Exception:
                    pass
            rows.append(row)
    return rows


def _all_rows() -> dict[str, list[Dict[str, Any]]]:
    return {name: _rows(root) for name, root in SUMMARY_ROOTS.items()}


def _num(value: Any) -> float:
    try:
        if value in ("", None):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _fmt(value: Any) -> str:
    number = _num(value)
    if math.isnan(number):
        return "MISSING"
    return f"{number:.4f}"


def _metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    raw = row.get("metrics_json")
    if isinstance(raw, str) and raw:
        try:
            payload.update(json.loads(raw))
        except Exception:
            pass
    diagnostics = row.get("final_diagnostics")
    if isinstance(diagnostics, dict):
        for key, value in diagnostics.items():
            payload.setdefault(key, value)
    return payload


def _best(rows: Iterable[Dict[str, Any]], metric: str = "best_metric") -> Dict[str, Any] | None:
    best_row: Dict[str, Any] | None = None
    best_value = float("-inf")
    for row in rows:
        if row.get("status") != "COMPLETED":
            continue
        value = _num(row.get(metric))
        if math.isnan(value):
            value = _num(_metrics(row).get(metric))
        if value > best_value:
            best_value = value
            best_row = row
    return best_row


def _status_counts(rows: Iterable[Dict[str, Any]]) -> tuple[int, int, int]:
    completed = failed = not_run = 0
    for row in rows:
        status = row.get("status")
        if status == "COMPLETED":
            completed += 1
        elif status == "FAILED":
            failed += 1
        elif status == "NOT RUN":
            not_run += 1
    return completed, failed, not_run


def _status_section(all_rows: dict[str, list[Dict[str, Any]]]) -> list[str]:
    lines = ["## Run Status", "| suite | completed | failed | not run |", "| --- | ---: | ---: | ---: |"]
    for name, rows in all_rows.items():
        completed, failed, not_run = _status_counts(rows)
        lines.append(f"| {name} | {completed} | {failed} | {not_run} |")
    return lines


def _best_table(title: str, rows: list[Dict[str, Any]], metric_name: str = "best_metric") -> list[str]:
    lines = [f"## {title}", "| run_id | status | model | dataset | best | final | reason |", "| --- | --- | --- | --- | ---: | ---: | --- |"]
    best = _best(rows, metric_name)
    if best is None:
        lines.append("| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | no completed rows |")
        return lines
    lines.append(
        "| "
        + " | ".join(
            [
                str(best.get("run_id", "")),
                str(best.get("status", "")),
                str(best.get("model_name", "")),
                str(best.get("dataset_name", "")),
                _fmt(best.get("best_metric")),
                _fmt(best.get("final_metric")),
                str(best.get("reason", "")),
            ]
        )
        + " |"
    )
    return lines


def _mechanism_summary(rows: list[Dict[str, Any]]) -> list[str]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "COMPLETED":
            continue
        metrics = _metrics(row)
        grouped[(str(row.get("dataset_name", "")), str(row.get("model_name", "")))].append(_num(metrics.get("accuracy")))
    lines = [
        "## Mechanism Probe Results",
        "| probe | mechanism | mean success | n |",
        "| --- | --- | ---: | ---: |",
    ]
    if not grouped:
        lines.append("| MISSING | MISSING | MISSING | 0 |")
        return lines
    for (probe, mechanism), values in sorted(grouped.items()):
        valid = [value for value in values if not math.isnan(value)]
        mean = sum(valid) / len(valid) if valid else float("nan")
        lines.append(f"| {probe} | {mechanism} | {_fmt(mean)} | {len(values)} |")
    return lines


def _head_conclusion(rows: list[Dict[str, Any]]) -> str:
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    if not completed:
        return "MISSING: head isolation has no completed runs."
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in completed:
        metrics = _metrics(row)
        value = _num(metrics.get("test_accuracy", row.get("final_metric")))
        grouped[str(row.get("model_name", ""))].append(value)
    means = {name: sum(vals) / len(vals) for name, vals in grouped.items() if vals and not any(math.isnan(v) for v in vals)}
    if not means:
        return "MISSING: completed head rows did not contain accuracy metrics."
    best_name = max(means, key=means.get)
    best_value = means[best_name]
    eml_names = [name for name in means if name.startswith("eml")]
    best_eml = max((means[name], name) for name in eml_names)[1] if eml_names else "MISSING"
    best_eml_value = means.get(best_eml, float("nan"))
    if best_name.startswith("eml"):
        return (
            f"NOT PROVEN: current artifacts include `{best_name}` as the best mean smoke-scale head at {_fmt(best_value)}, "
            "but this is not sufficient to claim EML heads beat ordinary heads under matched medium/real-data conditions."
        )
    return f"NOT PROVEN: best mean head result is `{best_name}` at {_fmt(best_value)}; best EML head is `{best_eml}` at {_fmt(best_eml_value)}."


def _efficient_image_gate(rows: list[Dict[str, Any]]) -> tuple[float, str]:
    best = 0.0
    model = "MISSING"
    for row in rows:
        if row.get("status") != "COMPLETED":
            continue
        name = str(row.get("model_name", ""))
        if "efficient" not in name.lower():
            continue
        value = _num(row.get("best_metric"))
        if value > best:
            best = value
            model = name
    return best, model


def _raw_artifacts(all_rows: dict[str, list[Dict[str, Any]]]) -> list[str]:
    lines = ["## Raw Artifacts"]
    for report_name, report_path in REPORTS.items():
        status = "exists" if report_path.exists() else "MISSING"
        lines.append(f"- {report_name} report ({status}): `{report_path}`")
    for suite, rows in all_rows.items():
        lines.append(f"- {suite} summary: `{SUMMARY_ROOTS[suite] / 'summary.csv'}`")
        for row in rows:
            lines.append(f"  - `{row.get('run_id', '')}`: `{row.get('run_dir', '')}`")
    return lines


def _write_cnn_head_report(rows: list[Dict[str, Any]]) -> Path:
    output = REPORTS["cnn_head"]
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CNN Head End-to-End Report",
        "",
        "This report is generated from end-to-end head ablation run artifacts only.",
        "",
        "| run_id | status | model | loss mode | dataset | seed | test acc | test loss | reason |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    found = False
    for row in rows:
        if row.get("experiment_type") not in {"cnn_head_end_to_end", "end_to_end"}:
            continue
        found = True
        metrics = _metrics(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_id", "")),
                    str(row.get("status", "")),
                    str(row.get("model_name", "")),
                    str(row.get("loss_mode", "")),
                    str(row.get("dataset_name", "")),
                    str(row.get("seed", "")),
                    _fmt(metrics.get("test_accuracy")),
                    _fmt(metrics.get("test_loss")),
                    str(row.get("reason", "")),
                ]
            )
            + " |"
        )
    if not found:
        lines.append("| MISSING | NOT RUN | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | no end-to-end rows found |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def generate(output: str | Path) -> Path:
    all_rows = _all_rows()
    _write_cnn_head_report(all_rows["head"])
    image_gate, image_gate_model = _efficient_image_gate(all_rows["image"])
    head_statement = _head_conclusion(all_rows["head"])
    lines = [
        "# EML Master Next Step Report",
        "",
        "## Executive Summary",
        f"- Head isolation: {head_statement}",
        f"- Efficient image synthetic gate: `{image_gate_model}` reached `{image_gate:.4f}`; CIFAR medium remains gated below `0.8000`.",
        "- Mechanism conclusions should be read from probe success and diagnostics tables, not from synthetic task accuracy alone.",
        "- Representation trunk claims remain conditional until efficient image/text paths beat simple baselines under the same budgets.",
        "",
        "## What Is Proven",
        "- Completed runs in this report are backed by raw artifact directories listed below.",
        "- The available head-ablation data isolates ordinary heads and EML prototype heads on the same CNN feature source when those runs are present.",
        "- The synthetic mechanism probes exercise null responsibility, responsibility selection, precision updates, composition consistency, and attractor collapse checks.",
        "",
        "## What Is Not Proven",
        "- EML as a general backbone replacement is not proven by these artifacts.",
        "- If the efficient image synthetic gate is below `0.8`, CIFAR claims for that path should be skipped.",
        "- Any missing or failed rows are not interpreted as evidence.",
        "",
    ]
    lines.extend(_status_section(all_rows))
    lines.append("")
    lines.extend(_best_table("Head Isolation Results", all_rows["head"]))
    lines.append("")
    lines.extend(_mechanism_summary(all_rows["mechanism"]))
    lines.append("")
    lines.extend(_best_table("Image Representation Ablation", all_rows["image"]))
    lines.append("")
    lines.extend(_best_table("Text Representation Ablation", all_rows["text"]))
    lines.append("")
    lines.extend(_best_table("CIFAR Medium Status", all_rows["cifar"]))
    lines.extend(
        [
            "",
            "## EML Diagnostics",
            "Diagnostics are stored in each run's `diagnostics.csv` and `summary.json` under `final_diagnostics`. Key fields include drive, resistance, energy, null weight, update gate, ambiguity, and attractor diversity when a model exposes them.",
            "",
            "## Efficiency Analysis",
            "Per-step time, examples/sec, tokens/sec, parameter counts, and peak memory are stored in run metrics. This master report does not average incompatible task families.",
            "",
            "## Failure Modes",
            "- Mark all missing experiments as `NOT RUN` instead of interpreting them.",
            "- Treat all-null behavior, never-null behavior, excessive update gates, resistance collapse, and attractor collapse as diagnostic failures requiring targeted probe review.",
            "- Slow local-window implementations should be compared against CNN/local recurrent baselines before broad claims.",
            "",
            "## Recommended Next Experiment",
            "Run medium head isolation and end-to-end CNN+head ablations on CIFAR only after synthetic smoke remains stable, then compare EML centered ambiguity against cosine and MLP heads with bootstrap deltas.",
            "",
            "## Stop/Go Decisions",
            f"- EML as head: {head_statement}",
            "- EML as refinement: GO only as a controlled ablation against the same CNN feature extractor and losses.",
            "- EML as representation trunk: HOLD until synthetic image efficient path clears `0.8` and beats local baselines.",
            "- EML as foundation architecture: HOLD until representation trunk validation improves.",
            "",
        ]
    )
    lines.extend(_raw_artifacts(all_rows))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    report = generate(args.output)
    print(report)


if __name__ == "__main__":
    main()
