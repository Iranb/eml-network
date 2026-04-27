from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import run_eml_uncertainty_benchmark as benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate EML pluggable primitive report")
    parser.add_argument("--runs-root", default="reports/uncertainty_frozen/runs")
    parser.add_argument("--output", default="reports/EML_PLUGGABLE_PRIMITIVE_REPORT.md")
    parser.add_argument("--include-superseded", action="store_true")
    return parser


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return "MISSING"
    if not math.isfinite(number):
        return "MISSING"
    return f"{number:.{digits}f}"


def _load_rows(runs_root: Path) -> list[dict[str, Any]]:
    summary_path = runs_root / "summary.csv"
    if not summary_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metrics = {}
            raw = row.get("metrics_json")
            if raw:
                try:
                    metrics = json.loads(raw)
                except Exception:
                    metrics = {}
            run_dir = Path(row.get("run_dir", ""))
            summary_json = run_dir / "summary.json"
            if summary_json.exists():
                try:
                    metrics.update(json.loads(summary_json.read_text(encoding="utf-8")))
                except Exception:
                    pass
            merged = {**row, **metrics}
            rows.append(merged)
    return rows


def _latest_accepted_rows(rows: list[dict[str, Any]], include_superseded: bool) -> list[dict[str, Any]]:
    if include_superseded:
        return rows
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("dataset_name", "")), str(row.get("model_name", "")), str(row.get("seed", "")))
        grouped[key].append(row)
    accepted: list[dict[str, Any]] = []
    for entries in grouped.values():
        true_entries = [row for row in entries if str(row.get("early_stop_triggered", "")).lower() == "true"]
        accepted.append(true_entries[-1] if true_entries else entries[-1])
    return accepted


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    return float(sum(values) / len(values)) if values else float("nan")


def generate_report(runs_root: str | Path, output: str | Path, include_superseded: bool = False) -> Path:
    runs_root = Path(runs_root)
    output_path = Path(output)
    base_report = benchmark.generate_report(
        runs_root,
        output_path.with_name(f"{output_path.stem}_BASE{output_path.suffix}"),
    )
    rows = _latest_accepted_rows(_load_rows(runs_root), include_superseded=include_superseded)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "COMPLETED":
            grouped[(str(row.get("dataset_name", "")), str(row.get("model_name", "")))].append(row)

    lines = [
        "# EML as Pluggable Energy / Responsibility / Uncertainty Primitive",
        "",
        "## 1. Executive Summary",
        "",
        "This report focuses on EML as a plug-in uncertainty/resistance primitive on existing CNN features. It does not claim EML is a clean-accuracy backbone replacement.",
        "",
    ]
    if not rows:
        lines.append("- No completed runs were found. All claims are MISSING.")
    for dataset in sorted({key[0] for key in grouped}):
        cosine = grouped.get((dataset, "cosine_prototype"), [])
        eml = grouped.get((dataset, "eml_centered_ambiguity"), [])
        supervised = grouped.get((dataset, "eml_supervised_resistance"), [])
        if cosine and eml:
            lines.append(
                f"- `{dataset}` clean accuracy: EML centered {_fmt(_mean(eml, 'clean_accuracy'))} vs cosine {_fmt(_mean(cosine, 'clean_accuracy'))}; clean head advantage is not claimed unless EML is higher."
            )
            lines.append(
                f"- `{dataset}` calibration: EML centered ECE {_fmt(_mean(eml, 'clean_ece'))} vs cosine {_fmt(_mean(cosine, 'clean_ece'))}."
            )
            lines.append(
                f"- `{dataset}` selective prediction: EML centered AURC {_fmt(_mean(eml, 'clean_selective_aurc'))} vs cosine {_fmt(_mean(cosine, 'clean_selective_aurc'))}; lower is better."
            )
        if supervised:
            lines.append(
                f"- `{dataset}` supervised resistance correlations: noise {_fmt(_mean(supervised, 'pooled_resistance_noise_corr'))}, occlusion {_fmt(_mean(supervised, 'pooled_resistance_occlusion_corr'))}."
            )

    lines.extend(
        [
            "",
            "## 2. Current Claim Status",
            "",
            "| Claim | Status | Evidence |",
            "| --- | --- | --- |",
            "| EML as backbone | Not proven | Existing reports favor CNN baselines for image representation. |",
            "| EML as clean classification head | Not proven unless this report shows a matched win | Linear/MLP/cosine remain primary baselines. |",
            "| EML as uncertainty/risk primitive | Open, measured here | Use calibration, selective risk, AUROC, and resistance correlations, not clean top-1 alone. |",
            "| MERC as head/block | No-go currently | Included only as experimental comparison when present. |",
            "",
            "## 3. Datasets",
            "",
            "- Synthetic shape uncertainty: available through `SyntheticShapeEnergyDataset` corruptions.",
            "- CIFAR corruption wrapper: available when local CIFAR-10 and torchvision are available.",
            "- Text corruption: NOT RUN in this report.",
            "- Agent risk toy: NOT RUN unless `scripts/run_agent_risk_toy_benchmark.py` artifacts are supplied.",
            "",
            "## 4. Models",
            "",
            "- Baselines: `linear`, `mlp`, `cosine_prototype`.",
            "- EML heads: `eml_no_ambiguity`, `eml_centered_ambiguity`, `eml_supervised_resistance`.",
            "- MERC heads: `merc_linear`, `merc_energy` when present; experimental/no-go until evidence changes.",
            "",
            "## 5. Frozen Feature Results",
            "",
            "| dataset | model | seeds | clean acc | noisy acc | occluded acc | clean ECE | clean AURC | clean->noisy AUROC | clean->occluded AUROC | resistance-noise corr | resistance-occlusion corr |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for (dataset, model), bucket in sorted(grouped.items()):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                dataset,
                model,
                len(bucket),
                _fmt(_mean(bucket, "clean_accuracy")),
                _fmt(_mean(bucket, "noisy_accuracy")),
                _fmt(_mean(bucket, "occluded_accuracy")),
                _fmt(_mean(bucket, "clean_ece")),
                _fmt(_mean(bucket, "clean_selective_aurc")),
                _fmt(_mean(bucket, "clean_vs_noisy_auroc")),
                _fmt(_mean(bucket, "clean_vs_occluded_auroc")),
                _fmt(_mean(bucket, "pooled_resistance_noise_corr")),
                _fmt(_mean(bucket, "pooled_resistance_occlusion_corr")),
            )
        )

    lines.extend(
        [
            "",
            "## 6. End-to-End Results",
            "",
            "NOT RUN in this report. The current run isolates heads on frozen CNN features.",
            "",
            "## 7. Responsibility Plugin Results",
            "",
            "NOT RUN in this report unless separate responsibility-plugin artifacts are added.",
            "",
            "## 8. Agent Risk Toy Results",
            "",
            "NOT RUN in this report unless separate agent-risk artifacts are added.",
            "",
            "## 9. EML Diagnostics",
            "",
            "Drive/resistance/energy diagnostics are stored in each run directory's `diagnostics.csv`. Resistance correlations are summarized in the frozen-feature table when the head exposes resistance.",
            "",
            "## 10. Failure Modes",
            "",
            "- EML can improve calibration or selective risk while losing clean accuracy; this is not a clean-head win.",
            "- Resistance correlations may be weak, negative, or MISSING for non-EML heads.",
            "- Repeated run IDs can appear when capped rows are rerun. Accepted summaries prefer the latest early-stopped replacement.",
            "",
            "## 11. Conclusions",
            "",
            "A. Does EML beat established backbone families? No claim; not tested here.",
            "",
            "B. Does EML beat ordinary heads on clean classification? Only if the table shows a matched clean-accuracy win; otherwise not supported.",
            "",
            "C. Does EML help uncertainty/corruption/selective prediction? Use ECE, AURC, AUROC, and resistance correlations above; do not infer this from clean accuracy.",
            "",
            "D. Does EML help action/risk decision? NOT RUN in this report.",
            "",
            "E. Next experiment: run the same early-stop discipline for end-to-end uncertainty and responsibility-plugin artifacts before promoting any claim.",
            "",
            "## Raw Artifacts",
            "",
            f"- Runs root: `{runs_root}`",
            f"- Benchmark report generated by base runner: `{base_report}`",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    report_path = generate_report(args.runs_root, args.output, include_superseded=args.include_superseded)
    print(json.dumps({"report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
