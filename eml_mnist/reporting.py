from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_summary(summary_path: Path) -> List[Dict[str, str]]:
    if not summary_path.exists():
        return []
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(row: Dict[str, Any], key: str) -> float | None:
    value = row.get(key, "")
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def _metrics(row: Dict[str, str]) -> Dict[str, Any]:
    value = row.get("metrics_json", "")
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _status_table(rows: Iterable[Dict[str, str]]) -> list[str]:
    lines = [
        "| run_id | status | task | model | dataset | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row.get("run_id", ""),
                row.get("status", ""),
                row.get("task_name", ""),
                row.get("model_name", ""),
                row.get("dataset_name", ""),
                row.get("reason", ""),
            )
        )
    return lines


def _result_table(rows: Iterable[Dict[str, str]], task_filter: str) -> list[str]:
    selected = [row for row in rows if row.get("status") == "COMPLETED" and task_filter in row.get("task_name", "")]
    lines = [
        "| run_id | model | dataset | final metric | best metric | loss | accuracy | time sec | params |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not selected:
        lines.append("| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
        return lines
    for row in selected:
        metrics = _metrics(row)
        loss = metrics.get("final_train_loss", metrics.get("train_loss", ""))
        acc = metrics.get("final_train_accuracy", metrics.get("train_accuracy", metrics.get("token_accuracy", "")))
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("run_id", ""),
                row.get("model_name", ""),
                row.get("dataset_name", ""),
                row.get("final_metric", ""),
                row.get("best_metric", ""),
                f"{loss:.4f}" if isinstance(loss, (int, float)) else loss,
                f"{acc:.4f}" if isinstance(acc, (int, float)) else acc,
                row.get("total_train_time_sec", ""),
                row.get("num_params", ""),
            )
        )
    return lines


def _efficiency_table(rows: Iterable[Dict[str, str]]) -> list[str]:
    lines = [
        "| run_id | model | task | examples/sec | tokens/sec | step time | peak memory MB | params |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    if not completed:
        lines.append("| MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
        return lines
    for row in completed:
        metrics = _metrics(row)
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("run_id", ""),
                row.get("model_name", ""),
                row.get("task_name", ""),
                metrics.get("examples_per_sec", ""),
                metrics.get("tokens_per_sec", ""),
                metrics.get("step_time_sec", ""),
                metrics.get("peak_memory_mb", ""),
                row.get("num_params", ""),
            )
        )
    return lines


def _diagnostics_table(rows: Iterable[Dict[str, str]]) -> list[str]:
    keys = [
        "drive_mean",
        "drive_std",
        "resistance_mean",
        "resistance_std",
        "energy_mean",
        "energy_std",
        "null_weight_mean",
        "responsibility_entropy_mean",
        "update_strength_mean",
        "update_gate_mean",
        "attractor_diversity",
        "ambiguity_mean",
        "ambiguity_weight_mean",
        "sample_uncertainty_mean",
    ]
    metric_keys = [
        "resistance_noise_corr",
        "resistance_occlusion_corr",
        "corruption_resistance_corr",
    ]
    lines = ["| run_id | model | " + " | ".join(keys + metric_keys) + " |"]
    lines.append("| --- | --- | " + " | ".join(["---:"] * (len(keys) + len(metric_keys))) + " |")
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    if not completed:
        lines.append("| MISSING | MISSING | " + " | ".join(["MISSING"] * len(keys)) + " |")
        return lines
    for row in completed:
        summary = _read_json(Path(row.get("run_dir", "")) / "summary.json")
        diagnostics = summary.get("final_diagnostics", {})
        metrics = _metrics(row)
        values = []
        for key in keys:
            value = diagnostics.get(key, "")
            values.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
        for key in metric_keys:
            value = metrics.get(key, "")
            values.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
        lines.append("| {} | {} | {} |".format(row.get("run_id", ""), row.get("model_name", ""), " | ".join(values)))
    return lines


def _ablation_table(rows: Iterable[Dict[str, str]]) -> list[str]:
    selected = [row for row in rows if row.get("mode") == "ablation" or row.get("run_id", "").startswith("ablation_")]
    lines = [
        "| run_id | status | task | model | key settings | best | final | loss | reason |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    if not selected:
        return ["MISSING"]
    setting_keys = [
        "responsibility_mode",
        "use_null",
        "update_mode",
        "warmup_enabled",
        "early_stop",
        "patience",
        "num_attractors",
        "local_window_size",
        "seq_len",
    ]
    for row in selected:
        config = _read_json(Path(row.get("run_dir", "")) / "config.json")
        metrics = _metrics(row)
        settings = []
        for key in setting_keys:
            if key in config:
                settings.append(f"{key}={config[key]}")
        if not settings:
            settings.append("see config")
        loss = metrics.get("final_train_loss", metrics.get("train_loss", ""))
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("run_id", ""),
                row.get("status", ""),
                row.get("task_name", ""),
                row.get("model_name", ""),
                ", ".join(settings),
                row.get("best_metric", ""),
                row.get("final_metric", ""),
                f"{loss:.4f}" if isinstance(loss, (int, float)) else loss,
                row.get("reason", ""),
            )
        )
    return lines


def _comparison_table(
    rows: Iterable[Dict[str, str]],
    predicate: Any,
    reference_run_id: str | None = None,
) -> list[str]:
    selected = [row for row in rows if predicate(row)]
    lines = [
        "| run_id | status | model | best | final | delta vs ref | loss | time sec | notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if not selected:
        return ["MISSING"]
    reference_value: float | None = None
    if reference_run_id:
        for row in selected:
            if row.get("run_id") == reference_run_id:
                reference_value = _float(row, "best_metric")
                break
    if reference_value is None:
        completed = [row for row in selected if row.get("status") == "COMPLETED" and _float(row, "best_metric") is not None]
        if completed:
            reference_value = _float(completed[0], "best_metric")
    for row in selected:
        metrics = _metrics(row)
        best = _float(row, "best_metric")
        delta = "" if best is None or reference_value is None else best - reference_value
        loss = metrics.get("final_train_loss", metrics.get("train_loss", ""))
        notes = row.get("reason", "")
        if row.get("status") == "COMPLETED":
            early = metrics.get("early_stop_triggered", "")
            if early != "":
                notes = f"early_stop={early}"
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("run_id", ""),
                row.get("status", ""),
                row.get("model_name", ""),
                row.get("best_metric", ""),
                row.get("final_metric", ""),
                f"{delta:.4f}" if isinstance(delta, float) else delta,
                f"{loss:.4f}" if isinstance(loss, (int, float)) else loss,
                row.get("total_train_time_sec", ""),
                notes,
            )
        )
    return lines


def _history_snippets(rows: Iterable[Dict[str, str]], limit: int = 5) -> list[str]:
    lines: list[str] = []
    for row in rows:
        run_dir = Path(row.get("run_dir", ""))
        history = _read_json(run_dir / "history.json")
        if not isinstance(history, list) or not history:
            continue
        lines.append(f"### {row.get('run_id', '')}")
        lines.append("")
        lines.append("| step | train loss | train acc | token acc | wall time |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for item in history[-limit:]:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    item.get("step", ""),
                    item.get("train_loss", item.get("next_token_loss", "")),
                    item.get("train_accuracy", ""),
                    item.get("token_accuracy", ""),
                    item.get("wall_clock_time_sec", ""),
                )
            )
        lines.append("")
    if not lines:
        return ["MISSING"]
    return lines


def _best(rows: list[Dict[str, str]], task_token: str) -> str:
    candidates = [row for row in rows if row.get("status") == "COMPLETED" and task_token in row.get("task_name", "")]
    if not candidates:
        return "MISSING"
    candidates.sort(key=lambda row: _float(row, "best_metric") if _float(row, "best_metric") is not None else -1.0, reverse=True)
    top = candidates[0]
    return f"{top.get('model_name', '')} ({top.get('best_metric', 'MISSING')})"


def _strongest_baseline(rows: list[Dict[str, str]]) -> str:
    baseline_names = {"cnn_eml", "pure_eml", "pure_eml_v2", "cnn_eml_workers0", "pure_eml_workers0"}
    candidates = [
        row
        for row in rows
        if row.get("status") == "COMPLETED"
        and row.get("model_name") in baseline_names
        and "image" in row.get("task_name", "")
    ]
    if not candidates:
        return "MISSING"
    candidates.sort(key=lambda row: _float(row, "best_metric") if _float(row, "best_metric") is not None else -1.0, reverse=True)
    top = candidates[0]
    return f"{top.get('model_name', '')} ({top.get('best_metric', 'MISSING')})"


def generate_validation_report(
    runs_root: str | Path = "reports/runs",
    output_path: str | Path = "reports/EML_VALIDATION_REPORT.md",
) -> Path:
    runs_root = Path(runs_root)
    output_path = Path(output_path)
    rows = _read_summary(runs_root / "summary.csv")
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    not_run = [row for row in rows if row.get("status") == "NOT RUN"]
    failed = [row for row in rows if row.get("status") == "FAILED"]
    by_env: Dict[str, list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_env[row.get("hostname", "unknown")].append(row)

    lines: list[str] = []
    lines.append("# EML Validation and Ablation Report")
    lines.append("")
    lines.append("Generated: " + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append("- Best image result: " + _best(rows, "image"))
    lines.append("- Best text result: " + _best(rows, "text"))
    lines.append("- Strongest baseline: " + _strongest_baseline(rows))
    lines.append("- Responsibility evidence weighting: see mechanism probe and downstream ablation tables.")
    lines.append("- Precision update: see update probe rows and text/image ablation cells; model-quality conclusions need longer runs.")
    if any(row.get("run_id") == "ablation_no_attractor" and row.get("status") == "COMPLETED" for row in rows):
        lines.append("- Attractor memory: direct no-attractor comparison is present in this report.")
    else:
        lines.append("- Attractor memory: no-attractor comparison is MISSING unless a completed row exists below.")
    lines.append(f"- Major failure modes: {len(failed)} failed runs and {len(not_run)} not-run cells are recorded in the status table.")
    if not_run:
        lines.append("- Recommended next step: standardize the remaining NOT RUN switches, then repeat the best image/text runs across seeds.")
    else:
        lines.append("- Recommended next step: rerun the most promising synthetic cells for more steps/seeds, then repeat CIFAR and text-medium validation.")
    lines.append("")
    lines.append("## 2. Repository and Environment")
    lines.append("")
    if rows:
        env_row = rows[-1]
        env_keys = ["git_commit", "hostname", "python_version", "torch_version", "torchvision_version", "cuda_available", "device", "timestamp"]
        for key in env_keys:
            lines.append(f"- {key}: {env_row.get(key, 'MISSING')}")
    else:
        lines.append("MISSING: no run summary found.")
    lines.append("")
    lines.append("## 3. Experimental Scope")
    lines.append("")
    lines.extend(_status_table(rows))
    lines.append("")
    lines.append("Failed runs: " + str(len(failed)))
    lines.append("Not-run entries: " + str(len(not_run)))
    lines.append("")
    lines.append("## 4. Datasets")
    lines.append("")
    dataset_names = sorted({row.get("dataset_name", "MISSING") for row in rows if row.get("dataset_name")})
    if not dataset_names:
        lines.append("MISSING")
    else:
        lines.append("| dataset | synthetic/real | notes |")
        lines.append("| --- | --- | --- |")
        for name in dataset_names:
            kind = "synthetic" if "synthetic" in name.lower() else "real/optional"
            notes = "offline" if kind == "synthetic" else "requires local data/dependency"
            lines.append(f"| {name} | {kind} | {notes} |")
    lines.append("")
    lines.append("## 5. Models Compared")
    lines.append("")
    lines.append("| model | parameter count | task names | key mechanisms |")
    lines.append("| --- | ---: | --- | --- |")
    model_rows: Dict[str, list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("model_name"):
            model_rows[row["model_name"]].append(row)
    if not model_rows:
        lines.append("| MISSING | MISSING | MISSING | MISSING |")
    else:
        for model, items in sorted(model_rows.items()):
            params = next((item.get("num_params", "") for item in items if item.get("num_params")), "")
            tasks = ", ".join(sorted({item.get("task_name", "") for item in items if item.get("task_name")}))
            mechanisms = "see config artifacts"
            lines.append(f"| {model} | {params} | {tasks} | {mechanisms} |")
    lines.append("")
    lines.append("## 6. Main Results")
    lines.append("")
    lines.append("### Image")
    lines.extend(_result_table(rows, "image"))
    lines.append("")
    lines.append("### Text")
    lines.extend(_result_table(rows, "text"))
    lines.append("")
    lines.append("### Efficiency")
    lines.extend(_efficiency_table(rows))
    lines.append("")
    lines.append("### Stability")
    lines.append("NaN/Inf counts are recorded when runners emit `nan_inf_count`; otherwise MISSING.")
    lines.append("")
    lines.append("## 7. Ablation Results")
    lines.append("")
    lines.append("### Responsibility / Null / Update Probes")
    lines.extend(
        _comparison_table(
            rows,
            lambda row: row.get("task_name") == "mechanism_probe",
            "ablation_gate_sigmoid_seed0",
        )
    )
    lines.append("")
    lines.append("Interpretation: these probes validate finite propagation and diagnostic behavior. They do not by themselves prove downstream task quality.")
    lines.append("")
    lines.append("### Image Representation / Attractor / Warmup / Window")
    lines.extend(
        _comparison_table(
            rows,
            lambda row: row.get("task_name") == "image_synthetic",
            "ablation_image_cnn_eml_workers0",
        )
    )
    lines.append("")
    lines.append("### Text Local Window")
    lines.extend(
        _comparison_table(
            rows,
            lambda row: row.get("task_name") == "text_synthetic",
            "ablation_text_window8_baseline",
        )
    )
    lines.append("")
    lines.append("### CIFAR Medium")
    lines.extend(
        _comparison_table(
            rows,
            lambda row: row.get("task_name") == "image_cifar",
            "cifar_cnn_eml",
        )
    )
    lines.append("")
    lines.append("### Failed And Not Run Cells")
    lines.extend(_status_table([row for row in rows if row.get("status") != "COMPLETED"]))
    lines.append("")
    lines.append("### All Ablation Cells")
    lines.extend(_ablation_table(rows))
    lines.append("")
    if not_run:
        lines.append("Other ablation axes remain `NOT RUN` when listed in the status table.")
    else:
        lines.append("No planned ablation cells in this run were recorded as `NOT RUN`.")
    lines.append("")
    lines.append("## 8. EML Diagnostics")
    lines.append("")
    lines.extend(_diagnostics_table(rows))
    lines.append("")
    lines.append("Resistance-noise, resistance-occlusion, and resistance-corruption correlations are included when emitted by a run; otherwise MISSING.")
    lines.append("")
    lines.append("## 9. Training Curves")
    lines.append("")
    lines.extend(_history_snippets(completed))
    lines.append("")
    lines.append("## 10. Efficiency Analysis")
    lines.append("")
    lines.append("- Runtime and throughput are available in per-run summaries and the efficiency table.")
    lines.append("- Local-window cost and attractor count are recorded when model diagnostics expose them.")
    lines.append("- Short smoke runs are not enough to decide whether accuracy gain justifies added cost.")
    lines.append("")
    lines.append("## 11. Failure Modes")
    lines.append("")
    lines.append("- gate collapse: MISSING unless gate diagnostics are emitted.")
    lines.append("- all-null collapse: inspect `null_weight_mean`; high values indicate risk.")
    lines.append("- never-null collapse: inspect `null_weight_mean`; near zero indicates risk.")
    lines.append("- energy explosion: inspect `energy_mean/std` and NaN/Inf counts.")
    lines.append("- resistance collapse: inspect `resistance_mean/std`.")
    lines.append("- attractor collapse: inspect `attractor_diversity`.")
    lines.append("- update gate too high at init: inspect `update_gate_mean`.")
    lines.append("- poor causal text behavior: no-leak tests exist; training report includes only available run metrics.")
    lines.append("- slow local-window implementation: compare seconds and throughput.")
    lines.append("")
    lines.append("## 12. Conclusions")
    lines.append("")
    lines.append("- Current evidence remains preliminary when runs are short or single-seed.")
    lines.append("- Use the synthetic image and text ablation tables to identify mechanisms worth longer training before making CIFAR claims.")
    if any(row.get("task_name") == "image_cifar" for row in rows):
        lines.append("- The exact next experiment is repeat-seed CIFAR validation for the best efficient image setting and the strongest CNN baseline.")
    else:
        lines.append("- The exact next experiment is longer GPU ablation with the completed switches, followed by CIFAR medium for `cnn_eml`, `pure_eml_v2`, and `EfficientEMLImageClassifier`.")
    lines.append("")
    lines.append("## 13. Raw Artifacts")
    lines.append("")
    for row in rows:
        run_dir = row.get("run_dir", "")
        lines.append(f"- {row.get('run_id', '')}: {run_dir}")
        if run_dir:
            run_path = Path(run_dir)
            for label, filename in [
                ("history", "history.json"),
                ("metrics", "metrics.csv"),
                ("diagnostics", "diagnostics.csv"),
                ("summary", "summary.json"),
            ]:
                artifact_path = run_path / filename
                lines.append(f"  - {label}: {artifact_path if artifact_path.exists() else 'MISSING'}")
    lines.append("")
    lines.append("## 14. Appendix: Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("pytest")
    lines.append("python scripts/run_eml_validation_suite.py --mode smoke --device cpu")
    lines.append("python scripts/generate_eml_report.py")
    lines.append("python scripts/run_eml_validation_suite.py --mode ablation --device cuda")
    lines.append("python scripts/run_eml_validation_suite.py --mode cifar-medium --device cuda")
    lines.append("python scripts/run_eml_validation_suite.py --mode text-medium --device cuda")
    lines.append("```")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


__all__ = ["generate_validation_report"]
