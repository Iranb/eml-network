from __future__ import annotations

import csv
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch


SUMMARY_FIELDS = [
    "run_id",
    "timestamp",
    "status",
    "reason",
    "mode",
    "task_name",
    "model_name",
    "dataset_name",
    "seed",
    "device",
    "git_commit",
    "hostname",
    "python_version",
    "torch_version",
    "torchvision_version",
    "cuda_available",
    "num_params",
    "trainable_params",
    "total_train_time_sec",
    "best_metric",
    "final_metric",
    "run_dir",
    "metrics_json",
]


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return float(detached.item())
        return detached.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True), encoding="utf-8")


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_row = {str(key): json_safe(value) for key, value in row.items()}
    rows: list[dict[str, Any]] = []
    fields: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [dict(item) for item in reader]
    for key in safe_row:
        if key not in fields:
            fields.append(key)
    rows.append({key: safe_row.get(key, "") for key in fields})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({key: item.get(key, "") for key in fields})


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"num_params": int(total), "trainable_params": int(trainable)}


def safe_torchvision_available() -> tuple[bool, str]:
    try:
        import torchvision  # type: ignore

        return True, str(getattr(torchvision, "__version__", "unknown"))
    except Exception as exc:  # pragma: no cover - optional dependency.
        return False, repr(exc)


def current_git_commit(cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def environment_metadata(device: str | torch.device, seed: int, cwd: Path | None = None) -> Dict[str, Any]:
    torchvision_ok, torchvision_version = safe_torchvision_available()
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": current_git_commit(cwd),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version if torchvision_ok else "unavailable",
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "seed": seed,
    }


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    values = []
    for parameter in parameters:
        if parameter.grad is not None:
            values.append(parameter.grad.detach().float().norm())
    if not values:
        return 0.0
    return float(torch.stack(values).norm().cpu().item())


class ExperimentLogger:
    def __init__(
        self,
        run_id: str,
        config: Mapping[str, Any],
        root: str | Path = "reports/runs",
    ) -> None:
        self.root = Path(root)
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.run_dir = self.root / f"{self.timestamp}_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)
        self.history: list[dict[str, Any]] = []
        self.artifacts: dict[str, Any] = {}
        self.stdout_path = self.run_dir / "stdout.log"
        write_json(self.run_dir / "config.json", self.config)
        write_json(self.run_dir / "history.json", self.history)
        write_json(self.run_dir / "artifacts.json", self.artifacts)

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.csv"

    def log_text(self, message: str) -> None:
        with self.stdout_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def set_model_info(self, model: torch.nn.Module | None = None, extra: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        info: Dict[str, Any] = dict(extra or {})
        if model is not None:
            info.update(count_parameters(model))
            info["model_class"] = model.__class__.__name__
        write_json(self.run_dir / "model_info.json", info)
        return info

    def log_step(self, metrics: Mapping[str, Any], diagnostics: Mapping[str, Any] | None = None) -> None:
        metric_row = dict(metrics)
        diagnostic_row = dict(diagnostics or {})
        self.history.append({**metric_row, **{f"diag_{key}": value for key, value in diagnostic_row.items()}})
        append_csv(self.run_dir / "metrics.csv", metric_row)
        append_csv(self.run_dir / "diagnostics.csv", diagnostic_row)
        write_json(self.run_dir / "history.json", self.history)

    def add_artifact(self, key: str, value: Any) -> None:
        self.artifacts[key] = value
        write_json(self.run_dir / "artifacts.json", self.artifacts)

    def finalize(
        self,
        summary: Mapping[str, Any],
        status: str = "COMPLETED",
        reason: str = "",
        model_info: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        env = environment_metadata(self.config.get("device", "unknown"), int(self.config.get("seed", 0)))
        info = dict(model_info or {})
        if not info and (self.run_dir / "model_info.json").exists():
            try:
                info = json.loads((self.run_dir / "model_info.json").read_text(encoding="utf-8"))
            except Exception:
                info = {}
        payload = {
            **env,
            **self.config,
            **info,
            **dict(summary),
            "run_id": self.run_id,
            "status": status,
            "reason": reason,
            "run_dir": str(self.run_dir),
        }
        payload["metrics_json"] = json.dumps(json_safe(dict(summary)), sort_keys=True)
        write_json(self.run_dir / "summary.json", payload)
        write_json(self.run_dir / "artifacts.json", self.artifacts)
        append_csv(self.summary_path, {key: payload.get(key, "") for key in SUMMARY_FIELDS})
        return payload

    @classmethod
    def not_run(
        cls,
        run_id: str,
        config: Mapping[str, Any],
        reason: str,
        root: str | Path = "reports/runs",
    ) -> Dict[str, Any]:
        logger = cls(run_id=run_id, config=config, root=root)
        logger.set_model_info(extra={"num_params": 0, "trainable_params": 0})
        logger.log_text(f"NOT RUN: {reason}")
        return logger.finalize(summary={}, status="NOT RUN", reason=reason)


__all__ = [
    "ExperimentLogger",
    "append_csv",
    "count_parameters",
    "current_git_commit",
    "environment_metadata",
    "grad_norm",
    "safe_torchvision_available",
    "write_json",
]
