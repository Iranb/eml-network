from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return 0.0 if self.count == 0 else self.total / self.count


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def ensure_dir(path: str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        values = [move_batch_to_device(value, device) for value in batch]
        return type(batch)(values)
    return batch


def iter_batches(loader: torch.utils.data.DataLoader, max_batches: int | None = None) -> Iterator[Any]:
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch_index, batch


def compute_warmup_eta(global_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return max(0.0, min(1.0, global_step / warmup_steps))


def scalar(value: float | torch.Tensor) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return scalar((logits.argmax(dim=-1) == targets).float().mean())


def parse_slot_layout(layout_text: str) -> dict[str, int]:
    layout: dict[str, int] = {}
    for item in layout_text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, count = item.split(":", 1)
            layout[name.strip()] = int(count.strip())
        else:
            layout[item] = 1
    if not layout:
        raise ValueError("slot layout must not be empty")
    return layout


__all__ = [
    "AverageMeter",
    "classification_accuracy",
    "compute_warmup_eta",
    "ensure_dir",
    "iter_batches",
    "move_batch_to_device",
    "parse_slot_layout",
    "resolve_device",
    "save_json",
    "scalar",
    "set_seed",
]
