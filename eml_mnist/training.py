import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


DATASET_SPECS: Dict[str, Dict[str, int]] = {
    "mnist": {
        "num_classes": 10,
        "image_size": 28,
        "input_channels": 1,
        "default_patch_size": 7,
    },
    "cifar10": {
        "num_classes": 10,
        "image_size": 32,
        "input_channels": 3,
        "default_patch_size": 4,
    },
}


class OptionalDatasetDependencyError(RuntimeError):
    pass


def _load_torchvision() -> Any:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - depends on optional binary compatibility.
        raise OptionalDatasetDependencyError(
            "MNIST/CIFAR loaders require a working torchvision installation"
        ) from exc
    return datasets, transforms


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
        if self.count == 0:
            return 0.0
        return self.total / self.count


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


def get_dataset_spec(dataset_name: str) -> Dict[str, int]:
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DATASET_SPECS[dataset_name]


def build_classification_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    datasets, transforms = _load_torchvision()

    if dataset_name == "mnist":
        train_transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = datasets.MNIST(root=data_dir, train=True, transform=train_transform, download=download)
        test_dataset = datasets.MNIST(root=data_dir, train=False, transform=test_transform, download=download)
    elif dataset_name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=download)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=download)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


def compute_warmup_eta(
    epoch_index: int,
    step_index: int,
    steps_per_epoch: int,
    warmup_epochs: int,
) -> float:
    if warmup_epochs <= 0:
        return 1.0
    progress = (epoch_index + step_index / max(1, steps_per_epoch)) / warmup_epochs
    return max(0.0, min(1.0, progress))


def compute_entropy_weight(
    base_weight: float,
    epoch_index: int,
    total_epochs: int,
    decay_epochs: int,
) -> float:
    if base_weight <= 0.0:
        return 0.0
    if decay_epochs <= 0:
        return 0.0
    progress = min(1.0, epoch_index / max(1, decay_epochs))
    return base_weight * (1.0 - progress)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def compute_loss_bundle(
    outputs: Dict[str, Any],
    targets: torch.Tensor,
    label_smoothing: float,
    pairwise_weight: float,
    resistance_weight: float,
    energy_weight: float,
    entropy_weight: float,
    prototype_diversity_weight: float,
    pairwise_margin: float,
    resistance_margin: float,
    energy_margin: float,
    activation_budget_weight: float = 0.0,
    activation_budget_target: float = 0.35,
) -> Dict[str, torch.Tensor]:
    logits = outputs["logits"]
    drive = outputs["drive"]
    resistance = outputs["resistance"]
    probs = outputs["probs"]
    batch_size, num_classes = logits.shape

    ce = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)

    target_mask = F.one_hot(targets, num_classes=num_classes).bool()
    positive_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_logits = logits.masked_fill(target_mask, float("-inf"))
    hardest_negative, hard_neg_indices = negative_logits.max(dim=1)
    pairwise = F.softplus(hardest_negative - positive_logits + pairwise_margin).mean()

    target_resistance = resistance.gather(1, targets.unsqueeze(1)).squeeze(1)
    hard_neg_resistance = resistance.gather(1, hard_neg_indices.unsqueeze(1)).squeeze(1)
    resistance_loss = F.softplus(target_resistance - hard_neg_resistance + resistance_margin).mean()

    energy_loss = F.relu(logits.abs() - energy_margin).square().mean()
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
    prototype_diversity = logits.new_tensor(0.0)
    prototypes = outputs.get("prototypes")
    if torch.is_tensor(prototypes) and prototypes.size(0) > 1:
        normalized_prototypes = F.normalize(prototypes, dim=-1)
        cosine = normalized_prototypes @ normalized_prototypes.t()
        off_diagonal = cosine.masked_select(~torch.eye(cosine.size(0), device=cosine.device, dtype=torch.bool))
        prototype_diversity = off_diagonal.square().mean()

    gate_values = []
    for block in outputs.get("block_stats", []):
        if "gate" in block:
            gate_values.append(block["gate"].mean())
    mean_gate = torch.stack(gate_values).mean() if gate_values else logits.new_tensor(0.0)
    activation_budget = F.relu(mean_gate - activation_budget_target).square()

    total = ce
    total = total + pairwise_weight * pairwise
    total = total + resistance_weight * resistance_loss
    total = total + energy_weight * energy_loss
    total = total + prototype_diversity_weight * prototype_diversity
    total = total + activation_budget_weight * activation_budget
    total = total - entropy_weight * entropy

    positive_drive = drive.gather(1, targets.unsqueeze(1)).squeeze(1)
    hard_neg_drive = drive.gather(1, hard_neg_indices.unsqueeze(1)).squeeze(1)
    class_radius = outputs.get("class_radius")
    class_radius_mean = class_radius.mean() if torch.is_tensor(class_radius) else logits.new_tensor(0.0)
    eml_gamma = outputs.get("eml_gamma")
    eml_lambda = outputs.get("eml_lambda")

    return {
        "loss": total,
        "ce": ce.detach(),
        "pairwise": pairwise.detach(),
        "resistance": resistance_loss.detach(),
        "energy": energy_loss.detach(),
        "prototype_diversity": prototype_diversity.detach(),
        "activation_budget": activation_budget.detach(),
        "entropy": entropy.detach(),
        "acc": logits.new_tensor(accuracy(logits, targets)),
        "mean_gate": mean_gate.detach(),
        "sample_uncertainty_mean": outputs["sample_uncertainty"].mean().detach(),
        "mean_uncertainty": outputs["sample_uncertainty"].mean().detach(),
        "drive_pos_mean": positive_drive.mean().detach(),
        "drive_hard_neg_mean": hard_neg_drive.mean().detach(),
        "resistance_pos_mean": target_resistance.mean().detach(),
        "resistance_hard_neg_mean": hard_neg_resistance.mean().detach(),
        "energy_pos_mean": positive_logits.mean().detach(),
        "energy_hard_neg_mean": hardest_negative.mean().detach(),
        "margin_mean": (positive_logits - hardest_negative).mean().detach(),
        "class_radius_mean": class_radius_mean.detach(),
        "eml_gamma": eml_gamma.detach() if torch.is_tensor(eml_gamma) else logits.new_tensor(0.0),
        "eml_lambda": eml_lambda.detach() if torch.is_tensor(eml_lambda) else logits.new_tensor(0.0),
    }


def move_batch_to_device(batch: Iterable[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    images, targets = batch
    return images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
