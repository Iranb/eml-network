from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eml_mnist.experiment_utils import safe_torchvision_available, write_json
from eml_mnist.image_datasets import SyntheticShapeEnergyDataset, SyntheticShapeEvidenceDataset
from eml_mnist.model import ConvBackbone
from eml_mnist.training import OptionalDatasetDependencyError, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frozen CNN features for head ablations")
    parser.add_argument("--dataset", choices=["synthetic_shape", "synthetic_evidence", "cifar10"], default="synthetic_shape")
    parser.add_argument("--output-dir", default="reports/head_ablation/features/synthetic_shape_seed0")
    parser.add_argument("--data-dir", default="~/dataset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _set_worker_strategy(num_workers: int) -> None:
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _cycle(loader: DataLoader) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def _batch_to_image_label(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["image"].to(device), batch["label"].to(device).long()
    image, label = batch
    return image.to(device), label.to(device).long()


def _metadata_tensor(batch: Any, key: str, batch_size: int) -> torch.Tensor | None:
    if isinstance(batch, dict) and key in batch:
        value = batch[key]
        if torch.is_tensor(value):
            return value.detach().float().cpu()
    return None


def _build_synthetic(args: argparse.Namespace | SimpleNamespace) -> tuple[Dataset, Dataset, Dataset, int, int]:
    train = SyntheticShapeEnergyDataset(
        size=args.train_size,
        image_size=args.image_size,
        seed=args.seed,
        target_type="shape",
    )
    val = SyntheticShapeEnergyDataset(
        size=args.val_size,
        image_size=args.image_size,
        seed=args.seed + 100_000,
        target_type="shape",
    )
    test = SyntheticShapeEnergyDataset(
        size=args.test_size,
        image_size=args.image_size,
        seed=args.seed + 200_000,
        target_type="shape",
    )
    return train, val, test, 3, train.num_classes


def _build_synthetic_evidence(args: argparse.Namespace | SimpleNamespace) -> tuple[Dataset, Dataset, Dataset, int, int]:
    train = SyntheticShapeEvidenceDataset(size=args.train_size, image_size=args.image_size, seed=args.seed)
    val = SyntheticShapeEvidenceDataset(size=args.val_size, image_size=args.image_size, seed=args.seed + 100_000)
    test = SyntheticShapeEvidenceDataset(size=args.test_size, image_size=args.image_size, seed=args.seed + 200_000)
    return train, val, test, 3, 2


def _build_cifar(args: argparse.Namespace | SimpleNamespace) -> tuple[Dataset, Dataset, Dataset, int, int]:
    ok, _version = safe_torchvision_available()
    if not ok:
        raise OptionalDatasetDependencyError("CIFAR-10 requires a working torchvision installation")
    from torchvision import datasets, transforms  # type: ignore

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    data_dir = str(Path(args.data_dir).expanduser())
    train_full = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=args.allow_download)
    test_full = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=args.allow_download)
    generator = torch.Generator().manual_seed(args.seed)
    train_size = min(args.train_size, len(train_full))
    remaining = len(train_full) - train_size
    train_subset, _unused = random_split(train_full, [train_size, remaining], generator=generator)
    val_size = min(args.val_size, len(test_full))
    test_size = min(args.test_size, max(0, len(test_full) - val_size))
    indices = torch.randperm(len(test_full), generator=generator).tolist()
    val_subset = Subset(test_full, indices[:val_size])
    test_subset = Subset(test_full, indices[val_size : val_size + test_size])
    return train_subset, val_subset, test_subset, 3, 10


def _build_datasets(args: argparse.Namespace | SimpleNamespace) -> tuple[Dataset, Dataset, Dataset, int, int]:
    if args.dataset == "synthetic_shape":
        return _build_synthetic(args)
    if args.dataset == "synthetic_evidence":
        return _build_synthetic_evidence(args)
    if args.dataset == "cifar10":
        return _build_cifar(args)
    raise ValueError(f"unsupported dataset: {args.dataset}")


def _loader(dataset: Dataset, args: argparse.Namespace | SimpleNamespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )


def _train_backbone(
    backbone: ConvBackbone,
    train_loader: DataLoader,
    num_classes: int,
    args: argparse.Namespace | SimpleNamespace,
    device: torch.device,
) -> Dict[str, float]:
    classifier = nn.Linear(args.feature_dim, num_classes).to(device)
    optimizer = AdamW(list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr)
    iterator = _cycle(train_loader)
    last_loss = 0.0
    last_acc = 0.0
    if args.steps <= 0:
        return {"feature_pretrain_loss": 0.0, "feature_pretrain_accuracy": 0.0}
    backbone.train()
    classifier.train()
    for _step in range(args.steps):
        batch = next(iterator)
        image, label = _batch_to_image_label(batch, device)
        logits = classifier(backbone(image))
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        last_acc = float((logits.argmax(dim=-1) == label).float().mean().detach().cpu().item())
    return {"feature_pretrain_loss": last_loss, "feature_pretrain_accuracy": last_acc}


@torch.no_grad()
def _extract_split(backbone: ConvBackbone, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    backbone.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    noise: list[torch.Tensor] = []
    occlusion: list[torch.Tensor] = []
    resistance_target: list[torch.Tensor] = []
    evidence_target: list[torch.Tensor] = []
    for batch in loader:
        image, label = _batch_to_image_label(batch, device)
        features.append(backbone(image).detach().cpu())
        labels.append(label.detach().cpu())
        noise_value = _metadata_tensor(batch, "noise_level", image.size(0))
        occlusion_value = _metadata_tensor(batch, "occlusion_level", image.size(0))
        resistance_value = _metadata_tensor(batch, "resistance_target", image.size(0))
        evidence_value = _metadata_tensor(batch, "evidence_target", image.size(0))
        if noise_value is not None:
            noise.append(noise_value)
        if occlusion_value is not None:
            occlusion.append(occlusion_value)
        if resistance_value is not None:
            resistance_target.append(resistance_value)
        if evidence_value is not None:
            evidence_target.append(evidence_value)
    result = {"features": torch.cat(features, dim=0), "labels": torch.cat(labels, dim=0)}
    if noise:
        result["noise_level"] = torch.cat(noise, dim=0)
    if occlusion:
        result["occlusion_level"] = torch.cat(occlusion, dim=0)
    if resistance_target:
        result["resistance_target"] = torch.cat(resistance_target, dim=0)
    if evidence_target:
        result["evidence_target"] = torch.cat(evidence_target, dim=0)
    return result


def _cache_complete(output_dir: Path) -> bool:
    required = [
        "features_train.pt",
        "labels_train.pt",
        "features_val.pt",
        "labels_val.pt",
        "features_test.pt",
        "labels_test.pt",
        "metadata.json",
    ]
    return all((output_dir / name).exists() for name in required)


def extract_features(args: argparse.Namespace | SimpleNamespace) -> Path:
    _set_worker_strategy(args.num_workers)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _cache_complete(output_dir) and not getattr(args, "force", False):
        return output_dir

    device = _device(args.device)
    train_set, val_set, test_set, input_channels, num_classes = _build_datasets(args)
    train_loader = _loader(train_set, args, shuffle=True)
    val_loader = _loader(val_set, args, shuffle=False)
    test_loader = _loader(test_set, args, shuffle=False)

    backbone = ConvBackbone(feature_dim=args.feature_dim, input_channels=input_channels).to(device)
    pretrain = _train_backbone(backbone, train_loader, num_classes, args, device)
    for split_name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        split = _extract_split(backbone, loader, device)
        torch.save(split["features"], output_dir / f"features_{split_name}.pt")
        torch.save(split["labels"], output_dir / f"labels_{split_name}.pt")
        if "noise_level" in split:
            torch.save(split["noise_level"], output_dir / f"noise_level_{split_name}.pt")
        if "occlusion_level" in split:
            torch.save(split["occlusion_level"], output_dir / f"occlusion_level_{split_name}.pt")
        if "resistance_target" in split:
            torch.save(split["resistance_target"], output_dir / f"resistance_target_{split_name}.pt")
        if "evidence_target" in split:
            torch.save(split["evidence_target"], output_dir / f"evidence_target_{split_name}.pt")

    torch.save(backbone.state_dict(), output_dir / "backbone.pt")
    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "image_size": args.image_size,
        "feature_dim": args.feature_dim,
        "input_channels": input_channels,
        "num_classes": num_classes,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "pretrain": pretrain,
    }
    write_json(output_dir / "metadata.json", metadata)
    return output_dir


def main() -> None:
    output = extract_features(build_parser().parse_args())
    print(json.dumps({"feature_dir": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
