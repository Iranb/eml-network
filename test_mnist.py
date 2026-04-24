import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from eml_mnist import build_mnist_eml_model
from eml_mnist.training import build_classification_loaders, ensure_dir, move_batch_to_device, resolve_device, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an EML-based image classification checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    per_class_total = torch.zeros(num_classes, dtype=torch.long)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch in tqdm(loader, desc="test", leave=False):
        images, targets = move_batch_to_device(batch, device)
        outputs = model(images, warmup_eta=1.0)
        predictions = outputs["logits"].argmax(dim=1)

        total += targets.size(0)
        correct += (predictions == targets).sum().item()

        for cls in range(num_classes):
            cls_mask = targets == cls
            per_class_total[cls] += cls_mask.sum().cpu()
            per_class_correct[cls] += ((predictions == targets) & cls_mask).sum().cpu()

        for target, pred in zip(targets.view(-1), predictions.view(-1)):
            confusion[target.long().cpu(), pred.long().cpu()] += 1

    per_class_accuracy: List[float] = []
    for cls in range(num_classes):
        if per_class_total[cls].item() == 0:
            per_class_accuracy.append(0.0)
        else:
            per_class_accuracy.append((per_class_correct[cls].item() / per_class_total[cls].item()))

    return {
        "accuracy": correct / total,
        "num_samples": total,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    _, test_loader = build_classification_loaders(
        dataset_name=config.get("dataset", "mnist"),
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_mnist_eml_model(config).to(device)
    model.load_state_dict(checkpoint["model"])

    metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=int(config.get("num_classes", 10)),
    )
    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["dataset"] = config.get("dataset", "mnist")
    metrics["model_name"] = config.get("model_name", "cnn_eml")

    print(
        f"dataset={metrics['dataset']} "
        f"model={metrics['model_name']} "
        f"test_accuracy={metrics['accuracy']:.4f}"
    )

    if args.output_dir:
        output_dir = ensure_dir(args.output_dir)
        save_json(output_dir / "test_metrics.json", metrics)
    else:
        checkpoint_dir = ensure_dir(str(Path(args.checkpoint).resolve().parent))
        save_json(checkpoint_dir / "test_metrics.json", metrics)


if __name__ == "__main__":
    main()
