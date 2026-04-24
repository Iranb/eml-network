from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eml_mnist import EfficientEMLImageClassifier, EfficientEMLTextEncoder, EfficientEMLTextGenerationHead, build_mnist_eml_model
from eml_mnist.diagnostics import collect_eml_diagnostics
from eml_mnist.experiment_utils import ExperimentLogger, count_parameters, grad_norm, safe_torchvision_available
from eml_mnist.graph import EMLMessagePassing, EMLStateUpdateCell
from eml_mnist.image_datasets import SyntheticShapeEnergyDataset
from eml_mnist.metrics import classification_accuracy, perplexity, token_accuracy, topk_accuracy
from eml_mnist.text_codecs import CharVocabulary
from eml_mnist.text_datasets import SyntheticTextEnergyDataset
from eml_mnist.training import build_classification_loaders, resolve_device, set_seed


class EfficientTextLM(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, state_dim: int = 32, hidden_dim: int = 64, window_size: int = 8) -> None:
        super().__init__()
        self.encoder = EfficientEMLTextEncoder(
            vocab_size=vocab_size,
            embed_dim=24,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_hypotheses=4,
            num_attractors=4,
            representation_dim=state_dim,
            causal_window_size=window_size,
            chunk_size=8,
            pad_id=pad_id,
        )
        self.head = EfficientEMLTextGenerationHead(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
        )

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor, warmup_eta: float = 1.0) -> Dict[str, Any]:
        encoder_out = self.encoder(input_ids, padding_mask=padding_mask, warmup_eta=warmup_eta)
        head_out = self.head(encoder_out["sequence_states"], padding_mask=padding_mask, warmup_eta=warmup_eta)
        return {
            **head_out,
            "encoder": encoder_out,
            "sequence_states": encoder_out["sequence_states"],
            "representation": encoder_out["representation"],
            "diagnostics": encoder_out["diagnostics"],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standardized EML validation experiments")
    parser.add_argument("--mode", choices=["smoke", "ablation", "cifar-medium", "text-medium"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-root", default="reports/runs")
    parser.add_argument("--data-dir", default="~/dataset/data")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--eval-batches", type=int, default=20)
    return parser


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def _record_failed(
    run_id: str,
    args: argparse.Namespace,
    task_name: str,
    model_name: str,
    dataset_name: str,
    exc: BaseException,
) -> None:
    logger = ExperimentLogger(
        run_id=run_id,
        config={
            "mode": args.mode,
            "task_name": task_name,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seed": args.seed,
            "device": args.device,
        },
        root=args.runs_root,
    )
    logger.set_model_info(extra={"num_params": 0, "trainable_params": 0})
    logger.log_text(f"FAILED: {repr(exc)}")
    logger.finalize(summary={}, status="FAILED", reason=repr(exc))


def _safe_run(
    run_id: str,
    args: argparse.Namespace,
    task_name: str,
    model_name: str,
    dataset_name: str,
    fn: Any,
) -> None:
    try:
        fn()
    except Exception as exc:
        _record_failed(run_id, args, task_name, model_name, dataset_name, exc)


def _warmup(step: int, steps: int, enabled: bool = True) -> float:
    if not enabled:
        return 1.0
    return min(1.0, float(step + 1) / max(1, steps))


def _finalize_training(
    logger: ExperimentLogger,
    model: nn.Module,
    final_metrics: Dict[str, Any],
    final_diagnostics: Dict[str, Any],
    total_time: float,
) -> None:
    model_info = count_parameters(model)
    summary = {
        **final_metrics,
        "best_metric": final_metrics.get("best_metric", final_metrics.get("final_metric", "")),
        "final_metric": final_metrics.get("final_metric", final_metrics.get("best_metric", "")),
        "total_train_time_sec": total_time,
        "final_diagnostics": final_diagnostics,
    }
    logger.finalize(summary=summary, model_info=model_info)


def _update_early_stop(
    current_loss: float,
    best_loss: float,
    stale_steps: int,
    args: argparse.Namespace,
) -> tuple[float, int, bool]:
    if current_loss < best_loss - args.min_delta:
        return current_loss, 0, False
    stale_steps += 1
    return best_loss, stale_steps, bool(args.early_stop and stale_steps >= args.patience)


def _image_dataset(size: int, image_size: int, seed: int) -> SyntheticShapeEnergyDataset:
    return SyntheticShapeEnergyDataset(size=size, image_size=image_size, seed=seed, target_type="shape")


def _train_image_model(
    run_id: str,
    model_name: str,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    warmup_enabled: bool = True,
) -> None:
    config = {
        "mode": args.mode,
        "task_name": "image_synthetic",
        "model_name": model_name,
        "dataset_name": "SyntheticShapeEnergyDataset",
        "seed": seed,
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "warmup_enabled": warmup_enabled,
        "early_stop": args.early_stop,
        "patience": args.patience,
        "min_delta": args.min_delta,
    }
    logger = ExperimentLogger(run_id=run_id, config=config, root=args.runs_root)
    logger.set_model_info(model)
    set_seed(seed)
    dataset = _image_dataset(max(args.steps * args.batch_size * 2, 64), args.image_size, seed)
    loader = itertools.cycle(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_acc = 0.0
    final_metrics: Dict[str, Any] = {}
    final_diag: Dict[str, Any] = {}
    best_loss = float("inf")
    stale_steps = 0
    early_stop_step = 0
    start = time.time()
    for step in range(args.steps):
        batch = next(loader)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        step_start = time.time()
        warmup_eta = _warmup(step, args.steps, warmup_enabled)
        outputs = model(images, warmup_eta=warmup_eta)
        loss = F.cross_entropy(outputs["logits"], labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_time = time.time() - step_start
        accuracy = classification_accuracy(outputs["logits"].detach(), labels)
        top5 = topk_accuracy(outputs["logits"].detach(), labels, k=5)
        best_acc = max(best_acc, accuracy)
        diagnostics = collect_eml_diagnostics(outputs)
        metrics = {
            "step": step + 1,
            "train_loss": float(loss.detach().cpu().item()),
            "train_accuracy": accuracy,
            "top1_accuracy": accuracy,
            "top5_accuracy": top5,
            "learning_rate": args.lr,
            "grad_norm": float(norm.detach().cpu().item()) if torch.is_tensor(norm) else float(norm),
            "step_time_sec": step_time,
            "wall_clock_time_sec": time.time() - start,
            "examples_per_sec": float(images.size(0) / max(step_time, 1.0e-9)),
            "peak_memory_mb": _peak_memory_mb(device),
        }
        logger.log_step(metrics, diagnostics)
        final_metrics = {
            **metrics,
            "final_train_loss": metrics["train_loss"],
            "final_train_accuracy": accuracy,
            "best_metric": best_acc,
            "final_metric": accuracy,
            "completed_steps": step + 1,
            "early_stop_triggered": False,
            "early_stop_step": "",
        }
        final_diag = diagnostics
        best_loss, stale_steps, should_stop = _update_early_stop(metrics["train_loss"], best_loss, stale_steps, args)
        if should_stop:
            early_stop_step = step + 1
            final_metrics["early_stop_triggered"] = True
            final_metrics["early_stop_step"] = early_stop_step
            logger.log_text(f"early_stop step={early_stop_step} best_loss={best_loss:.6f}")
            break
    if early_stop_step == 0 and final_metrics:
        final_metrics["early_stop_triggered"] = False
    _finalize_training(logger, model, final_metrics, final_diag, time.time() - start)


def _train_text_model(
    run_id: str,
    model_name: str,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> None:
    vocab = CharVocabulary()
    config = {
        "mode": args.mode,
        "task_name": "text_synthetic",
        "model_name": model_name,
        "dataset_name": "SyntheticTextEnergyDataset",
        "seed": seed,
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "early_stop": args.early_stop,
        "patience": args.patience,
        "min_delta": args.min_delta,
    }
    logger = ExperimentLogger(run_id=run_id, config=config, root=args.runs_root)
    logger.set_model_info(model)
    set_seed(seed)
    dataset = SyntheticTextEnergyDataset(size=max(args.steps * args.batch_size * 2, 64), seq_len=args.seq_len, vocab=vocab, seed=seed)
    loader = itertools.cycle(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_acc = 0.0
    final_metrics: Dict[str, Any] = {}
    final_diag: Dict[str, Any] = {}
    best_loss = float("inf")
    stale_steps = 0
    early_stop_step = 0
    start = time.time()
    for step in range(args.steps):
        batch = next(loader)
        input_ids = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)
        mask = batch["padding_mask"].to(device).bool()
        step_start = time.time()
        outputs = model(input_ids, mask, warmup_eta=_warmup(step, args.steps, True))
        logits = outputs["logits"]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=vocab.pad_id)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_time = time.time() - step_start
        tok_acc = token_accuracy(logits.detach(), targets, mask)
        best_acc = max(best_acc, tok_acc)
        diagnostics = collect_eml_diagnostics(outputs)
        loss_value = float(loss.detach().cpu().item())
        valid_tokens = int(mask.sum().detach().cpu().item())
        metrics = {
            "step": step + 1,
            "next_token_loss": loss_value,
            "train_loss": loss_value,
            "token_accuracy": tok_acc,
            "char_accuracy": tok_acc,
            "perplexity": perplexity(loss_value),
            "bits_per_char": loss_value / math.log(2.0),
            "learning_rate": args.lr,
            "grad_norm": float(norm.detach().cpu().item()) if torch.is_tensor(norm) else float(norm),
            "step_time_sec": step_time,
            "wall_clock_time_sec": time.time() - start,
            "tokens_per_sec": float(valid_tokens / max(step_time, 1.0e-9)),
            "peak_memory_mb": _peak_memory_mb(device),
        }
        logger.log_step(metrics, diagnostics)
        final_metrics = {
            **metrics,
            "final_train_loss": loss_value,
            "final_train_accuracy": tok_acc,
            "best_metric": best_acc,
            "final_metric": tok_acc,
            "completed_steps": step + 1,
            "early_stop_triggered": False,
            "early_stop_step": "",
        }
        final_diag = diagnostics
        best_loss, stale_steps, should_stop = _update_early_stop(loss_value, best_loss, stale_steps, args)
        if should_stop:
            early_stop_step = step + 1
            final_metrics["early_stop_triggered"] = True
            final_metrics["early_stop_step"] = early_stop_step
            logger.log_text(f"early_stop step={early_stop_step} best_loss={best_loss:.6f}")
            break
    if early_stop_step == 0 and final_metrics:
        final_metrics["early_stop_triggered"] = False
    _finalize_training(logger, model, final_metrics, final_diag, time.time() - start)


def _run_mechanism_probe(
    run_id: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    responsibility_mode: bool,
    use_null: bool,
    update_mode: str,
) -> None:
    config = {
        "mode": args.mode,
        "task_name": "mechanism_probe",
        "model_name": run_id,
        "dataset_name": "synthetic_probe_tensors",
        "seed": seed,
        "device": str(device),
        "responsibility_mode": responsibility_mode,
        "use_null": use_null,
        "update_mode": update_mode,
    }
    logger = ExperimentLogger(run_id=run_id, config=config, root=args.runs_root)
    set_seed(seed)
    message = EMLMessagePassing(
        slot_dim=16,
        event_dim=8,
        hidden_dim=32,
        responsibility_mode=responsibility_mode,
        responsibility_use_null=use_null,
    ).to(device)
    update = EMLStateUpdateCell(slot_dim=16, event_dim=8, hidden_dim=32, update_mode=update_mode).to(device)
    wrapper = nn.ModuleDict({"message_layer": message, "state_update_cell": update})
    logger.set_model_info(wrapper)
    active_states = torch.randn(4, 5, 16, device=device)
    event = torch.randn(4, 8, device=device)
    start = time.time()
    message_out = message(active_states, event, warmup_eta=1.0)
    update_out = update(active_states, message_out["aggregated_messages"], event, warmup_eta=1.0)
    outputs = {"message": message_out, "update": update_out}
    diagnostics = collect_eml_diagnostics(outputs)
    finite_values = [
        torch.isfinite(message_out["aggregated_messages"]).all(),
        torch.isfinite(update_out["slot_states"]).all(),
    ]
    finite_ok = bool(torch.stack(finite_values).all().item())
    metrics = {
        "step": 1,
        "train_loss": 0.0,
        "final_metric": 1.0 if finite_ok else 0.0,
        "best_metric": 1.0 if finite_ok else 0.0,
        "message_norm_mean": float(message_out["message_norm"].detach().float().mean().cpu().item()),
        "update_norm_mean": float((update_out["slot_states"] - active_states).detach().float().norm(dim=-1).mean().cpu().item()),
        "wall_clock_time_sec": time.time() - start,
        "step_time_sec": time.time() - start,
        "peak_memory_mb": _peak_memory_mb(device),
        "nan_inf_count": 0.0 if finite_ok else 1.0,
    }
    logger.log_step(metrics, diagnostics)
    logger.finalize(
        summary={**metrics, "final_diagnostics": diagnostics, "total_train_time_sec": metrics["wall_clock_time_sec"]},
        model_info=count_parameters(wrapper),
    )


def _train_cifar_model(
    run_id: str,
    model_name: str,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> None:
    config = {
        "mode": args.mode,
        "task_name": "image_cifar",
        "model_name": model_name,
        "dataset_name": "CIFAR10",
        "seed": seed,
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "early_stop": args.early_stop,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "eval_batches": args.eval_batches,
    }
    logger = ExperimentLogger(run_id=run_id, config=config, root=args.runs_root)
    logger.set_model_info(model)
    set_seed(seed)
    train_loader, eval_loader = build_classification_loaders(
        dataset_name="cifar10",
        data_dir=str(Path(args.data_dir).expanduser()),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.allow_download,
    )
    train_iter = itertools.cycle(train_loader)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    best_eval_acc = 0.0
    best_eval_loss = float("inf")
    stale_evals = 0
    final_metrics: Dict[str, Any] = {}
    final_diag: Dict[str, Any] = {}
    start = time.time()

    for step in range(args.steps):
        images, labels = next(train_iter)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        step_start = time.time()
        outputs = model(images, warmup_eta=_warmup(step, args.steps, True))
        loss = F.cross_entropy(outputs["logits"], labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        diagnostics = collect_eml_diagnostics(outputs)
        accuracy = classification_accuracy(outputs["logits"].detach(), labels)
        metrics = {
            "step": step + 1,
            "train_loss": float(loss.detach().cpu().item()),
            "train_accuracy": accuracy,
            "top1_accuracy": accuracy,
            "learning_rate": args.lr,
            "grad_norm": float(norm.detach().cpu().item()) if torch.is_tensor(norm) else float(norm),
            "step_time_sec": time.time() - step_start,
            "wall_clock_time_sec": time.time() - start,
            "examples_per_sec": float(images.size(0) / max(time.time() - step_start, 1.0e-9)),
            "peak_memory_mb": _peak_memory_mb(device),
            "early_stop_triggered": False,
            "early_stop_step": "",
            "completed_steps": step + 1,
        }

        should_eval = (step + 1 == args.steps) or ((step + 1) % max(1, args.eval_batches) == 0)
        if should_eval:
            eval_loss = 0.0
            eval_correct = 0
            eval_seen = 0
            model.eval()
            with torch.no_grad():
                for eval_index, (eval_images, eval_labels) in enumerate(eval_loader):
                    if args.eval_batches and eval_index >= args.eval_batches:
                        break
                    eval_images = eval_images.to(device, non_blocking=True)
                    eval_labels = eval_labels.to(device, non_blocking=True)
                    eval_out = model(eval_images, warmup_eta=1.0)
                    batch_loss = F.cross_entropy(eval_out["logits"], eval_labels)
                    eval_loss += float(batch_loss.detach().cpu().item()) * eval_images.size(0)
                    eval_correct += int((eval_out["logits"].detach().argmax(dim=1) == eval_labels).sum().cpu())
                    eval_seen += int(eval_images.size(0))
            model.train()
            eval_loss = eval_loss / max(1, eval_seen)
            eval_acc = eval_correct / max(1, eval_seen)
            best_eval_acc = max(best_eval_acc, eval_acc)
            if eval_loss < best_eval_loss - args.min_delta:
                best_eval_loss = eval_loss
                stale_evals = 0
            else:
                stale_evals += 1
            metrics.update(
                {
                    "val_loss": eval_loss,
                    "val_accuracy": eval_acc,
                    "test_loss": eval_loss,
                    "test_accuracy": eval_acc,
                    "best_eval_accuracy": best_eval_acc,
                }
            )
            if args.early_stop and stale_evals >= args.patience:
                metrics["early_stop_triggered"] = True
                metrics["early_stop_step"] = step + 1
                logger.log_text(f"early_stop step={step + 1} best_eval_loss={best_eval_loss:.6f}")

        logger.log_step(metrics, diagnostics)
        final_metrics = {
            **metrics,
            "final_train_loss": metrics["train_loss"],
            "final_train_accuracy": metrics["train_accuracy"],
            "best_metric": best_eval_acc if best_eval_acc > 0.0 else metrics["train_accuracy"],
            "final_metric": metrics.get("val_accuracy", metrics["train_accuracy"]),
        }
        final_diag = diagnostics
        if metrics["early_stop_triggered"]:
            break

    _finalize_training(logger, model, final_metrics, final_diag, time.time() - start)


def _build_old_image_model(model_name: str, image_size: int) -> nn.Module:
    return build_mnist_eml_model(
        {
            "model_name": model_name,
            "num_classes": 5,
            "image_size": image_size,
            "input_channels": 3,
            "feature_dim": 32,
            "hidden_dim": 64,
            "bank_dim": 32,
            "bank_blocks": 2 if model_name == "pure_eml_v2" else 1,
            "patch_size": 4,
            "patch_stride": 2,
            "local_window_size": 3,
        }
    )


def _build_cnn_model(image_size: int) -> nn.Module:
    return _build_old_image_model("cnn_eml", image_size)


def _build_efficient_image(num_attractors: int = 4, local_window_size: int = 3) -> nn.Module:
    return EfficientEMLImageClassifier(
        num_classes=5,
        input_channels=3,
        state_dim=32,
        hidden_dim=64,
        num_hypotheses=4,
        num_attractors=num_attractors,
        representation_dim=32,
        patch_stride=4,
        local_window_size=local_window_size,
        composition_region_size=2,
    )


def _mark_planned_not_run(args: argparse.Namespace, reason_prefix: str) -> None:
    planned = [
        ("local_conv_baseline", "image_synthetic", "LocalConvBaseline", "SyntheticShapeEnergyDataset", "not implemented"),
        ("local_text_linear_baseline", "text_synthetic", "LocalTextCodecLinear", "SyntheticTextEnergyDataset", "not standardized"),
        ("cifar_medium_suite", "image_cifar", "selected_image_models", "CIFAR10", "not requested in this mode"),
        ("text_medium_suite", "text_synthetic", "selected_text_models", "SyntheticTextEnergyDataset", "not requested in this mode"),
        ("full_seeded_ablation", "mechanism_ablation", "all_supported_cells", "mixed", "not requested in this mode"),
    ]
    for run_id, task, model, dataset, reason in planned:
        ExperimentLogger.not_run(
            run_id=run_id,
            config={
                "mode": args.mode,
                "task_name": task,
                "model_name": model,
                "dataset_name": dataset,
                "seed": args.seed,
                "device": args.device,
            },
            reason=f"{reason_prefix}: {reason}",
            root=args.runs_root,
        )


def run_smoke(args: argparse.Namespace, device: torch.device) -> None:
    _safe_run(
        "smoke_image_cnn_eml_baseline",
        args,
        "image_synthetic",
        "cnn_eml",
        "SyntheticShapeEnergyDataset",
        lambda: _train_image_model("smoke_image_cnn_eml_baseline", "cnn_eml", _build_cnn_model(args.image_size), args, device, args.seed),
    )
    _safe_run(
        "smoke_image_efficient_eml",
        args,
        "image_synthetic",
        "EfficientEMLImageClassifier",
        "SyntheticShapeEnergyDataset",
        lambda: _train_image_model("smoke_image_efficient_eml", "EfficientEMLImageClassifier", _build_efficient_image(4), args, device, args.seed + 1),
    )
    vocab = CharVocabulary()
    _safe_run(
        "smoke_text_efficient_eml",
        args,
        "text_synthetic",
        "EfficientEMLTextEncoder",
        "SyntheticTextEnergyDataset",
        lambda: _train_text_model("smoke_text_efficient_eml", "EfficientEMLTextEncoder", EfficientTextLM(len(vocab), vocab.pad_id), args, device, args.seed + 2),
    )
    _safe_run(
        "probe_gate_compat_sigmoid_update",
        args,
        "mechanism_probe",
        "probe_gate_compat_sigmoid_update",
        "synthetic_probe_tensors",
        lambda: _run_mechanism_probe("probe_gate_compat_sigmoid_update", args, device, args.seed + 3, False, False, "sigmoid"),
    )
    _safe_run(
        "probe_responsibility_no_null_precision",
        args,
        "mechanism_probe",
        "probe_responsibility_no_null_precision",
        "synthetic_probe_tensors",
        lambda: _run_mechanism_probe("probe_responsibility_no_null_precision", args, device, args.seed + 4, True, False, "precision"),
    )
    _safe_run(
        "probe_responsibility_with_null_precision",
        args,
        "mechanism_probe",
        "probe_responsibility_with_null_precision",
        "synthetic_probe_tensors",
        lambda: _run_mechanism_probe("probe_responsibility_with_null_precision", args, device, args.seed + 5, True, True, "precision"),
    )
    _mark_planned_not_run(args, "smoke mode")


def run_ablation(args: argparse.Namespace, device: torch.device) -> None:
    if args.device == "cuda" and device.type != "cuda":
        ExperimentLogger.not_run(
            "ablation_cuda_unavailable",
            {
                "mode": args.mode,
                "task_name": "mechanism_ablation",
                "model_name": "ablation_suite",
                "dataset_name": "mixed",
                "seed": args.seed,
                "device": args.device,
            },
            "CUDA requested but unavailable",
            root=args.runs_root,
        )
        return
    for offset, seed in enumerate((args.seed, args.seed + 1, args.seed + 2)):
        _safe_run(
            f"ablation_gate_sigmoid_seed{offset}",
            args,
            "mechanism_probe",
            "gate_sigmoid_update",
            "synthetic_probe_tensors",
            lambda seed=seed, offset=offset: _run_mechanism_probe(f"ablation_gate_sigmoid_seed{offset}", args, device, seed, False, False, "sigmoid"),
        )
        _safe_run(
            f"ablation_resp_no_null_seed{offset}",
            args,
            "mechanism_probe",
            "responsibility_no_null_precision",
            "synthetic_probe_tensors",
            lambda seed=seed, offset=offset: _run_mechanism_probe(f"ablation_resp_no_null_seed{offset}", args, device, seed + 10, True, False, "precision"),
        )
        _safe_run(
            f"ablation_resp_null_seed{offset}",
            args,
            "mechanism_probe",
            "responsibility_null_precision",
            "synthetic_probe_tensors",
            lambda seed=seed, offset=offset: _run_mechanism_probe(f"ablation_resp_null_seed{offset}", args, device, seed + 20, True, True, "precision"),
        )

    image_runs = [
        ("ablation_image_cnn_eml", "cnn_eml", lambda: _build_cnn_model(args.image_size), True, args.seed + 30),
        ("ablation_image_pure_eml", "pure_eml", lambda: _build_old_image_model("pure_eml", args.image_size), True, args.seed + 31),
        ("ablation_image_pure_eml_v2", "pure_eml_v2", lambda: _build_old_image_model("pure_eml_v2", args.image_size), True, args.seed + 32),
        ("ablation_image_eff_attr4", "EfficientEMLImageClassifier_attr4", lambda: _build_efficient_image(4, 3), True, args.seed + 33),
        ("ablation_image_eff_attr8", "EfficientEMLImageClassifier_attr8", lambda: _build_efficient_image(8, 3), True, args.seed + 34),
        ("ablation_image_eff_window5", "EfficientEMLImageClassifier_window5", lambda: _build_efficient_image(4, 5), True, args.seed + 35),
        ("ablation_image_eff_no_warmup", "EfficientEMLImageClassifier_no_warmup", lambda: _build_efficient_image(4, 3), False, args.seed + 36),
    ]
    for run_id, model_name, factory, warmup_enabled, seed in image_runs:
        _safe_run(
            run_id,
            args,
            "image_synthetic",
            model_name,
            "SyntheticShapeEnergyDataset",
            lambda run_id=run_id, model_name=model_name, factory=factory, warmup_enabled=warmup_enabled, seed=seed: _train_image_model(
                run_id,
                model_name,
                factory(),
                args,
                device,
                seed,
                warmup_enabled,
            ),
        )

    vocab = CharVocabulary()
    for window_size, seed in [(8, args.seed + 40), (16, args.seed + 41), (32, args.seed + 42)]:
        run_id = f"ablation_text_eff_window{window_size}"
        _safe_run(
            run_id,
            args,
            "text_synthetic",
            f"EfficientEMLTextEncoder_window{window_size}",
            "SyntheticTextEnergyDataset",
            lambda run_id=run_id, window_size=window_size, seed=seed: _train_text_model(
                run_id,
                f"EfficientEMLTextEncoder_window{window_size}",
                EfficientTextLM(len(vocab), vocab.pad_id, window_size=window_size),
                args,
                device,
                seed,
            ),
        )

    unsupported = [
        ("ablation_no_composition", "image_synthetic", "EfficientEMLImageClassifier_no_composition", "SyntheticShapeEnergyDataset", "model switch is not standardized"),
        ("ablation_no_attractor", "image_synthetic", "EfficientEMLImageClassifier_no_attractor", "SyntheticShapeEnergyDataset", "model switch is not standardized"),
        ("ablation_head_without_ambiguity", "image_synthetic", "prototype_head_without_ambiguity", "SyntheticShapeEnergyDataset", "head switch is not standardized"),
        ("ablation_sigmoid_gate_mean", "mechanism_probe", "sigmoid_gate_mean", "synthetic_probe_tensors", "graph mode is not implemented"),
        ("ablation_thresholded_null", "mechanism_probe", "responsibility_thresholded_null", "synthetic_probe_tensors", "threshold mode is not implemented"),
    ]
    for run_id, task, model_name, dataset_name, reason in unsupported:
        ExperimentLogger.not_run(
            run_id=run_id,
            config={
                "mode": args.mode,
                "task_name": task,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "seed": args.seed,
                "device": str(device),
            },
            reason=reason,
            root=args.runs_root,
        )


def run_cifar_medium(args: argparse.Namespace, device: torch.device) -> None:
    torchvision_ok, reason = safe_torchvision_available()
    if not torchvision_ok:
        ExperimentLogger.not_run(
            "cifar_medium_torchvision_unavailable",
            {"mode": args.mode, "task_name": "image_cifar", "model_name": "selected_image_models", "dataset_name": "CIFAR10", "seed": args.seed, "device": str(device)},
            reason,
            root=args.runs_root,
        )
        return
    try:
        build_classification_loaders(
            dataset_name="cifar10",
            data_dir=str(Path(args.data_dir).expanduser()),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=args.allow_download,
        )
    except Exception as exc:
        ExperimentLogger.not_run(
            "cifar_medium_data_unavailable",
            {"mode": args.mode, "task_name": "image_cifar", "model_name": "selected_image_models", "dataset_name": "CIFAR10", "seed": args.seed, "device": str(device)},
            repr(exc),
            root=args.runs_root,
        )
        return
    cifar_runs = [
        ("cifar_cnn_eml", "cnn_eml", lambda: build_mnist_eml_model({"model_name": "cnn_eml", "num_classes": 10, "image_size": 32, "input_channels": 3, "feature_dim": 64, "hidden_dim": 96, "bank_dim": 64, "bank_blocks": 2})),
        ("cifar_pure_eml", "pure_eml", lambda: build_mnist_eml_model({"model_name": "pure_eml", "num_classes": 10, "image_size": 32, "input_channels": 3, "feature_dim": 64, "hidden_dim": 96, "bank_dim": 64, "bank_blocks": 1, "patch_size": 4, "patch_stride": 2, "local_window_size": 3})),
        ("cifar_pure_eml_v2", "pure_eml_v2", lambda: build_mnist_eml_model({"model_name": "pure_eml_v2", "num_classes": 10, "image_size": 32, "input_channels": 3, "feature_dim": 64, "hidden_dim": 96, "bank_dim": 64, "bank_blocks": 2, "patch_size": 4, "patch_stride": 2, "local_window_size": 3})),
        ("cifar_efficient_eml_image", "EfficientEMLImageClassifier", lambda: EfficientEMLImageClassifier(num_classes=10, input_channels=3, state_dim=32, hidden_dim=96, num_hypotheses=4, num_attractors=4, representation_dim=32, patch_stride=4, local_window_size=3, composition_region_size=2)),
    ]
    for offset, (run_id, model_name, factory) in enumerate(cifar_runs):
        _safe_run(
            run_id,
            args,
            "image_cifar",
            model_name,
            "CIFAR10",
            lambda run_id=run_id, model_name=model_name, factory=factory, offset=offset: _train_cifar_model(
                run_id,
                model_name,
                factory(),
                args,
                device,
                args.seed + offset,
            ),
        )


def run_text_medium(args: argparse.Namespace, device: torch.device) -> None:
    old_steps = args.steps
    args.steps = max(args.steps, 12)
    vocab = CharVocabulary()
    _train_text_model("text_medium_efficient_eml", "EfficientEMLTextEncoder", EfficientTextLM(len(vocab), vocab.pad_id), args, device, args.seed)
    ExperimentLogger.not_run(
        "text_medium_old_backbone",
        {"mode": args.mode, "task_name": "text_synthetic", "model_name": "EMLTextBackbone", "dataset_name": "SyntheticTextEnergyDataset", "seed": args.seed, "device": str(device)},
        "standard old-backbone wrapper is pending",
        root=args.runs_root,
    )
    args.steps = old_steps


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    Path(args.runs_root).mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if args.mode == "smoke":
        run_smoke(args, device)
    elif args.mode == "ablation":
        run_ablation(args, device)
    elif args.mode == "cifar-medium":
        run_cifar_medium(args, device)
    elif args.mode == "text-medium":
        run_text_medium(args, device)
    else:
        raise ValueError(f"unknown mode: {args.mode}")
    print(f"validation suite complete: mode={args.mode} runs_root={args.runs_root}")


if __name__ == "__main__":
    main()
