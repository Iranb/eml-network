from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
import traceback
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
from eml_mnist.metrics import classification_accuracy, pearson_corr, perplexity, token_accuracy, topk_accuracy
from eml_mnist.primitives import EMLPrecisionUpdate, EMLResponsibility, EMLUnit
from eml_mnist.representation import EMLAttractorMemory, EMLComposition
from eml_mnist.schedules import get_staged_hardening_values
from eml_mnist.text_codecs import CharVocabulary
from eml_mnist.text_datasets import SyntheticTextEnergyDataset
from eml_mnist.training import build_classification_loaders, resolve_device, set_seed


class EfficientTextLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        state_dim: int = 32,
        hidden_dim: int = 64,
        window_size: int = 8,
        responsibility_mode: str = "standard",
        precision_old_confidence_init: float = 5.0,
        enable_composition: bool = True,
        enable_attractor: bool = True,
    ) -> None:
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
            responsibility_mode=responsibility_mode,
            precision_old_confidence_init=precision_old_confidence_init,
            enable_composition=enable_composition,
            enable_attractor=enable_attractor,
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
    parser.add_argument("--staged-hardening", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--responsibility-temp-start", type=float, default=2.0)
    parser.add_argument("--responsibility-temp-end", type=float, default=0.8)
    parser.add_argument("--ambiguity-warmup-steps", type=int, default=100)
    parser.add_argument("--null-threshold-start", type=float, default=1.0)
    parser.add_argument("--null-threshold-end", type=float, default=0.0)
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
            "num_workers": args.num_workers,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "early_stop": args.early_stop,
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
        root=args.runs_root,
    )
    trace = traceback.format_exc()
    logger.set_model_info(extra={"num_params": 0, "trainable_params": 0})
    logger.log_text(f"FAILED: {repr(exc)}")
    logger.log_text(trace)
    logger.finalize(summary={"error_trace": trace}, status="FAILED", reason=repr(exc))


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


def _schedule_values(args: argparse.Namespace, step: int, steps: int, warmup_enabled: bool = True) -> Dict[str, float]:
    if not getattr(args, "staged_hardening", False):
        return {
            "warmup_eta": _warmup(step, steps, warmup_enabled),
            "responsibility_temperature": float("nan"),
            "ambiguity_weight": float("nan"),
            "null_threshold": float("nan"),
            "attractor_entropy_weight": float("nan"),
            "precision_update_threshold": float("nan"),
        }
    return get_staged_hardening_values(
        step + 1,
        steps,
        {
            "warmup_steps": args.warmup_steps,
            "responsibility_temp_start": args.responsibility_temp_start,
            "responsibility_temp_end": args.responsibility_temp_end,
            "ambiguity_warmup_steps": args.ambiguity_warmup_steps,
            "null_threshold_start": args.null_threshold_start,
            "null_threshold_end": args.null_threshold_end,
        },
    )


def _apply_schedule(model: nn.Module, values: Dict[str, float]) -> None:
    if not values or not math.isfinite(values.get("responsibility_temperature", float("nan"))):
        return
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, EMLResponsibility):
                module.temperature.fill_(float(values["responsibility_temperature"]))
                if hasattr(module, "evidence_threshold"):
                    module.evidence_threshold.fill_(float(values["null_threshold"]))
            if hasattr(module, "ambiguity_weight"):
                try:
                    module.ambiguity_weight = float(values["ambiguity_weight"])
                except Exception:
                    pass


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
        "num_workers": args.num_workers,
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
        schedule = _schedule_values(args, step, args.steps, warmup_enabled)
        _apply_schedule(model, schedule)
        warmup_eta = schedule["warmup_eta"]
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
        resistance_noise_corr = float("nan")
        resistance_occlusion_corr = float("nan")
        if torch.is_tensor(outputs.get("resistance")):
            sample_resistance = outputs["resistance"].detach().float()
            while sample_resistance.ndim > 1:
                sample_resistance = sample_resistance.mean(dim=-1)
            if torch.is_tensor(batch.get("noise_level")):
                resistance_noise_corr = pearson_corr(sample_resistance, batch["noise_level"].to(device))
            if torch.is_tensor(batch.get("occlusion_level")):
                resistance_occlusion_corr = pearson_corr(sample_resistance, batch["occlusion_level"].to(device))
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
            "resistance_noise_corr": resistance_noise_corr,
            "resistance_occlusion_corr": resistance_occlusion_corr,
            **schedule,
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
        "num_workers": args.num_workers,
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
        schedule = _schedule_values(args, step, args.steps, True)
        _apply_schedule(model, schedule)
        outputs = model(input_ids, mask, warmup_eta=schedule["warmup_eta"])
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
        corruption_resistance_corr = float("nan")
        if torch.is_tensor(outputs.get("resistance")) and torch.is_tensor(batch.get("corruption_mask")):
            token_resistance = outputs["resistance"].detach().float()
            while token_resistance.ndim > 2:
                token_resistance = token_resistance.mean(dim=-1)
            corruption_resistance_corr = pearson_corr(token_resistance[mask], batch["corruption_mask"].to(device).float()[mask])
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
            "corruption_resistance_corr": corruption_resistance_corr,
            **schedule,
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
    responsibility_distribution_mode: str = "standard",
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
        "responsibility_distribution_mode": responsibility_distribution_mode,
    }
    logger = ExperimentLogger(run_id=run_id, config=config, root=args.runs_root)
    set_seed(seed)
    message = EMLMessagePassing(
        slot_dim=16,
        event_dim=8,
        hidden_dim=32,
        responsibility_mode=responsibility_mode,
        responsibility_use_null=use_null,
        responsibility_distribution_mode=responsibility_distribution_mode,
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


def _run_named_mechanism_probe(
    run_id: str,
    probe_name: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> None:
    config = {
        "mode": args.mode,
        "task_name": "mechanism_probe",
        "model_name": probe_name,
        "dataset_name": "synthetic_probe_tensors",
        "seed": seed,
        "device": str(device),
        "probe_name": probe_name,
    }
    logger = ExperimentLogger(run_id=run_id, config=config, root=args.runs_root)
    set_seed(seed)
    start = time.time()
    modules: nn.Module | None = None
    outputs: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {"step": 1, "train_loss": 0.0}
    success = False

    if probe_name == "strong_signal_vs_many_weak_noise":
        responsibility = EMLResponsibility(mode="thresholded_null", use_null=True, temperature=1.0).to(device)
        energy = torch.tensor([[-2.0, -1.5, 5.0, -3.0, -2.5]], device=device)
        out = responsibility(energy)
        success = bool(out["neighbor_weights"][0, 2].detach().cpu().item() > 0.7)  # type: ignore[index]
        outputs = {"responsibility": out}
        metrics["selected_neighbor_weight"] = float(out["neighbor_weights"][0, 2].detach().cpu().item())  # type: ignore[index]
        modules = responsibility
    elif probe_name == "all_noise_should_choose_null":
        responsibility = EMLResponsibility(mode="thresholded_null", use_null=True, temperature=1.0).to(device)
        energy = torch.full((2, 6), -4.0, device=device)
        out = responsibility(energy)
        success = bool(out["null_weight"].min().detach().cpu().item() > 0.7)  # type: ignore[union-attr]
        outputs = {"responsibility": out}
        metrics["null_weight_min"] = float(out["null_weight"].min().detach().cpu().item())  # type: ignore[union-attr]
        modules = responsibility
    elif probe_name == "conflicting_neighbors_increase_resistance":
        unit = EMLUnit(dim=1).to(device)
        drive = torch.ones(4, 1, device=device)
        low_resistance = torch.zeros(4, 1, device=device)
        high_resistance = torch.full((4, 1), 5.0, device=device)
        low_energy = unit(drive, low_resistance)
        high_energy = unit(drive, high_resistance)
        success = bool(high_energy.mean().detach().cpu().item() < low_energy.mean().detach().cpu().item())
        outputs = {"drive": drive, "resistance": high_resistance, "energy": high_energy}
        metrics["low_resistance_energy"] = float(low_energy.mean().detach().cpu().item())
        metrics["high_resistance_energy"] = float(high_energy.mean().detach().cpu().item())
        modules = unit
    elif probe_name == "old_state_confident_new_evidence_weak_should_not_update":
        update = EMLPrecisionUpdate(old_confidence_init=0.0).to(device)
        state = torch.zeros(2, 3, 8, device=device)
        candidate = torch.ones_like(state)
        out = update(state, candidate, torch.full((2, 3, 1), -8.0, device=device), torch.full((2, 3, 1), 8.0, device=device))
        delta = (out["updated_state"] - state).norm(dim=-1).mean()
        success = bool(delta.detach().cpu().item() < 0.05)
        outputs = out
        metrics["update_delta"] = float(delta.detach().cpu().item())
        modules = update
    elif probe_name == "old_state_weak_new_evidence_strong_should_update":
        update = EMLPrecisionUpdate(old_confidence_init=0.0).to(device)
        state = torch.zeros(2, 3, 8, device=device)
        candidate = torch.ones_like(state)
        out = update(state, candidate, torch.full((2, 3, 1), 8.0, device=device), torch.full((2, 3, 1), -8.0, device=device))
        gate = out["update_gate"].mean()
        success = bool(gate.detach().cpu().item() > 0.7)
        outputs = out
        metrics["update_gate_mean"] = float(gate.detach().cpu().item())
        modules = update
    elif probe_name == "composition_requires_consistent_children":
        composition = EMLComposition(state_dim=8, hidden_dim=16, mode="text", region_size=4).to(device)
        base = torch.randn(2, 1, 8, device=device)
        consistent = base.expand(-1, 8, -1) + 0.01 * torch.randn(2, 8, 8, device=device)
        inconsistent = torch.randn(2, 8, 8, device=device) * 2.0
        mask = torch.ones(2, 8, device=device, dtype=torch.bool)
        out_consistent = composition(consistent, padding_mask=mask)
        out_inconsistent = composition(inconsistent, padding_mask=mask)
        consistent_var = consistent.view(2, 2, 4, 8).var(dim=2, unbiased=False).mean()
        inconsistent_var = inconsistent.view(2, 2, 4, 8).var(dim=2, unbiased=False).mean()
        success = bool(inconsistent_var.detach().cpu().item() > consistent_var.detach().cpu().item())
        outputs = {"consistent": out_consistent, "inconsistent": out_inconsistent}
        metrics["consistent_child_variance"] = float(consistent_var.detach().cpu().item())
        metrics["inconsistent_child_variance"] = float(inconsistent_var.detach().cpu().item())
        modules = composition
    elif probe_name == "attractor_should_not_collapse":
        attractor = EMLAttractorMemory(state_dim=8, hidden_dim=16, num_attractors=4).to(device)
        states = torch.randn(3, 10, 8, device=device)
        out = attractor(states)
        normalized = F.normalize(out["attractor_states"].detach().float(), dim=-1)  # type: ignore[index]
        cosine = normalized @ normalized.transpose(1, 2)
        mask = ~torch.eye(4, device=device, dtype=torch.bool).unsqueeze(0)
        diversity_penalty = cosine.masked_select(mask).square().mean()
        success = bool(torch.isfinite(diversity_penalty).item() and diversity_penalty.detach().cpu().item() < 0.95)
        outputs = out
        metrics["attractor_diversity_penalty"] = float(diversity_penalty.detach().cpu().item())
        modules = attractor
    else:
        raise ValueError(f"unknown mechanism probe: {probe_name}")

    diagnostics = collect_eml_diagnostics(outputs)
    elapsed = time.time() - start
    metrics.update(
        {
            "final_metric": 1.0 if success else 0.0,
            "best_metric": 1.0 if success else 0.0,
            "wall_clock_time_sec": elapsed,
            "step_time_sec": elapsed,
            "peak_memory_mb": _peak_memory_mb(device),
            "nan_inf_count": 0.0 if all(torch.isfinite(torch.tensor(v)).item() for v in metrics.values() if isinstance(v, (int, float))) else 1.0,
        }
    )
    logger.set_model_info(modules)
    logger.log_step(metrics, diagnostics)
    logger.finalize(
        summary={**metrics, "final_diagnostics": diagnostics, "total_train_time_sec": elapsed},
        model_info=count_parameters(modules) if modules is not None else {"num_params": 0, "trainable_params": 0},
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
        "num_workers": args.num_workers,
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
        schedule = _schedule_values(args, step, args.steps, True)
        _apply_schedule(model, schedule)
        outputs = model(images, warmup_eta=schedule["warmup_eta"])
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
            **schedule,
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


def _build_efficient_image(
    num_attractors: int = 4,
    local_window_size: int = 3,
    enable_composition: bool = True,
    enable_attractor: bool = True,
    center_ambiguity: bool = True,
    ambiguity_weight: float = 1.0,
    schedule_ambiguity_weight: bool = True,
    responsibility_mode: str = "standard",
    precision_old_confidence_init: float = 5.0,
    sensor_bypass: bool = True,
) -> nn.Module:
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
        enable_composition=enable_composition,
        enable_attractor=enable_attractor,
        center_ambiguity=center_ambiguity,
        ambiguity_weight=ambiguity_weight,
        schedule_ambiguity_weight=schedule_ambiguity_weight,
        responsibility_mode=responsibility_mode,
        precision_old_confidence_init=precision_old_confidence_init,
        sensor_bypass=sensor_bypass,
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
    _safe_run(
        "probe_thresholded_null",
        args,
        "mechanism_probe",
        "thresholded_null",
        "synthetic_probe_tensors",
        lambda: _run_named_mechanism_probe("probe_thresholded_null", "all_noise_should_choose_null", args, device, args.seed + 6),
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
    _safe_run(
        "ablation_sigmoid_gate_mean",
        args,
        "mechanism_probe",
        "sigmoid_gate_mean",
        "synthetic_probe_tensors",
        lambda: _run_mechanism_probe("ablation_sigmoid_gate_mean", args, device, args.seed + 24, False, False, "sigmoid"),
    )
    _safe_run(
        "ablation_thresholded_null",
        args,
        "mechanism_probe",
        "responsibility_thresholded_null",
        "synthetic_probe_tensors",
        lambda: _run_mechanism_probe("ablation_thresholded_null", args, device, args.seed + 25, True, True, "precision", "thresholded_null"),
    )
    probe_names = [
        "strong_signal_vs_many_weak_noise",
        "all_noise_should_choose_null",
        "conflicting_neighbors_increase_resistance",
        "old_state_confident_new_evidence_weak_should_not_update",
        "old_state_weak_new_evidence_strong_should_update",
        "composition_requires_consistent_children",
        "attractor_should_not_collapse",
    ]
    for probe_index, probe_name in enumerate(probe_names):
        run_id = f"probe_{probe_name}"
        _safe_run(
            run_id,
            args,
            "mechanism_probe",
            probe_name,
            "synthetic_probe_tensors",
            lambda run_id=run_id, probe_name=probe_name, probe_index=probe_index: _run_named_mechanism_probe(
                run_id,
                probe_name,
                args,
                device,
                args.seed + 60 + probe_index,
            ),
        )

    image_runs = [
        ("ablation_image_cnn_eml", "cnn_eml", lambda: _build_cnn_model(args.image_size), True, args.seed + 30),
        ("ablation_image_pure_eml", "pure_eml", lambda: _build_old_image_model("pure_eml", args.image_size), True, args.seed + 31),
        ("ablation_image_pure_eml_v2", "pure_eml_v2", lambda: _build_old_image_model("pure_eml_v2", args.image_size), True, args.seed + 32),
        ("ablation_image_eff_attr4", "EfficientEMLImageClassifier_attr4", lambda: _build_efficient_image(4, 3), True, args.seed + 33),
        ("ablation_image_eff_attr8", "EfficientEMLImageClassifier_attr8", lambda: _build_efficient_image(8, 3), True, args.seed + 34),
        ("ablation_image_eff_window5", "EfficientEMLImageClassifier_window5", lambda: _build_efficient_image(4, 5), True, args.seed + 35),
        ("ablation_image_eff_no_warmup", "EfficientEMLImageClassifier_no_warmup", lambda: _build_efficient_image(4, 3), False, args.seed + 36),
        ("ablation_no_composition", "EfficientEMLImageClassifier_no_composition", lambda: _build_efficient_image(4, 3, enable_composition=False), True, args.seed + 37),
        ("ablation_no_attractor", "EfficientEMLImageClassifier_no_attractor", lambda: _build_efficient_image(1, 3, enable_attractor=False), True, args.seed + 38),
        (
            "ablation_head_without_ambiguity",
            "prototype_head_without_ambiguity",
            lambda: _build_efficient_image(4, 3, center_ambiguity=False, ambiguity_weight=0.0, schedule_ambiguity_weight=False),
            True,
            args.seed + 39,
        ),
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
    text_runs = [
        ("ablation_text_window8_baseline", "EfficientEMLTextEncoder_window8", lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8), args.seed + 40),
        (
            "ablation_text_window8_thresholded_null",
            "EfficientEMLTextEncoder_window8_thresholded_null",
            lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, responsibility_mode="thresholded_null"),
            args.seed + 41,
        ),
        (
            "ablation_text_window8_precision_identity",
            "EfficientEMLTextEncoder_window8_precision_identity",
            lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, precision_old_confidence_init=5.0),
            args.seed + 42,
        ),
        (
            "ablation_text_window8_no_chunk",
            "EfficientEMLTextEncoder_window8_no_chunk",
            lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, enable_composition=False, enable_attractor=False),
            args.seed + 43,
        ),
        (
            "ablation_text_window8_chunk_no_attractor",
            "EfficientEMLTextEncoder_window8_chunk_no_attractor",
            lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, enable_composition=True, enable_attractor=False),
            args.seed + 44,
        ),
        (
            "ablation_text_window8_chunk_attractor",
            "EfficientEMLTextEncoder_window8_chunk_attractor",
            lambda: EfficientTextLM(len(vocab), vocab.pad_id, window_size=8, enable_composition=True, enable_attractor=True),
            args.seed + 45,
        ),
    ]
    for run_id, model_name, factory, seed in text_runs:
        _safe_run(
            run_id,
            args,
            "text_synthetic",
            model_name,
            "SyntheticTextEnergyDataset",
            lambda run_id=run_id, model_name=model_name, factory=factory, seed=seed: _train_text_model(
                run_id,
                model_name,
                factory(),
                args,
                device,
                seed,
            ),
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
    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
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
