from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


FP32_DTYPE = torch.float32


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus inverse expects a positive value")
    return value + math.log(-math.expm1(-value))


def _reset_linear(layer: nn.Linear) -> None:
    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _coerce_warmup_eta(warmup_eta: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(warmup_eta):
        eta = warmup_eta.to(device=reference.device, dtype=FP32_DTYPE)
    else:
        eta = torch.tensor(warmup_eta, device=reference.device, dtype=FP32_DTYPE)
    return eta.clamp(0.0, 1.0)


def _to_fp32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=FP32_DTYPE)


def _restore_dtype(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=reference.dtype)


class EMLUnit(nn.Module):
    """Stable EML primitive with fp32 islands for exp/log/softplus."""

    def __init__(
        self,
        dim: int,
        clip_value: float = 3.0,
        init_gamma: float = 0.1,
        init_lambda: float = 1.0,
        init_bias: float = 0.0,
        learnable_lambda: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if clip_value <= 0.0:
            raise ValueError("clip_value must be positive")
        if init_gamma <= 0.0:
            raise ValueError("init_gamma must be positive")
        if init_lambda <= 0.0:
            raise ValueError("init_lambda must be positive")

        self.dim = dim
        self.clip_value = float(clip_value)
        self.log_gamma = nn.Parameter(torch.tensor(math.log(init_gamma), dtype=FP32_DTYPE))

        raw_lambda = torch.tensor(inverse_softplus(init_lambda), dtype=FP32_DTYPE)
        if learnable_lambda:
            self.raw_lambda = nn.Parameter(raw_lambda)
        else:
            self.register_buffer("raw_lambda", raw_lambda)

        self.bias = nn.Parameter(torch.full((dim,), float(init_bias), dtype=FP32_DTYPE))
        rho0 = math.log1p(F.softplus(torch.tensor(0.0, dtype=FP32_DTYPE)).item())
        self.register_buffer("rho0", torch.tensor(rho0, dtype=FP32_DTYPE))

    def softclip(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = _to_fp32(x)
        clip_value = torch.tensor(self.clip_value, device=x_fp32.device, dtype=FP32_DTYPE)
        return clip_value * torch.tanh(x_fp32 / clip_value)

    def resistance_transform(self, resistance: torch.Tensor) -> torch.Tensor:
        resistance_fp32 = _to_fp32(resistance)
        rho0 = self.rho0.to(device=resistance.device, dtype=FP32_DTYPE)
        softplus_fp32 = F.softplus(resistance_fp32)
        return torch.log1p(softplus_fp32) - rho0

    def compute(
        self,
        drive: torch.Tensor,
        resistance: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor | torch.dtype]:
        if drive.shape != resistance.shape:
            raise ValueError("drive and resistance must have the same shape")
        if drive.size(-1) != self.dim:
            raise ValueError("drive/resistance last dimension does not match this EMLUnit")

        input_dtype = drive.dtype
        clipped_drive_fp32 = self.softclip(drive)
        centered_resistance_fp32 = self.resistance_transform(resistance)
        eta_fp32 = _coerce_warmup_eta(warmup_eta, clipped_drive_fp32)

        exp_part_fp32 = torch.expm1(clipped_drive_fp32)
        drive_part_fp32 = torch.lerp(clipped_drive_fp32, exp_part_fp32, eta_fp32)

        lambda_fp32 = F.softplus(self.raw_lambda).to(device=drive.device, dtype=FP32_DTYPE)
        gamma_fp32 = torch.exp(self.log_gamma).to(device=drive.device, dtype=FP32_DTYPE)
        bias_fp32 = self.bias.to(device=drive.device, dtype=FP32_DTYPE)

        energy_fp32 = gamma_fp32 * (drive_part_fp32 - lambda_fp32 * centered_resistance_fp32) + bias_fp32
        energy = energy_fp32.to(dtype=input_dtype)

        return {
            "drive": drive,
            "resistance": resistance,
            "clipped_drive": clipped_drive_fp32.to(dtype=input_dtype),
            "centered_resistance": centered_resistance_fp32.to(dtype=input_dtype),
            "energy": energy,
            "drive_fp32": _to_fp32(drive),
            "resistance_fp32": _to_fp32(resistance),
            "clipped_drive_fp32": clipped_drive_fp32,
            "centered_resistance_fp32": centered_resistance_fp32,
            "energy_fp32": energy_fp32,
            "warmup_eta_fp32": eta_fp32,
            "gamma_fp32": gamma_fp32,
            "lambda_fp32": lambda_fp32,
            "internal_dtype": energy_fp32.dtype,
        }

    def forward(
        self,
        drive: torch.Tensor,
        resistance: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        return self.compute(drive, resistance, warmup_eta=warmup_eta)["energy"]  # type: ignore[index]


class _BaseEMLGate(nn.Module):
    gate_bias: float

    def __init__(
        self,
        dim: int,
        clip_value: float = 3.0,
        temperature: float = 1.0,
        init_gamma: float = 0.1,
        init_lambda: float = 1.0,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.eml = EMLUnit(
            dim=dim,
            clip_value=clip_value,
            init_gamma=init_gamma,
            init_lambda=init_lambda,
            init_bias=init_bias,
        )
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=FP32_DTYPE))

    def forward(
        self,
        drive: torch.Tensor,
        resistance: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor | torch.dtype]:
        diagnostics = self.eml.compute(drive, resistance, warmup_eta=warmup_eta)
        temperature_fp32 = self.temperature.to(device=drive.device, dtype=FP32_DTYPE)
        gate_fp32 = torch.sigmoid(diagnostics["energy_fp32"] / temperature_fp32)  # type: ignore[index]
        gate = _restore_dtype(gate_fp32, drive)
        diagnostics["gate"] = gate
        diagnostics["gate_fp32"] = gate_fp32
        return diagnostics


class EMLGate(_BaseEMLGate):
    """General sEML gate."""

    def __init__(
        self,
        dim: int,
        clip_value: float = 3.0,
        temperature: float = 1.0,
        init_gamma: float = 0.1,
        init_lambda: float = 1.0,
        init_bias: float = -1.0,
    ) -> None:
        super().__init__(
            dim=dim,
            clip_value=clip_value,
            temperature=temperature,
            init_gamma=init_gamma,
            init_lambda=init_lambda,
            init_bias=init_bias,
        )


class EMLUpdateGate(_BaseEMLGate):
    """sEML gate specialized for recurrent state updates."""

    def __init__(
        self,
        dim: int,
        clip_value: float = 3.0,
        temperature: float = 1.0,
        init_gamma: float = 0.1,
        init_lambda: float = 1.0,
        init_bias: float = -1.0,
    ) -> None:
        super().__init__(
            dim=dim,
            clip_value=clip_value,
            temperature=temperature,
            init_gamma=init_gamma,
            init_lambda=init_lambda,
            init_bias=init_bias,
        )


class EMLMessageGate(_BaseEMLGate):
    """sEML gate specialized for message-passing edges."""

    def __init__(
        self,
        dim: int,
        clip_value: float = 3.0,
        temperature: float = 1.0,
        init_gamma: float = 0.1,
        init_lambda: float = 1.0,
        init_bias: float = -0.5,
    ) -> None:
        super().__init__(
            dim=dim,
            clip_value=clip_value,
            temperature=temperature,
            init_gamma=init_gamma,
            init_lambda=init_lambda,
            init_bias=init_bias,
        )


class EMLScore(nn.Module):
    """Turn sEML energy into normalized scores."""

    def __init__(
        self,
        dim: int,
        clip_value: float = 3.0,
        temperature: float = 1.0,
        init_gamma: float = 0.1,
        init_lambda: float = 1.0,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.eml = EMLUnit(
            dim=dim,
            clip_value=clip_value,
            init_gamma=init_gamma,
            init_lambda=init_lambda,
            init_bias=init_bias,
        )
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=FP32_DTYPE))

    def forward(
        self,
        drive: torch.Tensor,
        resistance: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor | torch.dtype]:
        diagnostics = self.eml.compute(drive, resistance, warmup_eta=warmup_eta)
        temperature_fp32 = self.temperature.to(device=drive.device, dtype=FP32_DTYPE)
        probs_fp32 = torch.softmax(diagnostics["energy_fp32"] / temperature_fp32, dim=-1)  # type: ignore[index]
        diagnostics["score"] = diagnostics["energy"]
        diagnostics["energy"] = diagnostics["score"]
        diagnostics["probs"] = _restore_dtype(probs_fp32, drive)
        diagnostics["probs_fp32"] = probs_fp32
        return diagnostics


class EMLActivationBudget(nn.Module):
    """Convert energy into activations with optional sparse budget pressure."""

    def __init__(
        self,
        temperature: float = 1.0,
        target_rate: float | None = None,
        budget_weight: float = 1.0,
        soft_sparse: bool = False,
        sparse_threshold: float = 0.5,
        sparse_temperature: float = 0.25,
        top_k: int | None = None,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if target_rate is not None and not 0.0 <= target_rate <= 1.0:
            raise ValueError("target_rate must be in [0, 1]")
        if budget_weight < 0.0:
            raise ValueError("budget_weight must be non-negative")
        if sparse_temperature <= 0.0:
            raise ValueError("sparse_temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=FP32_DTYPE))
        self.target_rate = target_rate
        self.budget_weight = float(budget_weight)
        self.soft_sparse = bool(soft_sparse)
        self.sparse_threshold = float(sparse_threshold)
        self.sparse_temperature = float(sparse_temperature)
        self.top_k = top_k
        self.eps = float(eps)

    def forward(
        self,
        energy: torch.Tensor,
        mask: torch.Tensor | None = None,
        top_k: int | None = None,
        target_rate: float | torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if energy.ndim == 0:
            raise ValueError("energy must have at least one dimension")
        if mask is not None and mask.shape != energy.shape:
            raise ValueError("mask must match energy shape")

        energy_fp32 = _to_fp32(energy)
        temperature_fp32 = self.temperature.to(device=energy.device, dtype=FP32_DTYPE)
        activation_fp32 = torch.sigmoid(energy_fp32 / temperature_fp32)

        if self.soft_sparse:
            sparse_gate = torch.sigmoid(
                (activation_fp32 - self.sparse_threshold) / self.sparse_temperature
            )
            activation_fp32 = activation_fp32 * sparse_gate

        mask_bool: torch.Tensor | None = None
        if mask is not None:
            mask_bool = mask.to(device=energy.device, dtype=torch.bool)
            activation_fp32 = activation_fp32.masked_fill(~mask_bool, 0.0)

        effective_top_k = top_k if top_k is not None else self.top_k
        topk_mask = torch.ones_like(activation_fp32, dtype=torch.bool)
        if effective_top_k is not None:
            k = min(int(effective_top_k), activation_fp32.size(-1))
            ranked = activation_fp32
            if mask_bool is not None:
                ranked = ranked.masked_fill(~mask_bool, float("-inf"))
            topk_indices = ranked.topk(k=k, dim=-1).indices
            topk_mask = torch.zeros_like(activation_fp32, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_indices, True)
            if mask_bool is not None:
                topk_mask = topk_mask & mask_bool
            activation_fp32 = activation_fp32 * topk_mask.to(dtype=FP32_DTYPE)
        elif mask_bool is not None:
            topk_mask = mask_bool

        if mask_bool is not None:
            valid_count = mask_bool.to(dtype=FP32_DTYPE).sum().clamp_min(1.0)
            active_rate = activation_fp32.sum() / valid_count
            entropy_values = _binary_entropy(activation_fp32.clamp(self.eps, 1.0 - self.eps))
            entropy = (entropy_values * mask_bool.to(dtype=FP32_DTYPE)).sum() / valid_count
        else:
            active_rate = activation_fp32.mean()
            entropy = _binary_entropy(activation_fp32.clamp(self.eps, 1.0 - self.eps)).mean()

        budget_target = target_rate if target_rate is not None else self.target_rate
        if budget_target is None:
            budget_loss = activation_fp32.new_zeros(())
        elif torch.is_tensor(budget_target):
            target_fp32 = budget_target.to(device=energy.device, dtype=FP32_DTYPE)
            budget_loss = self.budget_weight * (active_rate - target_fp32).square()
        else:
            target_fp32 = torch.tensor(float(budget_target), device=energy.device, dtype=FP32_DTYPE)
            budget_loss = self.budget_weight * (active_rate - target_fp32).square()

        activation = _restore_dtype(activation_fp32, energy)
        return {
            "activation": activation,
            "activation_fp32": activation_fp32,
            "budget_loss": budget_loss,
            "entropy": entropy,
            "active_rate": active_rate,
            "topk_mask": topk_mask,
            "gate_mass": activation_fp32.sum(dim=-1),
        }


def _binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    return -(probability * torch.log(probability) + (1.0 - probability) * torch.log(1.0 - probability))


class EMLResponsibility(nn.Module):
    """Convert sEML energy into stable responsibility weights."""

    def __init__(
        self,
        temperature: float = 1.0,
        use_null: bool = True,
        null_logit: float = 0.0,
        learnable_null: bool = False,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=FP32_DTYPE))
        self.use_null = bool(use_null)
        self.eps = float(eps)
        null_value = torch.tensor(float(null_logit), dtype=FP32_DTYPE)
        if learnable_null:
            self.null_logit = nn.Parameter(null_value)
        else:
            self.register_buffer("null_logit", null_value)

    def forward(
        self,
        energy: torch.Tensor,
        mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor | None = None,
        use_null: bool | None = None,
    ) -> Dict[str, torch.Tensor | None]:
        if energy.ndim == 0:
            raise ValueError("energy must have at least one dimension")
        if mask is not None and mask.shape != energy.shape:
            raise ValueError("mask must match energy shape")

        original_dtype = energy.dtype
        energy_fp32 = _to_fp32(energy)
        if temperature is None:
            temperature_fp32 = self.temperature.to(device=energy.device, dtype=FP32_DTYPE)
        elif torch.is_tensor(temperature):
            temperature_fp32 = temperature.to(device=energy.device, dtype=FP32_DTYPE)
        else:
            temperature_fp32 = torch.tensor(float(temperature), device=energy.device, dtype=FP32_DTYPE)
        temperature_fp32 = temperature_fp32.clamp_min(self.eps)

        logits = energy_fp32 / temperature_fp32
        mask_bool: torch.Tensor | None = None
        if mask is not None:
            mask_bool = mask.to(device=energy.device, dtype=torch.bool)
            logits = logits.masked_fill(~mask_bool, torch.finfo(FP32_DTYPE).min)

        effective_use_null = self.use_null if use_null is None else bool(use_null)
        if effective_use_null:
            null_logit = self.null_logit.to(device=energy.device, dtype=FP32_DTYPE)
            null_shape = (*logits.shape[:-1], 1)
            null_logits = null_logit.expand(null_shape)
            all_logits = torch.cat([logits, null_logits], dim=-1)
            weights = torch.softmax(all_logits, dim=-1)
            neighbor_weights_fp32 = weights[..., :-1]
            null_weight_fp32 = weights[..., -1]
        else:
            all_logits = logits
            if mask_bool is not None:
                valid = mask_bool.any(dim=-1, keepdim=True)
                safe_logits = torch.where(valid, logits, torch.zeros_like(logits))
                neighbor_weights_fp32 = torch.softmax(safe_logits, dim=-1)
                neighbor_weights_fp32 = torch.where(mask_bool, neighbor_weights_fp32, torch.zeros_like(neighbor_weights_fp32))
                normalizer = neighbor_weights_fp32.sum(dim=-1, keepdim=True)
                neighbor_weights_fp32 = torch.where(
                    normalizer > 0.0,
                    neighbor_weights_fp32 / normalizer.clamp_min(self.eps),
                    torch.zeros_like(neighbor_weights_fp32),
                )
            else:
                neighbor_weights_fp32 = torch.softmax(logits, dim=-1)
            null_weight_fp32 = None

        if mask_bool is not None:
            neighbor_weights_fp32 = neighbor_weights_fp32.masked_fill(~mask_bool, 0.0)

        weight_mass_fp32 = neighbor_weights_fp32.sum(dim=-1)
        if null_weight_fp32 is None:
            update_strength_fp32 = torch.ones_like(weight_mass_fp32)
        else:
            update_strength_fp32 = (1.0 - null_weight_fp32).clamp(0.0, 1.0)

        entropy_terms = neighbor_weights_fp32.clamp_min(self.eps)
        entropy_fp32 = -(neighbor_weights_fp32 * entropy_terms.log()).sum(dim=-1)
        if null_weight_fp32 is not None:
            null_terms = null_weight_fp32.clamp_min(self.eps)
            entropy_fp32 = entropy_fp32 - null_weight_fp32 * null_terms.log()
        max_weight_fp32 = neighbor_weights_fp32.max(dim=-1).values if neighbor_weights_fp32.size(-1) else weight_mass_fp32

        neighbor_weights = neighbor_weights_fp32.to(dtype=original_dtype)
        null_weight = null_weight_fp32.to(dtype=original_dtype) if null_weight_fp32 is not None else None
        update_strength = update_strength_fp32.to(dtype=original_dtype)
        return {
            "neighbor_weights": neighbor_weights,
            "neighbor_weights_fp32": neighbor_weights_fp32,
            "null_weight": null_weight,
            "null_weight_fp32": null_weight_fp32,
            "update_strength": update_strength,
            "update_strength_fp32": update_strength_fp32,
            "entropy": entropy_fp32,
            "max_weight": max_weight_fp32,
            "max_responsibility": max_weight_fp32,
            "weight_mass": weight_mass_fp32,
            "logits": all_logits.to(dtype=original_dtype),
            "logits_fp32": all_logits,
        }


class EMLPrecisionUpdate(nn.Module):
    """Precision-style EML state update with a compatibility sigmoid mode."""

    def __init__(
        self,
        mode: str = "precision",
        eps: float = 1.0e-6,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"precision", "sigmoid"}:
            raise ValueError("mode must be 'precision' or 'sigmoid'")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive")
        self.mode = mode
        self.eps = float(eps)
        self.register_buffer("gate_temperature", torch.tensor(float(gate_temperature), dtype=FP32_DTYPE))

    def forward(
        self,
        state: torch.Tensor,
        candidate: torch.Tensor,
        new_energy: torch.Tensor,
        old_confidence: torch.Tensor | None = None,
        update_strength: torch.Tensor | None = None,
        mode: str | None = None,
    ) -> Dict[str, torch.Tensor]:
        if state.shape != candidate.shape:
            raise ValueError("state and candidate must have the same shape")
        effective_mode = self.mode if mode is None else mode
        if effective_mode not in {"precision", "sigmoid"}:
            raise ValueError("mode must be 'precision' or 'sigmoid'")

        state_dtype = state.dtype
        new_energy_fp32 = _to_fp32(new_energy)
        if old_confidence is None:
            old_confidence_fp32 = torch.zeros_like(new_energy_fp32)
        else:
            old_confidence_fp32 = _to_fp32(old_confidence)

        if effective_mode == "precision":
            new_precision_fp32 = F.softplus(new_energy_fp32) + self.eps
            old_precision_fp32 = F.softplus(old_confidence_fp32) + self.eps
            update_gate_fp32 = new_precision_fp32 / (new_precision_fp32 + old_precision_fp32).clamp_min(self.eps)
        else:
            temperature_fp32 = self.gate_temperature.to(device=state.device, dtype=FP32_DTYPE).clamp_min(self.eps)
            new_precision_fp32 = F.softplus(new_energy_fp32) + self.eps
            old_precision_fp32 = F.softplus(old_confidence_fp32) + self.eps
            update_gate_fp32 = torch.sigmoid(new_energy_fp32 / temperature_fp32)

        if update_strength is not None:
            update_strength_fp32 = _to_fp32(update_strength)
            while update_strength_fp32.ndim < update_gate_fp32.ndim:
                update_strength_fp32 = update_strength_fp32.unsqueeze(-1)
            update_gate_fp32 = update_gate_fp32 * update_strength_fp32

        while update_gate_fp32.ndim < state.ndim:
            update_gate_fp32 = update_gate_fp32.unsqueeze(-1)
        update_gate_fp32 = update_gate_fp32.clamp(0.0, 1.0)
        updated_fp32 = _to_fp32(state) + update_gate_fp32 * (_to_fp32(candidate) - _to_fp32(state))
        updated = updated_fp32.to(dtype=state_dtype)
        return {
            "state": updated,
            "updated": updated,
            "updated_state": updated,
            "update_gate": update_gate_fp32.to(dtype=state_dtype),
            "update_gate_fp32": update_gate_fp32,
            "new_precision": new_precision_fp32.to(dtype=state_dtype),
            "old_precision": old_precision_fp32.to(dtype=state_dtype),
            "new_precision_fp32": new_precision_fp32,
            "old_precision_fp32": old_precision_fp32,
        }


class EMLBank(nn.Module):
    """Project features into a bank of sEML units and mix them back out."""

    def __init__(
        self,
        input_dim: int,
        bank_dim: int,
        output_dim: int | None = None,
        clip_value: float = 3.0,
        dropout: float = 0.0,
        init_gamma: float = 0.1,
        init_lambda: float = 1.0,
        init_bias: float = -1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if bank_dim <= 0:
            raise ValueError("bank_dim must be positive")

        self.input_dim = input_dim
        self.bank_dim = bank_dim
        self.output_dim = output_dim or input_dim

        self.norm = nn.LayerNorm(input_dim)
        self.drive = nn.Linear(input_dim, bank_dim)
        self.resistance = nn.Linear(input_dim, bank_dim)
        self.gate = EMLGate(
            dim=bank_dim,
            clip_value=clip_value,
            init_gamma=init_gamma,
            init_lambda=init_lambda,
            init_bias=init_bias,
        )
        self.dropout = nn.Dropout(dropout)
        self.mix = nn.Linear(bank_dim, self.output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_linear(self.drive)
        _reset_linear(self.resistance)
        _reset_linear(self.mix)

    def forward(
        self,
        x: torch.Tensor,
        warmup_eta: float | torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor | torch.dtype]:
        if x.ndim < 2 or x.size(-1) != self.input_dim:
            raise ValueError("x must have shape [..., input_dim]")

        normalized = self.norm(x)
        drive = self.drive(normalized)
        resistance = self.resistance(normalized)
        gate_out = self.gate(drive, resistance, warmup_eta=warmup_eta)
        bank = gate_out["gate"]  # type: ignore[index]
        output = self.mix(self.dropout(bank))  # type: ignore[arg-type]

        return {
            "output": output,
            "bank": bank,
            "drive": gate_out["drive"],
            "resistance": gate_out["resistance"],
            "energy": gate_out["energy"],
            "gate": gate_out["gate"],
            "clipped_drive": gate_out["clipped_drive"],
            "centered_resistance": gate_out["centered_resistance"],
        }


__all__ = [
    "EMLActivationBudget",
    "EMLBank",
    "EMLGate",
    "EMLMessageGate",
    "EMLPrecisionUpdate",
    "EMLResponsibility",
    "EMLScore",
    "EMLUnit",
    "EMLUpdateGate",
    "FP32_DTYPE",
    "_coerce_warmup_eta",
    "_reset_linear",
    "inverse_softplus",
]
