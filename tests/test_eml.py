from __future__ import annotations

import torch

from eml_mnist.primitives import (
    EMLActivationBudget,
    EMLBank,
    EMLGate,
    EMLMessageGate,
    EMLScore,
    EMLUnit,
    EMLUpdateGate,
)


def test_eml_fp32_island() -> None:
    unit = EMLUnit(dim=2, clip_value=2.0)
    drive = torch.tensor([[1000.0, -1000.0]], dtype=torch.bfloat16, requires_grad=True)
    resistance = torch.tensor([[-1000.0, 1000.0]], dtype=torch.bfloat16, requires_grad=True)

    diagnostics = unit.compute(drive, resistance, warmup_eta=0.75)

    assert diagnostics["drive_fp32"].dtype == torch.float32
    assert diagnostics["clipped_drive_fp32"].dtype == torch.float32
    assert diagnostics["centered_resistance_fp32"].dtype == torch.float32
    assert diagnostics["energy_fp32"].dtype == torch.float32
    assert diagnostics["energy"].dtype == torch.bfloat16
    assert torch.isfinite(diagnostics["energy_fp32"]).all()


def test_eml_score_energy_alias() -> None:
    score = EMLScore(dim=4)
    drive = torch.randn(2, 4)
    resistance = torch.randn(2, 4)

    out = score(drive, resistance, warmup_eta=0.5)

    assert set(["score", "energy", "probs", "drive", "resistance"]).issubset(out.keys())
    assert torch.allclose(out["score"], out["energy"])


def test_eml_finite_gradients() -> None:
    unit = EMLUnit(dim=4, clip_value=3.0)
    drive = torch.randn(3, 4, requires_grad=True)
    resistance = torch.randn(3, 4, requires_grad=True)

    energy = unit(drive, resistance, warmup_eta=0.7)
    loss = energy.square().mean()
    loss.backward()

    assert drive.grad is not None and torch.isfinite(drive.grad).all()
    assert resistance.grad is not None and torch.isfinite(resistance.grad).all()


def test_eml_warmup_behavior() -> None:
    unit = EMLUnit(dim=3, clip_value=2.5, init_gamma=0.3, init_lambda=1.7, init_bias=-0.2)
    drive = torch.tensor([[0.25, -0.5, 1.75]], dtype=torch.float32)
    resistance = torch.tensor([[-1.0, 0.0, 2.0]], dtype=torch.float32)

    linear = unit.compute(drive, resistance, warmup_eta=0.0)["energy_fp32"]
    full = unit.compute(drive, resistance, warmup_eta=1.0)["energy_fp32"]
    middle = unit.compute(drive, resistance, warmup_eta=0.5)["energy_fp32"]

    assert not torch.allclose(linear, full)
    assert torch.all((middle >= torch.minimum(linear, full) - 1.0e-6) & (middle <= torch.maximum(linear, full) + 1.0e-6))


def test_eml_activation_budget() -> None:
    budget = EMLActivationBudget(target_rate=0.5, budget_weight=0.25, soft_sparse=True, top_k=2)
    energy = torch.tensor([[2.0, 1.0, -1.0, -2.0], [0.5, 0.25, 0.0, -0.25]], requires_grad=True)

    out = budget(energy)

    assert out["activation"].shape == energy.shape
    assert torch.isfinite(out["activation"]).all()
    assert torch.isfinite(out["budget_loss"])
    assert torch.isfinite(out["entropy"])
    assert torch.isfinite(out["active_rate"])
    assert out["topk_mask"].sum(dim=-1).le(2).all()
    out["activation"].sum().backward()
    assert energy.grad is not None and torch.isfinite(energy.grad).all()


def test_no_nan_inf_reasonable_inputs() -> None:
    drive = torch.linspace(-8.0, 8.0, steps=12).view(3, 4)
    resistance = torch.linspace(8.0, -8.0, steps=12).view(3, 4)
    unit = EMLUnit(dim=4)
    score = EMLScore(dim=4)
    gate = EMLGate(dim=4)
    budget = EMLActivationBudget()

    unit_out = unit.compute(drive, resistance)
    score_out = score(drive, resistance)
    gate_out = gate(drive, resistance)
    budget_out = budget(score_out["energy"])

    for tensor in (
        unit_out["energy"],
        score_out["score"],
        score_out["probs"],
        gate_out["gate"],
        budget_out["activation"],
    ):
        assert torch.isfinite(tensor).all()


def test_eml_score_has_energy_alias() -> None:
    score = EMLScore(dim=4)
    drive = torch.randn(2, 4)
    resistance = torch.randn(2, 4)

    out = score(drive, resistance, warmup_eta=0.5)

    assert "score" in out
    assert "energy" in out
    assert torch.allclose(out["score"], out["energy"])


def test_eml_fp32_island_no_nan() -> None:
    unit = EMLUnit(dim=2, clip_value=2.0)
    drive = torch.tensor([[1000.0, -1000.0]], dtype=torch.bfloat16, requires_grad=True)
    resistance = torch.tensor([[-1000.0, 1000.0]], dtype=torch.bfloat16, requires_grad=True)

    diagnostics = unit.compute(drive, resistance, warmup_eta=0.75)

    assert diagnostics["energy_fp32"].dtype == torch.float32
    assert diagnostics["energy"].dtype == torch.bfloat16
    assert torch.isfinite(diagnostics["energy_fp32"]).all()
    diagnostics["energy"].float().sum().backward()
    assert drive.grad is not None and torch.isfinite(drive.grad.float()).all()
    assert resistance.grad is not None and torch.isfinite(resistance.grad.float()).all()


def test_eml_score_shapes() -> None:
    score = EMLScore(dim=5)
    drive = torch.randn(2, 3, 5)
    resistance = torch.randn(2, 3, 5)

    out = score(drive, resistance)

    assert out["score"].shape == (2, 3, 5)
    assert out["energy"].shape == (2, 3, 5)
    assert out["probs"].shape == (2, 3, 5)
    assert out["drive"].shape == (2, 3, 5)
    assert out["resistance"].shape == (2, 3, 5)
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_stable_eml_has_no_nan_or_inf_and_finite_gradients() -> None:
    unit = EMLUnit(dim=4, clip_value=3.0)
    drive = torch.tensor([[-1.0e6, -1.0e3, 1.0e3, 1.0e6]], dtype=torch.float32, requires_grad=True)
    resistance = torch.tensor([[-1.0e6, -1.0e3, 1.0e3, 1.0e6]], dtype=torch.float32, requires_grad=True)

    energy = unit(drive, resistance, warmup_eta=1.0)
    assert torch.isfinite(energy).all()

    energy.sum().backward()
    assert torch.isfinite(drive.grad).all()
    assert torch.isfinite(resistance.grad).all()


def test_warmup_eta_behavior_matches_reference_formula() -> None:
    unit = EMLUnit(dim=3, clip_value=2.5, init_gamma=0.3, init_lambda=1.7, init_bias=-0.2)
    drive = torch.tensor([[0.25, -0.5, 1.75]], dtype=torch.float32)
    resistance = torch.tensor([[-1.0, 0.0, 2.0]], dtype=torch.float32)

    diagnostics = unit.compute(drive, resistance, warmup_eta=0.25)
    clipped = diagnostics["clipped_drive_fp32"]
    centered_resistance = diagnostics["centered_resistance_fp32"]
    gamma = diagnostics["gamma_fp32"]
    lam = diagnostics["lambda_fp32"]
    bias = unit.bias

    expected_linear = gamma * (clipped - lam * centered_resistance) + bias
    expected_full = gamma * (torch.expm1(clipped) - lam * centered_resistance) + bias
    expected_mid = gamma * (torch.lerp(clipped, torch.expm1(clipped), torch.tensor(0.25)) - lam * centered_resistance) + bias

    assert torch.allclose(unit.compute(drive, resistance, warmup_eta=0.0)["energy_fp32"], expected_linear, atol=1e-6)
    assert torch.allclose(unit.compute(drive, resistance, warmup_eta=1.0)["energy_fp32"], expected_full, atol=1e-6)
    assert torch.allclose(diagnostics["energy_fp32"], expected_mid, atol=1e-6)


def test_fp32_island_behavior_and_output_dtype_restore() -> None:
    unit = EMLUnit(dim=2, clip_value=3.0)
    drive = torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16)
    resistance = torch.tensor([[0.5, -0.5]], dtype=torch.bfloat16)

    diagnostics = unit.compute(drive, resistance, warmup_eta=0.5)

    assert diagnostics["internal_dtype"] == torch.float32
    assert diagnostics["drive_fp32"].dtype == torch.float32
    assert diagnostics["centered_resistance_fp32"].dtype == torch.float32
    assert diagnostics["energy_fp32"].dtype == torch.float32
    assert diagnostics["energy"].dtype == torch.bfloat16


def test_fp32_island_low_precision_path_has_finite_gradients() -> None:
    unit = EMLUnit(dim=3, clip_value=2.0)
    drive = torch.tensor([[20.0, -20.0, 0.5]], dtype=torch.bfloat16, requires_grad=True)
    resistance = torch.tensor([[-25.0, 25.0, 1.0]], dtype=torch.bfloat16, requires_grad=True)

    diagnostics = unit.compute(drive, resistance, warmup_eta=torch.tensor(0.75, dtype=torch.bfloat16))
    assert diagnostics["warmup_eta_fp32"].dtype == torch.float32
    assert diagnostics["clipped_drive_fp32"].dtype == torch.float32
    assert diagnostics["centered_resistance_fp32"].dtype == torch.float32
    assert diagnostics["energy_fp32"].dtype == torch.float32
    assert diagnostics["energy"].dtype == torch.bfloat16
    assert torch.isfinite(diagnostics["energy_fp32"]).all()

    diagnostics["energy"].float().sum().backward()
    assert drive.grad is not None
    assert resistance.grad is not None
    assert torch.isfinite(drive.grad.float()).all()
    assert torch.isfinite(resistance.grad.float()).all()


def test_eml_module_output_shapes_and_diagnostics() -> None:
    drive = torch.randn(2, 3, 5)
    resistance = torch.randn(2, 3, 5)

    gate = EMLGate(dim=5)
    update_gate = EMLUpdateGate(dim=5)
    message_gate = EMLMessageGate(dim=5)
    score = EMLScore(dim=5)
    bank = EMLBank(input_dim=6, bank_dim=5, output_dim=4)

    gate_out = gate(drive, resistance, warmup_eta=0.4)
    update_out = update_gate(drive, resistance, warmup_eta=0.4)
    message_out = message_gate(drive, resistance, warmup_eta=0.4)
    score_out = score(drive, resistance, warmup_eta=0.4)
    bank_out = bank(torch.randn(2, 3, 6), warmup_eta=0.4)

    for output in (gate_out, update_out, message_out):
        assert output["energy"].shape == (2, 3, 5)
        assert output["gate"].shape == (2, 3, 5)
        assert output["drive"].shape == (2, 3, 5)
        assert output["resistance"].shape == (2, 3, 5)
        assert torch.isfinite(output["gate"]).all()

    assert score_out["score"].shape == (2, 3, 5)
    assert score_out["energy"].shape == (2, 3, 5)
    assert score_out["probs"].shape == (2, 3, 5)
    assert torch.allclose(score_out["probs"].sum(dim=-1), torch.ones(2, 3), atol=1e-5)

    assert bank_out["output"].shape == (2, 3, 4)
    assert bank_out["gate"].shape == (2, 3, 5)
    assert bank_out["energy"].shape == (2, 3, 5)
