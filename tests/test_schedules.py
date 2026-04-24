from __future__ import annotations

from eml_mnist import StagedHardeningConfig, get_staged_hardening_values


def test_staged_hardening_values_progress() -> None:
    config = StagedHardeningConfig(
        warmup_steps=10,
        responsibility_temp_start=3.0,
        responsibility_temp_end=1.0,
        ambiguity_warmup_steps=10,
        null_threshold_start=2.0,
        null_threshold_end=0.5,
    )

    start = get_staged_hardening_values(0, 20, config)
    middle = get_staged_hardening_values(10, 20, config)
    end = get_staged_hardening_values(20, 20, config)

    assert start["warmup_eta"] == 0.0
    assert middle["warmup_eta"] == 1.0
    assert end["responsibility_temperature"] == 1.0
    assert start["responsibility_temperature"] > end["responsibility_temperature"]
    assert start["null_threshold"] > end["null_threshold"]


def test_staged_hardening_mapping_config() -> None:
    values = get_staged_hardening_values(5, 10, {"schedule": "cosine", "warmup_steps": 10})

    assert 0.0 < values["warmup_eta"] < 1.0
    assert "precision_update_threshold" in values
