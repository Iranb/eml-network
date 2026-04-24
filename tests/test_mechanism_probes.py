from __future__ import annotations

import math

from eml_mnist import MECHANISM_NAMES, PROBE_NAMES, run_mechanism_probe


def test_mechanism_probe_thresholded_null_is_nontrivial() -> None:
    out = run_mechanism_probe("all_noise_should_choose_null", "thresholded_null", seed=1)

    assert out["metrics"]["accuracy"] == 1.0
    assert out["metrics"]["null_weight"] > 0.7


def test_mechanism_probe_strong_signal_selects_neighbor() -> None:
    out = run_mechanism_probe("one_strong_signal_many_weak_noise", "thresholded_null", seed=2)

    assert out["metrics"]["accuracy"] == 1.0
    assert out["metrics"]["max_responsibility"] > 0.7


def test_mechanism_probe_precision_cases_are_finite() -> None:
    low = run_mechanism_probe(
        "old_state_confident_new_evidence_weak_should_not_update",
        "precision_update",
        seed=3,
    )
    high = run_mechanism_probe(
        "old_state_weak_new_evidence_strong_should_update",
        "precision_update",
        seed=4,
    )

    assert low["metrics"]["accuracy"] == 1.0
    assert high["metrics"]["accuracy"] == 1.0
    assert low["metrics"]["update_gate"] < high["metrics"]["update_gate"]


def test_all_probe_names_and_mechanisms_smoke() -> None:
    for probe_name in PROBE_NAMES:
        out = run_mechanism_probe(probe_name, MECHANISM_NAMES[0], seed=5)
        for value in out["metrics"].values():
            if isinstance(value, float):
                assert math.isfinite(value) or math.isnan(value)
