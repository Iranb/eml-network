from __future__ import annotations

from eml_mnist.merc_toy_tasks import TOY_MODELS, TOY_TASKS


def test_toy_task_batch_generation() -> None:
    for task in TOY_TASKS.values():
        batch = task(batch_size=8, input_dim=8, seed=0)
        assert batch.inputs.shape == (8, 8)
        assert batch.labels.shape == (8,)


def test_toy_models_forward() -> None:
    batch = TOY_TASKS["conjunctive"](batch_size=8, input_dim=8, seed=0)
    for ctor in TOY_MODELS.values():
        model = ctor(8)
        out = model(batch.inputs)
        assert out["logits"].shape == (8, 2)
