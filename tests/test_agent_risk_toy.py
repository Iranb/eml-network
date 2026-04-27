from __future__ import annotations

from torch.utils.data import DataLoader

from eml_mnist.agent_risk_toy import AgentRiskToyDataset, agent_risk_collate


def test_agent_risk_toy_dataset_and_collate() -> None:
    dataset = AgentRiskToyDataset(size=5, num_actions=3, feature_dim=6, seed=7)
    sample = dataset[0]

    assert sample["features"].shape == (3, 6)
    assert sample["utility"].shape == (3,)
    assert sample["risk_target"].shape == (3,)
    assert 0 <= int(sample["target_action"]) < 3

    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=agent_risk_collate)))
    assert batch["features"].shape == (2, 3, 6)
    assert batch["target_action"].shape == (2,)
