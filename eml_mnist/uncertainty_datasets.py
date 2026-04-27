from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .image_datasets import CIFARCorruptionDataset, SyntheticShapeEnergyDataset
from .text_datasets import SyntheticTextEnergyDataset


CORRUPTION_TYPES = (
    "clean",
    "gaussian_noise",
    "salt_pepper_noise",
    "cutout_occlusion",
    "random_patch_occlusion",
    "blur",
    "background_clutter",
    "label_ambiguity",
    "mixed",
)


def _choose_corruption(corruption_type: str, generator: torch.Generator) -> str:
    if corruption_type != "mixed":
        return corruption_type
    choices = CORRUPTION_TYPES[:-1]
    return choices[int(torch.randint(0, len(choices), (1,), generator=generator).item())]


def _apply_salt_pepper(image: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    mask = torch.rand(image.shape, generator=generator, dtype=image.dtype) < (0.08 + 0.22 * severity)
    salt = torch.rand(image.shape, generator=generator, dtype=image.dtype) > 0.5
    return torch.where(mask, salt.to(dtype=image.dtype), image).clamp(0.0, 1.0)


def _apply_cutout(image: torch.Tensor, severity: float, generator: torch.Generator) -> tuple[torch.Tensor, float]:
    _, height, width = image.shape
    frac = min(0.75, 0.15 + 0.45 * severity)
    patch_h = max(1, int(round(height * frac)))
    patch_w = max(1, int(round(width * frac)))
    top = int(torch.randint(0, height - patch_h + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, width - patch_w + 1, (1,), generator=generator).item())
    out = image.clone()
    out[:, top : top + patch_h, left : left + patch_w] *= 0.05
    return out, float((patch_h * patch_w) / (height * width))


def _apply_blur(image: torch.Tensor, severity: float) -> torch.Tensor:
    kernel = 3 if severity < 0.7 else 5
    pad = kernel // 2
    return F.avg_pool2d(image.unsqueeze(0), kernel_size=kernel, stride=1, padding=pad).squeeze(0).clamp(0.0, 1.0)


class SyntheticShapeUncertaintyDataset(Dataset):
    """Synthetic image uncertainty dataset with paired corruption metadata."""

    def __init__(
        self,
        size: int,
        image_size: int = 32,
        seed: int = 0,
        target_type: str = "shape",
        corruption_type: str = "mixed",
        severity: float = 0.6,
        include_clean_pair: bool = True,
    ) -> None:
        if corruption_type not in CORRUPTION_TYPES:
            raise ValueError(f"corruption_type must be one of {CORRUPTION_TYPES}")
        self.size = size
        self.seed = seed
        self.corruption_type = corruption_type
        self.severity = float(max(0.0, min(1.0, severity)))
        self.include_clean_pair = include_clean_pair
        self.clean = SyntheticShapeEnergyDataset(
            size=size,
            image_size=image_size,
            seed=seed,
            target_type=target_type,
            include_background_clutter=False,
            include_mask=False,
            forced_noise_name="low",
            forced_occlusion_name="none",
            forced_clutter_flag=0,
        )
        self.clutter = SyntheticShapeEnergyDataset(
            size=size,
            image_size=image_size,
            seed=seed + 50_000,
            target_type=target_type,
            include_background_clutter=True,
            include_mask=False,
            forced_noise_name="low",
            forced_occlusion_name="none",
            forced_clutter_flag=1,
        )
        self.num_classes = self.clean.num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int | str]:
        generator = torch.Generator().manual_seed(self.seed + 100_000 + index)
        corruption = _choose_corruption(self.corruption_type, generator)
        base = self.clutter[index] if corruption == "background_clutter" else self.clean[index]
        clean_base = self.clean[index]
        image = base["image"].clone().float()
        clean_image = clean_base["image"].clone().float()
        noise_level = 0.0
        occlusion_level = 0.0
        blur_level = 0.0
        clutter_level = float(base.get("background_clutter", torch.tensor(0.0)))
        if corruption == "gaussian_noise":
            noise_level = self.severity
            image = (image + (0.05 + 0.35 * self.severity) * torch.randn(image.shape, generator=generator)).clamp(0.0, 1.0)
        elif corruption == "salt_pepper_noise":
            noise_level = self.severity
            image = _apply_salt_pepper(image, self.severity, generator)
        elif corruption in {"cutout_occlusion", "random_patch_occlusion"}:
            image, occlusion_level = _apply_cutout(image, self.severity, generator)
        elif corruption == "blur":
            blur_level = self.severity
            image = _apply_blur(image, self.severity)
        elif corruption == "label_ambiguity":
            noise_level = 0.5 * self.severity
            occlusion_level = 0.5 * self.severity
            image, occ = _apply_cutout(image, self.severity, generator)
            occlusion_level = max(occlusion_level, occ)
        severity = max(noise_level, occlusion_level, blur_level, clutter_level)
        is_corrupted = float(corruption != "clean")
        resistance_target = min(1.0, 0.35 * noise_level + 0.35 * occlusion_level + 0.15 * blur_level + 0.15 * clutter_level)
        payload: Dict[str, torch.Tensor | int | str] = {
            "image": image,
            "label": int(base["label"]),
            "clean_label": int(clean_base["label"]),
            "corruption_type": corruption,
            "corruption_type_id": torch.tensor(CORRUPTION_TYPES.index(corruption), dtype=torch.long),
            "corruption_severity": torch.tensor(severity, dtype=torch.float32),
            "noise_level": torch.tensor(noise_level, dtype=torch.float32),
            "occlusion_level": torch.tensor(occlusion_level, dtype=torch.float32),
            "blur_level": torch.tensor(blur_level, dtype=torch.float32),
            "clutter_level": torch.tensor(clutter_level, dtype=torch.float32),
            "is_corrupted": torch.tensor(is_corrupted, dtype=torch.float32),
            "resistance_target": torch.tensor(resistance_target, dtype=torch.float32),
        }
        if self.include_clean_pair:
            payload["clean_image"] = clean_image
        return payload


class CIFARCorruptionWrapper(Dataset):
    """Offline CIFAR-style corruption wrapper; no external corruption dataset required."""

    def __init__(self, base_dataset: Dataset, corruption_type: str = "mixed", seed: int = 0, severity: float = 0.6) -> None:
        if corruption_type not in CORRUPTION_TYPES:
            raise ValueError(f"corruption_type must be one of {CORRUPTION_TYPES}")
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.seed = seed
        self.severity = float(max(0.0, min(1.0, severity)))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int | str]:
        sample = self.base_dataset[index]
        if isinstance(sample, dict):
            image = sample["image"].clone().float()
            label = int(sample["label"])
        else:
            image, label = sample
            image = image.clone().float()
            label = int(label)
        generator = torch.Generator().manual_seed(self.seed + index)
        corruption = _choose_corruption(self.corruption_type, generator)
        noise_level = 0.0
        occlusion_level = 0.0
        blur_level = 0.0
        clutter_level = 0.0
        if corruption == "gaussian_noise":
            noise_level = self.severity
            image = (image + (0.05 + 0.35 * self.severity) * torch.randn(image.shape, generator=generator)).clamp(0.0, 1.0)
        elif corruption == "salt_pepper_noise":
            noise_level = self.severity
            image = _apply_salt_pepper(image, self.severity, generator)
        elif corruption in {"cutout_occlusion", "random_patch_occlusion"}:
            image, occlusion_level = _apply_cutout(image, self.severity, generator)
        elif corruption == "blur":
            blur_level = self.severity
            image = _apply_blur(image, self.severity)
        elif corruption == "background_clutter":
            clutter_level = self.severity
            image = (0.8 * image + 0.2 * torch.rand(image.shape, generator=generator)).clamp(0.0, 1.0)
        severity = max(noise_level, occlusion_level, blur_level, clutter_level)
        resistance_target = min(1.0, 0.35 * noise_level + 0.35 * occlusion_level + 0.15 * blur_level + 0.15 * clutter_level)
        return {
            "image": image,
            "label": label,
            "clean_label": label,
            "corruption_type": corruption,
            "corruption_type_id": torch.tensor(CORRUPTION_TYPES.index(corruption), dtype=torch.long),
            "corruption_severity": torch.tensor(severity, dtype=torch.float32),
            "noise_level": torch.tensor(noise_level, dtype=torch.float32),
            "occlusion_level": torch.tensor(occlusion_level, dtype=torch.float32),
            "blur_level": torch.tensor(blur_level, dtype=torch.float32),
            "clutter_level": torch.tensor(clutter_level, dtype=torch.float32),
            "is_corrupted": torch.tensor(float(corruption != "clean"), dtype=torch.float32),
            "resistance_target": torch.tensor(resistance_target, dtype=torch.float32),
        }


class TextCorruptionDataset(Dataset):
    """Synthetic text corruption dataset with resistance targets."""

    def __init__(
        self,
        size: int,
        seq_len: int = 64,
        seed: int = 0,
        corruption_type: str = "mixed",
        corruption_prob: float = 0.15,
    ) -> None:
        self.base = SyntheticTextEnergyDataset(size=size, seq_len=seq_len, seed=seed, corruption_prob=0.0)
        self.seed = seed
        self.corruption_type = corruption_type
        self.corruption_prob = float(max(0.0, min(1.0, corruption_prob)))
        self.vocab_size = self.base.vocab_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        payload = dict(self.base[index])
        generator = torch.Generator().manual_seed(self.seed + 200_000 + index)
        input_ids = payload["input_ids"].clone()
        input_mask = payload["input_mask"].bool()
        corruption_mask = (torch.rand(input_ids.shape, generator=generator) < self.corruption_prob) & input_mask
        corruption_mask = corruption_mask & (input_ids != self.base.vocab.bos_id) & (input_ids != self.base.vocab.eos_id)
        corruption = self.corruption_type
        if corruption == "mixed":
            corruption = ("random_token", "span", "bracket_mismatch", "shuffle")[int(torch.randint(0, 4, (1,), generator=generator).item())]
        if corruption_mask.any() and corruption in {"random_token", "span", "bracket_mismatch"}:
            replacements = torch.randint(3, self.vocab_size, (int(corruption_mask.sum().item()),), generator=generator)
            input_ids[corruption_mask] = replacements
        elif corruption_mask.any() and corruption == "shuffle":
            indices = corruption_mask.nonzero(as_tuple=False).flatten()
            if indices.numel() > 1:
                input_ids[indices] = input_ids[indices[torch.randperm(indices.numel(), generator=generator)]]
        severity = corruption_mask.float().mean()
        payload["input_ids"] = input_ids
        payload["corruption_mask"] = corruption_mask
        payload["corruption_type"] = corruption
        payload["corruption_severity"] = severity
        payload["is_corrupted"] = torch.tensor(float(corruption_mask.any()), dtype=torch.float32)
        payload["resistance_target"] = torch.maximum(payload["resistance_target"].float(), corruption_mask.float())
        return payload


__all__ = [
    "CIFARCorruptionDataset",
    "CIFARCorruptionWrapper",
    "CORRUPTION_TYPES",
    "SyntheticShapeUncertaintyDataset",
    "TextCorruptionDataset",
]
