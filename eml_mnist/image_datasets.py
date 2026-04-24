from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset


SHAPES = ["circle", "square", "triangle", "cross", "diamond"]
COLORS: Dict[str, torch.Tensor] = {
    "red": torch.tensor([1.0, 0.15, 0.15]),
    "green": torch.tensor([0.15, 1.0, 0.15]),
    "blue": torch.tensor([0.15, 0.2, 1.0]),
    "yellow": torch.tensor([1.0, 0.9, 0.2]),
}
TEXTURES = ["solid", "stripes", "dots"]
SIZES = {"small": 0.18, "medium": 0.28, "large": 0.38}
POSITIONS = ["left", "center", "right", "random"]
NOISE_LEVELS = {"low": 0.02, "medium": 0.05, "high": 0.10}
OCCLUSIONS = ["none", "partial"]
TARGET_TYPES = ("shape", "color", "combo")


def _meshgrid(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-1.0, 1.0, size)
    return torch.meshgrid(coords, coords, indexing="ij")


def _center_for_position(position: str, generator: torch.Generator) -> tuple[float, float]:
    if position == "left":
        return -0.45, 0.0
    if position == "center":
        return 0.0, 0.0
    if position == "right":
        return 0.45, 0.0
    return (
        float(torch.empty(1).uniform_(-0.55, 0.55, generator=generator).item()),
        float(torch.empty(1).uniform_(-0.55, 0.55, generator=generator).item()),
    )


def _shape_mask(shape: str, yy: torch.Tensor, xx: torch.Tensor, cx: float, cy: float, scale: float) -> torch.Tensor:
    x = xx - cx
    y = yy - cy
    if shape == "circle":
        return x.square() + y.square() <= scale**2
    if shape == "square":
        return (x.abs() <= scale) & (y.abs() <= scale)
    if shape == "triangle":
        top = -scale
        bottom = scale
        left = -scale
        right = scale
        within_vertical = (y >= top) & (y <= bottom)
        allowed_width = (bottom - y) / (bottom - top + 1e-6) * scale
        return within_vertical & (x.abs() <= allowed_width) & (x >= left) & (x <= right)
    if shape == "cross":
        bar = scale * 0.35
        return ((x.abs() <= bar) & (y.abs() <= scale)) | ((y.abs() <= bar) & (x.abs() <= scale))
    if shape == "diamond":
        return (x.abs() + y.abs()) <= scale
    raise ValueError(f"unsupported shape: {shape}")


def _texture_mask(texture: str, xx: torch.Tensor, yy: torch.Tensor, scale: float) -> torch.Tensor:
    if texture == "solid":
        return torch.ones_like(xx, dtype=torch.bool)
    if texture == "stripes":
        stripe_width = max(scale * 3.0, 0.12)
        return torch.sin((xx + yy) * (3.14159 / stripe_width)).gt(0.0)
    if texture == "dots":
        freq = max(scale * 6.0, 0.18)
        return (torch.cos(xx * (3.14159 / freq)) * torch.cos(yy * (3.14159 / freq))).gt(0.35)
    raise ValueError(f"unsupported texture: {texture}")


def _apply_occlusion(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    _, height, width = image.shape
    occ_h = int(torch.randint(low=height // 5, high=height // 2, size=(1,), generator=generator).item())
    occ_w = int(torch.randint(low=width // 5, high=width // 2, size=(1,), generator=generator).item())
    top = int(torch.randint(low=0, high=height - occ_h + 1, size=(1,), generator=generator).item())
    left = int(torch.randint(low=0, high=width - occ_w + 1, size=(1,), generator=generator).item())
    image[:, top : top + occ_h, left : left + occ_w] *= 0.15
    return image


def _apply_occlusion_with_visibility(
    image: torch.Tensor,
    visible_mask: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    _, height, width = image.shape
    occ_h = int(torch.randint(low=height // 5, high=height // 2, size=(1,), generator=generator).item())
    occ_w = int(torch.randint(low=width // 5, high=width // 2, size=(1,), generator=generator).item())
    top = int(torch.randint(low=0, high=height - occ_h + 1, size=(1,), generator=generator).item())
    left = int(torch.randint(low=0, high=width - occ_w + 1, size=(1,), generator=generator).item())
    occlusion_mask = torch.zeros_like(visible_mask, dtype=torch.bool)
    occlusion_mask[top : top + occ_h, left : left + occ_w] = True
    original_visible = visible_mask.sum().clamp_min(1).float()
    updated_visible = visible_mask & ~occlusion_mask
    occlusion_level = float(1.0 - updated_visible.sum().float().div(original_visible).item())
    image[:, top : top + occ_h, left : left + occ_w] *= 0.15
    return image, updated_visible, occlusion_level


def _apply_background_clutter(
    image: torch.Tensor,
    yy: torch.Tensor,
    xx: torch.Tensor,
    generator: torch.Generator,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    clutter_mask = torch.zeros_like(xx, dtype=torch.bool)
    color_values = list(COLORS.values())
    clutter_shapes = ("circle", "square", "diamond")

    for _ in range(count):
        cx = float(torch.empty(1).uniform_(-0.8, 0.8, generator=generator).item())
        cy = float(torch.empty(1).uniform_(-0.8, 0.8, generator=generator).item())
        scale = float(torch.empty(1).uniform_(0.05, 0.22, generator=generator).item())
        shape_name = clutter_shapes[int(torch.randint(0, len(clutter_shapes), (1,), generator=generator).item())]
        alpha = float(torch.empty(1).uniform_(0.03, 0.12, generator=generator).item())
        color = color_values[int(torch.randint(0, len(color_values), (1,), generator=generator).item())].view(3, 1, 1)
        shape_mask = _shape_mask(shape_name, yy, xx, cx, cy, scale)
        image = torch.where(shape_mask.unsqueeze(0), image * (1.0 - alpha) + color * alpha, image)
        clutter_mask |= shape_mask

    return image, clutter_mask


class SyntheticShapeDataset(Dataset):
    """Offline synthetic shape dataset for image classification validation."""

    def __init__(
        self,
        size: int,
        image_size: int = 32,
        seed: int = 0,
        include_combo_label: bool = True,
    ) -> None:
        if size <= 0 or image_size <= 0:
            raise ValueError("size and image_size must be positive")
        self.size = size
        self.image_size = image_size
        self.seed = seed
        self.include_combo_label = include_combo_label
        self.yy, self.xx = _meshgrid(image_size)
        self.color_names = list(COLORS.keys())

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        generator = torch.Generator().manual_seed(self.seed + index)

        shape_index = int(torch.randint(0, len(SHAPES), (1,), generator=generator).item())
        color_index = int(torch.randint(0, len(self.color_names), (1,), generator=generator).item())
        texture_index = int(torch.randint(0, len(TEXTURES), (1,), generator=generator).item())
        size_index = int(torch.randint(0, len(SIZES), (1,), generator=generator).item())
        position_index = int(torch.randint(0, len(POSITIONS), (1,), generator=generator).item())
        noise_index = int(torch.randint(0, len(NOISE_LEVELS), (1,), generator=generator).item())
        occlusion_index = int(torch.randint(0, len(OCCLUSIONS), (1,), generator=generator).item())

        shape_name = SHAPES[shape_index]
        color_name = self.color_names[color_index]
        texture_name = TEXTURES[texture_index]
        size_name = list(SIZES.keys())[size_index]
        position_name = POSITIONS[position_index]
        noise_name = list(NOISE_LEVELS.keys())[noise_index]
        occlusion_name = OCCLUSIONS[occlusion_index]

        cx, cy = _center_for_position(position_name, generator)
        scale = SIZES[size_name]
        mask = _shape_mask(shape_name, self.yy, self.xx, cx, cy, scale)
        texture = _texture_mask(texture_name, self.xx - cx, self.yy - cy, scale)
        mask = mask & texture

        image = torch.full((3, self.image_size, self.image_size), 0.08, dtype=torch.float32)
        color = COLORS[color_name].view(3, 1, 1)
        image = image + color * mask.unsqueeze(0).float()
        image = image.clamp(0.0, 1.0)

        if occlusion_name == "partial":
            image = _apply_occlusion(image, generator)

        noise_scale = NOISE_LEVELS[noise_name]
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
        image = image + noise_scale * noise
        image = image.clamp(0.0, 1.0)

        payload: Dict[str, torch.Tensor | int] = {
            "image": image,
            "shape_label": shape_index,
            "color_label": color_index,
        }
        if self.include_combo_label:
            payload["combo_label"] = shape_index * len(self.color_names) + color_index
        return payload


class SyntheticShapeEnergyDataset(Dataset):
    """Offline synthetic image dataset with resistance metadata for EML image fields."""

    def __init__(
        self,
        size: int,
        image_size: int = 32,
        seed: int = 0,
        target_type: str = "shape",
        include_background_clutter: bool = True,
        include_mask: bool = True,
        forced_noise_name: str | None = None,
        forced_occlusion_name: str | None = None,
        forced_position_name: str | None = None,
        forced_clutter_flag: int | None = None,
    ) -> None:
        if size <= 0 or image_size <= 0:
            raise ValueError("size and image_size must be positive")
        if target_type not in TARGET_TYPES:
            raise ValueError(f"target_type must be one of {TARGET_TYPES}")

        self.size = size
        self.image_size = image_size
        self.seed = seed
        self.target_type = target_type
        self.include_background_clutter = include_background_clutter
        self.include_mask = include_mask
        if forced_noise_name is not None and forced_noise_name not in NOISE_LEVELS:
            raise ValueError(f"forced_noise_name must be one of {tuple(NOISE_LEVELS)}")
        if forced_occlusion_name is not None and forced_occlusion_name not in OCCLUSIONS:
            raise ValueError(f"forced_occlusion_name must be one of {tuple(OCCLUSIONS)}")
        if forced_position_name is not None and forced_position_name not in POSITIONS:
            raise ValueError(f"forced_position_name must be one of {tuple(POSITIONS)}")
        if forced_clutter_flag is not None and forced_clutter_flag not in (0, 1):
            raise ValueError("forced_clutter_flag must be 0 or 1")
        self.forced_noise_name = forced_noise_name
        self.forced_occlusion_name = forced_occlusion_name
        self.forced_position_name = forced_position_name
        self.forced_clutter_flag = forced_clutter_flag
        self.yy, self.xx = _meshgrid(image_size)
        self.color_names = list(COLORS.keys())
        self.num_classes = {
            "shape": len(SHAPES),
            "color": len(self.color_names),
            "combo": len(SHAPES) * len(self.color_names),
        }[target_type]

    def __len__(self) -> int:
        return self.size

    def _label_for(
        self,
        shape_index: int,
        color_index: int,
    ) -> int:
        if self.target_type == "shape":
            return shape_index
        if self.target_type == "color":
            return color_index
        return shape_index * len(self.color_names) + color_index

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        generator = torch.Generator().manual_seed(self.seed + index)

        shape_index = int(torch.randint(0, len(SHAPES), (1,), generator=generator).item())
        color_index = int(torch.randint(0, len(self.color_names), (1,), generator=generator).item())
        texture_index = int(torch.randint(0, len(TEXTURES), (1,), generator=generator).item())
        size_index = int(torch.randint(0, len(SIZES), (1,), generator=generator).item())
        position_index = int(torch.randint(0, len(POSITIONS), (1,), generator=generator).item())
        noise_index = int(torch.randint(0, len(NOISE_LEVELS), (1,), generator=generator).item())
        occlusion_index = int(torch.randint(0, len(OCCLUSIONS), (1,), generator=generator).item())
        clutter_flag = self.forced_clutter_flag
        if clutter_flag is None:
            clutter_flag = int(torch.randint(0, 2, (1,), generator=generator).item()) if self.include_background_clutter else 0

        shape_name = SHAPES[shape_index]
        color_name = self.color_names[color_index]
        texture_name = TEXTURES[texture_index]
        size_name = list(SIZES.keys())[size_index]
        position_name = self.forced_position_name or POSITIONS[position_index]
        noise_name = self.forced_noise_name or list(NOISE_LEVELS.keys())[noise_index]
        occlusion_name = self.forced_occlusion_name or OCCLUSIONS[occlusion_index]

        cx, cy = _center_for_position(position_name, generator)
        scale = SIZES[size_name]
        shape_mask = _shape_mask(shape_name, self.yy, self.xx, cx, cy, scale)
        texture_mask = _texture_mask(texture_name, self.xx - cx, self.yy - cy, scale)
        visible_mask = shape_mask & texture_mask

        image = torch.full((3, self.image_size, self.image_size), 0.08, dtype=torch.float32)
        if clutter_flag:
            clutter_count = int(torch.randint(1, 4, (1,), generator=generator).item())
            image, clutter_mask = _apply_background_clutter(image, self.yy, self.xx, generator, clutter_count)
        else:
            clutter_mask = torch.zeros_like(visible_mask, dtype=torch.bool)

        color = COLORS[color_name].view(3, 1, 1)
        image = torch.where(visible_mask.unsqueeze(0), image + color, image)
        image = image.clamp(0.0, 1.0)

        if occlusion_name == "partial":
            image, visible_mask, occlusion_level = _apply_occlusion_with_visibility(image, visible_mask, generator)
        else:
            occlusion_level = 0.0

        noise_scale = NOISE_LEVELS[noise_name]
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
        image = (image + noise_scale * noise).clamp(0.0, 1.0)

        noise_level = float(noise_scale / max(NOISE_LEVELS.values()))
        clutter_level = float(clutter_flag)
        resistance_target = min(1.0, 0.45 * noise_level + 0.45 * occlusion_level + 0.10 * clutter_level)
        combo_label = shape_index * len(self.color_names) + color_index
        label = self._label_for(shape_index, color_index)

        payload: Dict[str, torch.Tensor | int] = {
            "image": image,
            "label": label,
            "shape_label": shape_index,
            "color_label": color_index,
            "combo_label": combo_label,
            "noise_level": torch.tensor(noise_level, dtype=torch.float32),
            "occlusion_level": torch.tensor(occlusion_level, dtype=torch.float32),
            "background_clutter": torch.tensor(clutter_level, dtype=torch.float32),
            "resistance_target": torch.tensor(resistance_target, dtype=torch.float32),
        }
        if self.include_mask:
            payload["mask"] = visible_mask
            payload["clutter_mask"] = clutter_mask
        return payload


class CIFARCorruptionDataset(Dataset):
    """Deterministic clean/noisy/occluded wrapper for CIFAR-style tensor datasets."""

    def __init__(
        self,
        base_dataset: Dataset,
        mode: str = "clean",
        seed: int = 0,
        noise_std: float = 0.25,
        occlusion_frac: float = 0.33,
    ) -> None:
        if mode not in {"clean", "noisy", "occluded", "mixed"}:
            raise ValueError("mode must be one of clean, noisy, occluded, mixed")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if not 0.0 <= occlusion_frac <= 0.95:
            raise ValueError("occlusion_frac must be in [0, 0.95]")
        self.base_dataset = base_dataset
        self.mode = mode
        self.seed = seed
        self.noise_std = float(noise_std)
        self.occlusion_frac = float(occlusion_frac)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _choose_mode(self, index: int) -> str:
        if self.mode != "mixed":
            return self.mode
        generator = torch.Generator().manual_seed(self.seed + index)
        choice = int(torch.randint(0, 3, (1,), generator=generator).item())
        return ("clean", "noisy", "occluded")[choice]

    def _apply_noise(self, image: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, float]:
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
        return (image + self.noise_std * noise).clamp(0.0, 1.0), 1.0

    def _apply_occlusion(self, image: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, float]:
        _, height, width = image.shape
        occ_h = max(1, int(round(height * self.occlusion_frac)))
        occ_w = max(1, int(round(width * self.occlusion_frac)))
        top = int(torch.randint(0, height - occ_h + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, width - occ_w + 1, (1,), generator=generator).item())
        image = image.clone()
        image[:, top : top + occ_h, left : left + occ_w] *= 0.1
        return image, float((occ_h * occ_w) / float(height * width))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        sample = self.base_dataset[index]
        if isinstance(sample, dict):
            image = sample["image"].clone().float()
            label = int(sample["label"])
        else:
            image, label = sample
            image = image.clone().float()
            label = int(label)
        generator = torch.Generator().manual_seed(self.seed + index)
        mode = self._choose_mode(index)
        noise_level = 0.0
        occlusion_level = 0.0
        if mode == "noisy":
            image, noise_level = self._apply_noise(image, generator)
        elif mode == "occluded":
            image, occlusion_level = self._apply_occlusion(image, generator)
        resistance_target = min(1.0, 0.5 * noise_level + 0.5 * occlusion_level)
        return {
            "image": image,
            "label": label,
            "noise_level": torch.tensor(noise_level, dtype=torch.float32),
            "occlusion_level": torch.tensor(occlusion_level, dtype=torch.float32),
            "resistance_target": torch.tensor(resistance_target, dtype=torch.float32),
        }


__all__ = [
    "CIFARCorruptionDataset",
    "COLORS",
    "SHAPES",
    "SyntheticShapeDataset",
    "SyntheticShapeEnergyDataset",
    "TEXTURES",
]
