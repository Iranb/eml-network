from __future__ import annotations

from typing import Callable, Dict, List

import torch
from torch.utils.data import Dataset

from .text_codecs import CharVocabulary


def _balanced_brackets(generator: torch.Generator, max_pairs: int = 5) -> str:
    bracket_pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    pairs = int(torch.randint(1, max_pairs + 1, (1,), generator=generator).item())
    openings = []
    for _ in range(pairs):
        left, right = bracket_pairs[int(torch.randint(0, len(bracket_pairs), (1,), generator=generator).item())]
        openings.append((left, right))
    text = "".join(left for left, _ in openings) + "".join(right for _, right in reversed(openings))
    return text


def _repeat_pattern(generator: torch.Generator) -> str:
    alphabet = "abcdxyz"
    pattern_len = int(torch.randint(2, 5, (1,), generator=generator).item())
    repeats = int(torch.randint(2, 5, (1,), generator=generator).item())
    chars = [alphabet[int(torch.randint(0, len(alphabet), (1,), generator=generator).item())] for _ in range(pattern_len)]
    pattern = "".join(chars)
    return f"REP:{pattern}|{pattern * repeats}"


def _copy_sequence(generator: torch.Generator) -> str:
    alphabet = "mnopqr"
    seq_len = int(torch.randint(3, 7, (1,), generator=generator).item())
    chars = [alphabet[int(torch.randint(0, len(alphabet), (1,), generator=generator).item())] for _ in range(seq_len)]
    payload = "".join(chars)
    return f"COPY:{payload}|{payload}"


def _reverse_sequence(generator: torch.Generator) -> str:
    alphabet = "stuvwx"
    seq_len = int(torch.randint(3, 7, (1,), generator=generator).item())
    chars = [alphabet[int(torch.randint(0, len(alphabet), (1,), generator=generator).item())] for _ in range(seq_len)]
    payload = "".join(chars)
    return f"REV:{payload}|{payload[::-1]}"


def _key_value_sequence(generator: torch.Generator) -> str:
    keys = ["a", "b", "c"]
    values = [str(int(torch.randint(0, 4, (1,), generator=generator).item())) for _ in keys]
    pairs = ";".join(f"{key}={value}" for key, value in zip(keys, values))
    return f"{pairs};"


def _dsl_command(generator: torch.Generator) -> str:
    directions = ["L", "R", "U", "D"]
    steps = int(torch.randint(1, 5, (1,), generator=generator).item())
    command = directions[int(torch.randint(0, len(directions), (1,), generator=generator).item())]
    return f"MOVE {command} {steps};END"


GENERATORS = [
    _balanced_brackets,
    _repeat_pattern,
    _copy_sequence,
    _reverse_sequence,
    _key_value_sequence,
    _dsl_command,
]

ENERGY_GENERATORS: Dict[str, Callable[[torch.Generator], str]] = {
    "brackets": _balanced_brackets,
    "repeat": _repeat_pattern,
    "copy": _copy_sequence,
    "reverse": _reverse_sequence,
    "kv": _key_value_sequence,
    "dsl": _dsl_command,
}


def _corrupt_text(text: str, generator: torch.Generator) -> str:
    if len(text) <= 6:
        return text
    symbols = list("abcdefghijklmnopqrstuvwxyz[]{}()=;|0123456789")
    text_chars = list(text)
    index = int(torch.randint(1, len(text_chars) - 1, (1,), generator=generator).item())
    replacement = symbols[int(torch.randint(0, len(symbols), (1,), generator=generator).item())]
    text_chars[index] = replacement
    return "".join(text_chars)


class SyntheticGrammarDataset(Dataset):
    """Offline character-level grammar dataset for next-char prediction."""

    def __init__(
        self,
        size: int,
        vocab: CharVocabulary,
        max_length: int = 48,
        seed: int = 0,
        corruption_prob: float = 0.3,
    ) -> None:
        if size <= 0 or max_length < 8:
            raise ValueError("invalid SyntheticGrammarDataset configuration")
        if not 0.0 <= corruption_prob <= 1.0:
            raise ValueError("corruption_prob must be in [0, 1]")

        self.size = size
        self.vocab = vocab
        self.max_length = max_length
        self.seed = seed
        self.corruption_prob = corruption_prob

    def __len__(self) -> int:
        return self.size

    def _pad(self, ids: List[int]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.full((self.max_length,), self.vocab.pad_id, dtype=torch.long)
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        input_ids[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[: len(ids)] = True
        return input_ids, mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        generator_fn = GENERATORS[int(torch.randint(0, len(GENERATORS), (1,), generator=generator).item())]
        text = generator_fn(generator)
        is_valid = True
        if torch.rand(1, generator=generator).item() < self.corruption_prob:
            text = _corrupt_text(text, generator)
            is_valid = False

        ids = [self.vocab.bos_id] + self.vocab.encode(text) + [self.vocab.eos_id]
        ids = ids[: self.max_length + 1]
        if len(ids) < 2:
            ids = ids + [self.vocab.eos_id]

        input_ids, input_mask = self._pad(ids[:-1])
        target_ids, _ = self._pad(ids[1:])

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "input_mask": input_mask,
            "validity_label": torch.tensor(float(is_valid), dtype=torch.float32),
        }


class SyntheticTextEnergyDataset(Dataset):
    """Offline character sequences with drive and resistance targets."""

    def __init__(
        self,
        size: int,
        seq_len: int = 64,
        vocab: CharVocabulary | None = None,
        seed: int = 0,
        task_type: str = "mixed",
        corruption_prob: float = 0.08,
    ) -> None:
        if size <= 0 or seq_len < 12:
            raise ValueError("invalid SyntheticTextEnergyDataset configuration")
        if task_type != "mixed" and task_type not in ENERGY_GENERATORS:
            raise ValueError("task_type must be mixed or a known synthetic task")
        if not 0.0 <= corruption_prob <= 1.0:
            raise ValueError("corruption_prob must be in [0, 1]")

        self.size = size
        self.seq_len = seq_len
        self.vocab = vocab or CharVocabulary()
        self.seed = seed
        self.task_type = task_type
        self.corruption_prob = corruption_prob
        self.task_names = tuple(ENERGY_GENERATORS.keys())
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab.pad_id

    def __len__(self) -> int:
        return self.size

    def _select_task(self, generator: torch.Generator) -> tuple[str, Callable[[torch.Generator], str], int]:
        if self.task_type == "mixed":
            task_index = int(torch.randint(0, len(self.task_names), (1,), generator=generator).item())
            task_name = self.task_names[task_index]
        else:
            task_name = self.task_type
            task_index = self.task_names.index(task_name)
        return task_name, ENERGY_GENERATORS[task_name], task_index

    def _make_text(self, generator: torch.Generator) -> tuple[str, int]:
        task_name, task_fn, task_index = self._select_task(generator)
        pieces = [task_fn(generator)]
        while len("|".join(pieces)) < self.seq_len and torch.rand(1, generator=generator).item() < 0.55:
            pieces.append(task_fn(generator) if self.task_type != "mixed" else self._select_task(generator)[1](generator))
        text = "|".join(pieces)
        if task_name == "brackets":
            text = f"<{text}>"
        return text[: max(1, self.seq_len - 1)], task_index

    def _pad_ids(self, ids: List[int]) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.full((self.seq_len,), self.vocab.pad_id, dtype=torch.long)
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        keep = min(len(ids), self.seq_len)
        values[:keep] = torch.tensor(ids[:keep], dtype=torch.long)
        mask[:keep] = True
        return values, mask

    def _labels(self, input_ids: torch.Tensor, input_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        boundary_chars = set("[]{}()<>=:;| ")
        ambiguous_chars = set("abcxyzmnopqr")
        boundary = torch.zeros(self.seq_len, dtype=torch.float32)
        ambiguity = torch.zeros(self.seq_len, dtype=torch.float32)
        for index, token_id in enumerate(input_ids.tolist()):
            if not bool(input_mask[index]):
                continue
            token = self.vocab.itos[int(token_id)]
            boundary[index] = 1.0 if token in boundary_chars else 0.0
            ambiguity[index] = 1.0 if token in ambiguous_chars else 0.0
        return boundary, ambiguity

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        text, task_label = self._make_text(generator)
        ids = [self.vocab.bos_id] + self.vocab.encode(text) + [self.vocab.eos_id]
        ids = ids[: self.seq_len + 1]
        if len(ids) < 2:
            ids = ids + [self.vocab.eos_id]

        input_ids, input_mask = self._pad_ids(ids[:-1])
        target_ids, _ = self._pad_ids(ids[1:])
        clean_input_ids = input_ids.clone()

        random_values = torch.rand(self.seq_len, generator=generator)
        corruption_mask = (random_values < self.corruption_prob) & input_mask
        corruption_mask = corruption_mask & (input_ids != self.vocab.bos_id) & (input_ids != self.vocab.eos_id)
        if corruption_mask.any():
            replacement_count = int(corruption_mask.sum().item())
            replacements = torch.randint(3, self.vocab_size, (replacement_count,), generator=generator, dtype=torch.long)
            input_ids[corruption_mask] = replacements

        boundary_labels, ambiguity_labels = self._labels(clean_input_ids, input_mask)
        resistance_target = (
            0.75 * corruption_mask.to(dtype=torch.float32)
            + 0.15 * boundary_labels
            + 0.10 * ambiguity_labels
        ).clamp(0.0, 1.0)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "input_mask": input_mask,
            "padding_mask": input_mask,
            "corruption_mask": corruption_mask,
            "boundary_labels": boundary_labels,
            "ambiguity_labels": ambiguity_labels,
            "resistance_target": resistance_target,
            "task_label": torch.tensor(task_label, dtype=torch.long),
        }


__all__ = ["SyntheticGrammarDataset", "SyntheticTextEnergyDataset", "GENERATORS", "ENERGY_GENERATORS"]
