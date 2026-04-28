from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class SFTLatentDataset(IterableDataset[dict[str, Tensor]]):
    """Итеративный датасет: читает `.sft` файлы по одному, не держа всё в RAM."""

    def __init__(
        self,
        dataset_dir: str | Path,
        condition_key: str,
        target_key: str,
        latent_channels: int = 16,
        condition_height: int | None = None,
        condition_width: int | None = None,
        target_height: int | None = None,
        target_width: int | None = None,
        split: str = "train",
        train_ratio: float = 0.9,
        split_seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.condition_key = condition_key
        self.target_key = target_key
        self.latent_channels = latent_channels
        self.condition_height = condition_height
        self.condition_width = condition_width
        self.target_height = target_height
        self.target_width = target_width
        self.split = split
        self.train_ratio = train_ratio
        self.split_seed = split_seed

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Директория датасета не найдена: {self.dataset_dir}")
        if not self.dataset_dir.is_dir():
            raise ValueError(f"dataset_dir должен быть директорией: {self.dataset_dir}")
        if self.split not in {"train", "val"}:
            raise ValueError("split должен быть 'train' или 'val'.")
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio должен быть в диапазоне (0, 1).")

    def _resolve_files_for_worker(self) -> list[Path]:
        files = sorted(self.dataset_dir.glob("*.sft"))
        if not files:
            raise FileNotFoundError(
                f"В директории {self.dataset_dir} не найдено ни одного .sft файла."
            )

        worker = get_worker_info()
        if worker is None:
            return files
        return files[worker.id :: worker.num_workers]

    @staticmethod
    def _load_sft(path: Path) -> dict[str, Any]:
        try:
            from safetensors.torch import load_file

            data = load_file(str(path))
        except Exception:  # noqa: BLE001
            data = torch.load(path, map_location="cpu")

        if not isinstance(data, dict):
            raise ValueError(f"Ожидался dict в {path}, получено: {type(data).__name__}")
        return data

    def _normalize_latents(
        self,
        tensor: Tensor,
        path: Path,
        key: str,
        expected_height: int | None = None,
        expected_width: int | None = None,
    ) -> Tensor:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(
                f"{path.name}:{key} должен быть 3D или 4D тензором, получено: {tuple(tensor.shape)}"
            )

        if tensor.shape[1] == self.latent_channels:
            channels_first = tensor
        elif tensor.shape[-1] == self.latent_channels:
            channels_first = tensor.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(
                f"{path.name}:{key} не содержит {self.latent_channels} каналов, форма: {tuple(tensor.shape)}"
            )
        if expected_height is not None and expected_width is not None:
            current_height, current_width = channels_first.shape[2], channels_first.shape[3]
            if current_height == expected_width and current_width == expected_height:
                channels_first = channels_first.transpose(2, 3).contiguous()
                current_height, current_width = channels_first.shape[2], channels_first.shape[3]
            if current_height != expected_height or current_width != expected_width:
                raise ValueError(
                    f"{path.name}:{key} имеет shape [{current_height}, {current_width}], "
                    f"ожидается [{expected_height}, {expected_width}]."
                )

        return channels_first.float()

    def _resolve_split_indices(self, samples_count: int, file_path: Path) -> list[int]:
        if samples_count <= 0:
            return []

        file_crc = zlib.crc32(file_path.name.encode("utf-8")) & 0xFFFFFFFF
        base_seed = (file_crc + self.split_seed) & 0xFFFFFFFF
        generator = torch.Generator().manual_seed(base_seed)
        permutation = torch.randperm(samples_count, generator=generator).tolist()

        split_index = int(samples_count * self.train_ratio)
        if samples_count > 1:
            split_index = max(1, min(samples_count - 1, split_index))
        else:
            split_index = 1

        if self.split == "train":
            return permutation[:split_index]
        return permutation[split_index:]

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        files = self._resolve_files_for_worker()
        for file_path in files:
            payload = self._load_sft(file_path)
            if self.condition_key not in payload or self.target_key not in payload:
                missing = {self.condition_key, self.target_key}.difference(payload.keys())
                raise KeyError(f"В файле {file_path.name} отсутствуют ключи: {sorted(missing)}")

            condition = self._normalize_latents(
                tensor=payload[self.condition_key],
                path=file_path,
                key=self.condition_key,
                expected_height=self.condition_height,
                expected_width=self.condition_width,
            )
            target = self._normalize_latents(
                tensor=payload[self.target_key],
                path=file_path,
                key=self.target_key,
                expected_height=self.target_height,
                expected_width=self.target_width,
            )
            if condition.shape[0] != target.shape[0]:
                raise ValueError(
                    f"Размер batch не совпадает в {file_path.name}: "
                    f"{condition.shape[0]} != {target.shape[0]}"
                )

            split_indices = self._resolve_split_indices(
                samples_count=condition.shape[0],
                file_path=file_path,
            )
            for index in split_indices:
                yield {"condition": condition[index], "target": target[index]}


def create_sft_dataloader(
    dataset_dir: str | Path,
    condition_key: str,
    target_key: str,
    *,
    batch_size: int,
    num_workers: int = 0,
    latent_channels: int = 16,
    condition_height: int | None = None,
    condition_width: int | None = None,
    target_height: int | None = None,
    target_width: int | None = None,
    pin_memory: bool = False,
    split: str = "train",
    train_ratio: float = 0.9,
    split_seed: int = 42,
) -> DataLoader[dict[str, Tensor]]:
    dataset = SFTLatentDataset(
        dataset_dir=dataset_dir,
        condition_key=condition_key,
        target_key=target_key,
        latent_channels=latent_channels,
        condition_height=condition_height,
        condition_width=condition_width,
        target_height=target_height,
        target_width=target_width,
        split=split,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
