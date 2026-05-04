from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Iterator

import torch

from src.config import get_settings
from src.repositories.mlflow import MLFlowRepository
from src.utils.model_archive import (
    build_config_from_run_params,
    build_model,
    get_model_entry,
    list_architectures,
)
from src.utils.noise_scheduler import LinearNoiseScheduler


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Загрузка DiT (из MLflow или локального файла) и инференс на случайном "
            "или на всех примерах SFT-датасета."
        )
    )
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run_id с весами модели.")
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Локальный путь к файлу весов (.pt/.pth). Если задан, MLflow не используется для загрузки весов.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(settings.datasets_folder) if settings.datasets_folder else None,
        help="Путь к директории с .sft файлами.",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default="weights/final_model.pt",
        help="Путь к артефакту весов внутри MLflow run.",
    )
    parser.add_argument("--tracking-uri", type=str, default=settings.mlflow_tracking_uri)
    parser.add_argument("--registry-uri", type=str, default=settings.mlflow_registry_uri)
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        choices=list_architectures(),
        help="Имя архитектуры модели. Если не указано, берется из MLflow params или Settings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.seed,
        help="Сид для выбора случайного примера и генерации шума.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Число шагов денойзинга (по умолчанию равно NUM_TRAIN_TIMESTEPS).",
    )
    parser.add_argument(
        "--condition-key",
        type=str,
        default=None,
        help="Ключ condition в .sft (если не задан, возьмется из MLflow params).",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default=None,
        help="Ключ target в .sft (если не задан, возьмется из MLflow params).",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Куда сохранить предсказания в формате .sft (опционально).",
    )
    parser.add_argument(
        "--evaluate-all",
        action="store_true",
        help=(
            "Запустить инференс на всех примерах датасета, посчитать метрики и "
            "вывести top/bottom по качеству."
        ),
    )
    parser.add_argument(
        "--best-worst-count",
        type=int,
        default=2,
        help="Сколько лучших и худших примеров выводить (по умолчанию: 2).",
    )
    return parser.parse_args()


def _parse_int(params: dict[str, str], key: str, fallback: int) -> int:
    value = params.get(key)
    return int(value) if value is not None else fallback


def _parse_float(params: dict[str, str], key: str, fallback: float) -> float:
    value = params.get(key)
    return float(value) if value is not None else fallback


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_sft(path: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file

        payload = load_file(str(path))
    except Exception:  # noqa: BLE001
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError(f"Ожидался dict в {path}, получено: {type(payload).__name__}")
    return payload


def _save_sft(path: Path, payload: dict[str, torch.Tensor]) -> None:
    try:
        from safetensors.torch import save_file

        save_file({key: value.contiguous() for key, value in payload.items()}, str(path))
    except Exception:  # noqa: BLE001
        torch.save(payload, path)


def _normalize_latents(
    tensor: torch.Tensor,
    *,
    latent_channels: int,
    expected_height: int,
    expected_width: int,
    source: str,
) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"{source}: ожидается 3D/4D тензор, получено: {tuple(tensor.shape)}")

    if tensor.shape[1] == latent_channels:
        channels_first = tensor
    elif tensor.shape[-1] == latent_channels:
        channels_first = tensor.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(
            f"{source}: не найдено {latent_channels} каналов, форма: {tuple(tensor.shape)}"
        )

    current_height, current_width = channels_first.shape[2], channels_first.shape[3]
    if current_height == expected_width and current_width == expected_height:
        channels_first = channels_first.transpose(2, 3).contiguous()
        current_height, current_width = channels_first.shape[2], channels_first.shape[3]

    if current_height != expected_height or current_width != expected_width:
        raise ValueError(
            f"{source}: shape [{current_height}, {current_width}], "
            f"ожидалось [{expected_height}, {expected_width}]"
        )
    return channels_first.float()


def _pick_random_dataset_sample(
    *,
    dataset_dir: Path,
    condition_key: str,
    target_key: str,
    latent_channels: int,
    condition_height: int,
    condition_width: int,
    target_height: int,
    target_width: int,
) -> tuple[torch.Tensor, torch.Tensor, Path, int]:
    files = sorted(dataset_dir.glob("*.sft"))
    if not files:
        raise FileNotFoundError(f"В {dataset_dir} не найдено .sft файлов.")

    selected_file = random.choice(files)
    payload = _load_sft(selected_file)

    if condition_key not in payload or target_key not in payload:
        missing = {condition_key, target_key}.difference(payload.keys())
        raise KeyError(f"В файле {selected_file.name} отсутствуют ключи: {sorted(missing)}")

    condition = _normalize_latents(
        payload[condition_key],
        latent_channels=latent_channels,
        expected_height=condition_height,
        expected_width=condition_width,
        source=f"{selected_file.name}:{condition_key}",
    )
    target = _normalize_latents(
        payload[target_key],
        latent_channels=latent_channels,
        expected_height=target_height,
        expected_width=target_width,
        source=f"{selected_file.name}:{target_key}",
    )
    if condition.shape[0] != target.shape[0]:
        raise ValueError(
            f"{selected_file.name}: batch размер condition ({condition.shape[0]}) "
            f"не равен target ({target.shape[0]})."
        )

    sample_index = random.randrange(condition.shape[0])
    return (
        condition[sample_index : sample_index + 1],
        target[sample_index : sample_index + 1],
        selected_file,
        sample_index,
    )


def _iter_dataset_samples(
    *,
    dataset_dir: Path,
    condition_key: str,
    target_key: str,
    latent_channels: int,
    condition_height: int,
    condition_width: int,
    target_height: int,
    target_width: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, Path, int]]:
    files = sorted(dataset_dir.glob("*.sft"))
    if not files:
        raise FileNotFoundError(f"В {dataset_dir} не найдено .sft файлов.")

    for sft_file in files:
        payload = _load_sft(sft_file)
        if condition_key not in payload or target_key not in payload:
            missing = {condition_key, target_key}.difference(payload.keys())
            raise KeyError(f"В файле {sft_file.name} отсутствуют ключи: {sorted(missing)}")

        condition = _normalize_latents(
            payload[condition_key],
            latent_channels=latent_channels,
            expected_height=condition_height,
            expected_width=condition_width,
            source=f"{sft_file.name}:{condition_key}",
        )
        target = _normalize_latents(
            payload[target_key],
            latent_channels=latent_channels,
            expected_height=target_height,
            expected_width=target_width,
            source=f"{sft_file.name}:{target_key}",
        )
        if condition.shape[0] != target.shape[0]:
            raise ValueError(
                f"{sft_file.name}: batch размер condition ({condition.shape[0]}) "
                f"не равен target ({target.shape[0]})."
            )

        for sample_index in range(condition.shape[0]):
            yield (
                condition[sample_index : sample_index + 1],
                target[sample_index : sample_index + 1],
                sft_file,
                sample_index,
            )


def _load_dataset_sample_by_index(
    *,
    dataset_dir: Path,
    condition_key: str,
    target_key: str,
    latent_channels: int,
    condition_height: int,
    condition_width: int,
    target_height: int,
    target_width: int,
    sample_file: Path,
    sample_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    for condition, target, source_file, source_index in _iter_dataset_samples(
        dataset_dir=dataset_dir,
        condition_key=condition_key,
        target_key=target_key,
        latent_channels=latent_channels,
        condition_height=condition_height,
        condition_width=condition_width,
        target_height=target_height,
        target_width=target_width,
    ):
        if source_file == sample_file and source_index == sample_index:
            return condition, target

    raise IndexError(f"Не найден sample index={sample_index} в файле {sample_file}")


@torch.no_grad()
def _run_reverse_diffusion(
    *,
    model: torch.nn.Module,
    scheduler: LinearNoiseScheduler,
    condition_latents: torch.Tensor,
    query_height: int,
    query_width: int,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("steps должен быть > 0.")

    query_shape = (
        condition_latents.shape[0],
        model.config.latent_channels,
        query_height,
        query_width,
    )
    current = torch.randn(query_shape, device=device)
    max_timestep = scheduler.num_train_timesteps - 1
    timesteps = torch.linspace(
        max_timestep,
        0,
        steps=num_steps,
        device=device,
    ).long()

    for timestep in timesteps:
        t = int(timestep.item())
        t_batch = torch.full((current.shape[0],), t, device=device, dtype=torch.long)
        predicted_noise = model(
            noisy_query_latents=current,
            condition_latents=condition_latents,
            timesteps=t_batch,
        )

        alpha_t = scheduler.alphas[t]
        alpha_cumprod_t = scheduler.alphas_cumprod[t]
        beta_t = scheduler.betas[t]
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            current - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise
        )

        if t > 0:
            noise = torch.randn_like(current)
            current = mean + torch.sqrt(beta_t) * noise
        else:
            current = mean

    return current


def main() -> None:
    args = parse_args()
    settings = get_settings()

    random.seed(args.seed)
    _set_torch_seed(args.seed)

    if not args.dataset_dir:
        raise ValueError("Не задан путь к датасету. Укажите DATASETS_FOLDER или --dataset-dir.")
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Директория датасета не найдена: {dataset_dir}")

    if not args.weights_path and not args.run_id:
        raise ValueError("Укажите --weights-path (локально) или --run-id (MLflow).")

    run_params: dict[str, str] = {}
    mlflow_repo: MLFlowRepository | None = None
    if args.run_id:
        mlflow_repo = MLFlowRepository(
            tracking_uri=args.tracking_uri,
            registry_uri=args.registry_uri,
        )
        run = mlflow_repo.client.get_run(args.run_id)
        run_params = run.data.params

    condition_key = args.condition_key or run_params.get("condition_key") or settings.condition_key
    target_key = args.target_key or run_params.get("target_key") or settings.target_key

    architecture_name = (
        args.architecture or run_params.get("architecture_name") or settings.model_architecture
    )
    get_model_entry(architecture_name)
    model_config = build_config_from_run_params(
        settings=settings,
        architecture_name=architecture_name,
        run_params=run_params,
    )

    num_train_timesteps = _parse_int(
        run_params, "num_train_timesteps", settings.num_train_timesteps
    )
    beta_start = _parse_float(run_params, "beta_start", settings.beta_start)
    beta_end = _parse_float(run_params, "beta_end", settings.beta_end)
    num_steps = args.steps or num_train_timesteps

    if args.best_worst_count <= 0:
        raise ValueError("--best-worst-count должен быть > 0.")

    device = _resolve_device()
    model = build_model(model_config).to(device)
    model.eval()

    if args.weights_path:
        local_weights_path = Path(args.weights_path)
        if not local_weights_path.exists() or not local_weights_path.is_file():
            raise FileNotFoundError(f"Файл весов не найден: {local_weights_path}")
    else:
        if mlflow_repo is None or not args.run_id:
            raise RuntimeError("MLflow repository не инициализирован для загрузки весов.")
        local_weights_path = Path(
            mlflow_repo.mlflow.artifacts.download_artifacts(
                run_id=args.run_id,
                artifact_path=args.artifact_path,
            )
        )

    state_dict = torch.load(local_weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    scheduler = LinearNoiseScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    )

    saved_condition_batches: list[torch.Tensor] = []
    saved_prediction_batches: list[torch.Tensor] = []

    if args.evaluate_all:
        scored_samples: list[dict[str, Any]] = []
        total_samples = 0
        for global_index, (condition_sample, target_sample, source_file, sample_index) in enumerate(
            _iter_dataset_samples(
                dataset_dir=dataset_dir,
                condition_key=condition_key,
                target_key=target_key,
                latent_channels=model_config.latent_channels,
                condition_height=model_config.condition_height,
                condition_width=model_config.condition_width,
                target_height=model_config.query_height,
                target_width=model_config.query_width,
            )
        ):
            sample_seed = args.seed + global_index
            _set_torch_seed(sample_seed)
            condition_device = condition_sample.to(device)
            target_device = target_sample.to(device)
            prediction = _run_reverse_diffusion(
                model=model,
                scheduler=scheduler,
                condition_latents=condition_device,
                query_height=target_device.shape[2],
                query_width=target_device.shape[3],
                num_steps=num_steps,
                device=device,
            )
            mse = torch.mean((prediction - target_device) ** 2).item()
            l1 = torch.mean(torch.abs(prediction - target_device)).item()
            scored_samples.append(
                {
                    "file": source_file,
                    "sample_index": sample_index,
                    "seed": sample_seed,
                    "mse": mse,
                    "l1": l1,
                }
            )
            total_samples += 1

        if total_samples == 0:
            raise RuntimeError("В датасете не найдено ни одного примера.")

        scored_samples.sort(key=lambda item: (item["mse"], item["l1"]))
        selection_size = min(args.best_worst_count, len(scored_samples))
        best_samples = scored_samples[:selection_size]
        worst_samples = list(reversed(scored_samples[-selection_size:]))

        print("Inference completed.")
        if args.weights_path:
            print(f"weights source: local file ({local_weights_path})")
        else:
            print(f"weights source: mlflow run_id={args.run_id}")
            print(f"weights artifact: {args.artifact_path}")
        print(f"condition_key: {condition_key}")
        print(f"target_key: {target_key}")
        print(f"processed samples: {len(scored_samples)}")

        print("")
        print(f"Top {selection_size} samples (best quality, min mse):")
        for rank, sample in enumerate(best_samples, start=1):
            print(
                f"[best #{rank}] dataset sample: {sample['file'].name} [index={sample['sample_index']}]"
            )
            print(f"condition_key: {condition_key}")
            print(f"target_key: {target_key}")
            print(
                f"prediction shape: "
                f"(1, {model_config.latent_channels}, {model_config.query_height}, {model_config.query_width})"
            )
            print(
                f"target shape: "
                f"(1, {model_config.latent_channels}, {model_config.query_height}, {model_config.query_width})"
            )
            print(f"mse_to_target: {sample['mse']:.6f}")
            print(f"l1_to_target: {sample['l1']:.6f}")

        print("")
        print(f"Top {selection_size} samples (worst quality, max mse):")
        for rank, sample in enumerate(worst_samples, start=1):
            print(
                f"[worst #{rank}] dataset sample: "
                f"{sample['file'].name} [index={sample['sample_index']}]"
            )
            print(f"condition_key: {condition_key}")
            print(f"target_key: {target_key}")
            print(
                f"prediction shape: "
                f"(1, {model_config.latent_channels}, {model_config.query_height}, {model_config.query_width})"
            )
            print(
                f"target shape: "
                f"(1, {model_config.latent_channels}, {model_config.query_height}, {model_config.query_width})"
            )
            print(f"mse_to_target: {sample['mse']:.6f}")
            print(f"l1_to_target: {sample['l1']:.6f}")

        unique_selected: dict[tuple[str, int], dict[str, Any]] = {}
        for sample in [*best_samples, *worst_samples]:
            key = (str(sample["file"]), int(sample["sample_index"]))
            unique_selected[key] = sample

        for selected_sample in unique_selected.values():
            condition_sample, target_sample = _load_dataset_sample_by_index(
                dataset_dir=dataset_dir,
                condition_key=condition_key,
                target_key=target_key,
                latent_channels=model_config.latent_channels,
                condition_height=model_config.condition_height,
                condition_width=model_config.condition_width,
                target_height=model_config.query_height,
                target_width=model_config.query_width,
                sample_file=Path(selected_sample["file"]),
                sample_index=int(selected_sample["sample_index"]),
            )
            _set_torch_seed(int(selected_sample["seed"]))
            condition_device = condition_sample.to(device)
            target_device = target_sample.to(device)
            prediction = _run_reverse_diffusion(
                model=model,
                scheduler=scheduler,
                condition_latents=condition_device,
                query_height=target_device.shape[2],
                query_width=target_device.shape[3],
                num_steps=num_steps,
                device=device,
            )
            saved_condition_batches.append(condition_device.detach().cpu())
            saved_prediction_batches.append(prediction.detach().cpu())
    else:
        condition_sample, target_sample, selected_file, sample_index = _pick_random_dataset_sample(
            dataset_dir=dataset_dir,
            condition_key=condition_key,
            target_key=target_key,
            latent_channels=model_config.latent_channels,
            condition_height=model_config.condition_height,
            condition_width=model_config.condition_width,
            target_height=model_config.query_height,
            target_width=model_config.query_width,
        )
        condition_sample = condition_sample.to(device)
        target_sample = target_sample.to(device)
        prediction = _run_reverse_diffusion(
            model=model,
            scheduler=scheduler,
            condition_latents=condition_sample,
            query_height=target_sample.shape[2],
            query_width=target_sample.shape[3],
            num_steps=num_steps,
            device=device,
        )
        mse = torch.mean((prediction - target_sample) ** 2).item()
        l1 = torch.mean(torch.abs(prediction - target_sample)).item()

        print("Inference completed.")
        if args.weights_path:
            print(f"weights source: local file ({local_weights_path})")
        else:
            print(f"weights source: mlflow run_id={args.run_id}")
            print(f"weights artifact: {args.artifact_path}")
        print(f"dataset sample: {selected_file.name} [index={sample_index}]")
        print(f"condition_key: {condition_key}")
        print(f"target_key: {target_key}")
        print(f"prediction shape: {tuple(prediction.shape)}")
        print(f"target shape: {tuple(target_sample.shape)}")
        print(f"mse_to_target: {mse:.6f}")
        print(f"l1_to_target: {l1:.6f}")

        saved_condition_batches.append(condition_sample.detach().cpu())
        saved_prediction_batches.append(prediction.detach().cpu())

    if args.save_output:
        output_path = Path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() != ".sft":
            output_path = output_path.with_suffix(".sft")

        condition_cpu = torch.cat(saved_condition_batches, dim=0)
        prediction_cpu = torch.cat(saved_prediction_batches, dim=0)
        condition_out = condition_cpu.transpose(2, 3).contiguous()
        prediction_out = prediction_cpu.transpose(2, 3).contiguous()

        if prediction_out.shape[3] % 2 != 0:
            raise ValueError(
                "Невозможно разделить prediction на левую/правую части: "
                f"ширина {prediction_out.shape[3]} нечетная."
            )
        half_width = prediction_out.shape[3] // 2
        left_prediction = prediction_out[:, :, :, :half_width]
        right_prediction = prediction_out[:, :, :, half_width:]
        if left_prediction.shape[3] != right_prediction.shape[3]:
            raise ValueError("Левая и правая части prediction имеют разную ширину.")
        combined_out = torch.cat((left_prediction, condition_out, right_prediction), dim=3)

        _save_sft(
            output_path,
            {
                "latents_54x30": condition_out,
                "latents_16x30": prediction_out,
                "latents_70x30": combined_out,
            },
        )
        print(f"Saved output: {output_path}")


if __name__ == "__main__":
    main()
