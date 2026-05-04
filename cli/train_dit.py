from __future__ import annotations

import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import get_settings
from src.data.sft_dataset import create_sft_dataloader
from src.repositories.mlflow import MLFlowRepository
from src.utils.model_archive import build_config_from_settings, build_model, list_architectures
from src.utils.noise_scheduler import LinearNoiseScheduler

settings = get_settings()


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение DiT с логированием в MLflow.")
    parser.add_argument(
        "--architecture",
        type=str,
        default=settings.model_architecture,
        choices=list_architectures(),
        help="Имя архитектуры модели из model archive.",
    )
    parser.add_argument("--run-name", type=str, default="baseline-dit-training-local")
    parser.add_argument("--dataset-dir", type=str, default=settings.datasets_folder)
    parser.add_argument("--tracking-uri", type=str, default=settings.mlflow_tracking_uri)
    parser.add_argument("--registry-uri", type=str, default=settings.mlflow_registry_uri)
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default=settings.mlflow_registered_model_name,
        help="Имя модели в MLflow Model Registry для финальной версии.",
    )
    parser.add_argument(
        "--preview-every-n-epochs",
        type=int,
        default=1,
        help="Период тестового прогона для визуализации (каждые N эпох). 0 отключает.",
    )
    parser.add_argument(
        "--preview-images-count",
        type=int,
        default=4,
        help="Количество фиксированных изображений M для тестового прогона.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(gpu_support: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_noise_metrics(
    *,
    predicted_noise: torch.Tensor,
    noise: torch.Tensor,
    clean_target: torch.Tensor,
    noisy_target: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: LinearNoiseScheduler,
) -> dict[str, float]:
    loss_mse = float(functional.mse_loss(predicted_noise, noise).item())
    loss_l1 = float(functional.l1_loss(predicted_noise, noise).item())
    cosine_similarity = float(
        functional.cosine_similarity(
            predicted_noise.flatten(1),
            noise.flatten(1),
            dim=1,
        )
        .mean()
        .item()
    )
    predicted_noise_std = float(predicted_noise.std().item())
    target_noise_std = float(noise.std().item())
    timestep_mean = float(timesteps.float().mean().item())
    sqrt_alpha = scheduler.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    reconstructed_x0 = (noisy_target - sqrt_one_minus_alpha * predicted_noise) / (sqrt_alpha + 1e-8)
    x0_reconstruction_mse = float(functional.mse_loss(reconstructed_x0, clean_target).item())
    snr = scheduler.alphas_cumprod[timesteps] / (1.0 - scheduler.alphas_cumprod[timesteps] + 1e-8)
    snr_mean = float(snr.mean().item())
    return {
        "loss_mse": loss_mse,
        "loss_l1": loss_l1,
        "noise_cosine_similarity": cosine_similarity,
        "x0_reconstruction_mse": x0_reconstruction_mse,
        "noise_pred_std": predicted_noise_std,
        "noise_target_std": target_noise_std,
        "timestep_mean": timestep_mean,
        "snr_mean": snr_mean,
    }


def evaluate_on_validation(
    *,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scheduler: LinearNoiseScheduler,
    device: torch.device,
    use_cuda: bool,
) -> dict[str, float]:
    model.eval()
    start_time = time.perf_counter()
    weighted_sums: dict[str, float] = {}
    samples_seen = 0
    batches_seen = 0

    with torch.no_grad():
        for batch in dataloader:
            condition = batch["condition"].to(device, non_blocking=use_cuda)
            clean_target = batch["target"].to(device, non_blocking=use_cuda)
            timesteps = scheduler.sample_timesteps(batch_size=clean_target.shape[0], device=device)
            noise = torch.randn_like(clean_target)
            noisy_target = scheduler.add_noise(clean_target, noise, timesteps)
            predicted_noise = model(
                noisy_query_latents=noisy_target,
                condition_latents=condition,
                timesteps=timesteps,
            )
            batch_metrics = compute_noise_metrics(
                predicted_noise=predicted_noise,
                noise=noise,
                clean_target=clean_target,
                noisy_target=noisy_target,
                timesteps=timesteps,
                scheduler=scheduler,
            )
            batch_size = clean_target.shape[0]
            samples_seen += batch_size
            batches_seen += 1
            for key, value in batch_metrics.items():
                weighted_sums[key] = weighted_sums.get(key, 0.0) + value * batch_size

    if samples_seen == 0:
        return {"num_samples": 0.0, "num_batches": float(batches_seen)}

    duration = max(time.perf_counter() - start_time, 1e-8)
    aggregated = {key: value / samples_seen for key, value in weighted_sums.items()}
    aggregated["num_samples"] = float(samples_seen)
    aggregated["num_batches"] = float(batches_seen)
    aggregated["steps_per_second"] = float(batches_seen / duration)
    aggregated["samples_per_second"] = float(samples_seen / duration)
    return aggregated


def _collect_fixed_preview_batch(
    *,
    dataloader: torch.utils.data.DataLoader,
    max_images: int,
) -> dict[str, torch.Tensor] | None:
    if max_images <= 0:
        return None

    fixed_condition: list[torch.Tensor] = []
    fixed_target: list[torch.Tensor] = []
    collected = 0
    for batch in dataloader:
        condition = batch["condition"]
        target = batch["target"]
        batch_size = condition.shape[0]
        take_count = min(max_images - collected, batch_size)
        if take_count <= 0:
            break
        fixed_condition.append(condition[:take_count].detach().cpu())
        fixed_target.append(target[:take_count].detach().cpu())
        collected += take_count
        if collected >= max_images:
            break

    if collected == 0:
        return None

    return {
        "condition": torch.cat(fixed_condition, dim=0),
        "target": torch.cat(fixed_target, dim=0),
    }


def _latent_to_display_image(latent: torch.Tensor) -> np.ndarray:
    array = latent.detach().cpu().float().numpy()
    if array.ndim != 3:
        raise ValueError(f"Ожидается тензор [C, H, W], получено: {tuple(array.shape)}")
    if array.shape[0] >= 3:
        rgb = array[:3]
    else:
        rgb = np.repeat(array[:1], 3, axis=0)
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb - rgb.min()
    denom = float(rgb.max()) + 1e-8
    rgb = rgb / denom
    return rgb


def _run_preview_sampling(
    *,
    model: nn.Module,
    scheduler: LinearNoiseScheduler,
    condition_latents: torch.Tensor,
    initial_noise: torch.Tensor,
    device: torch.device,
    use_cuda: bool,
    preview_steps: list[int],
) -> dict[int, torch.Tensor]:
    valid_steps = sorted(
        {step for step in preview_steps if 1 <= step <= scheduler.num_train_timesteps}
    )
    if not valid_steps:
        return {}

    model.eval()
    with torch.no_grad():
        condition = condition_latents.to(device, non_blocking=use_cuda)
        noisy_target = initial_noise.to(device, non_blocking=use_cuda)
        batch_size = condition.shape[0]
        snapshots: dict[int, torch.Tensor] = {}

        for denoise_step in range(1, scheduler.num_train_timesteps + 1):
            timestep_value = scheduler.num_train_timesteps - denoise_step
            timesteps = torch.full(
                (batch_size,),
                timestep_value,
                device=device,
                dtype=torch.long,
            )
            predicted_noise = model(
                noisy_query_latents=noisy_target,
                condition_latents=condition,
                timesteps=timesteps,
            )
            sqrt_alpha_t = scheduler.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = scheduler.sqrt_one_minus_alphas_cumprod[timesteps].view(
                -1, 1, 1, 1
            )
            x0_prediction = (noisy_target - sqrt_one_minus_alpha_t * predicted_noise) / (
                sqrt_alpha_t + 1e-8
            )

            if timestep_value > 0:
                previous_timesteps = timesteps - 1
                sqrt_alpha_prev = scheduler.sqrt_alphas_cumprod[previous_timesteps].view(
                    -1, 1, 1, 1
                )
                sqrt_one_minus_alpha_prev = scheduler.sqrt_one_minus_alphas_cumprod[
                    previous_timesteps
                ].view(-1, 1, 1, 1)
                noisy_target = (
                    sqrt_alpha_prev * x0_prediction + sqrt_one_minus_alpha_prev * predicted_noise
                )
            else:
                noisy_target = x0_prediction

            if denoise_step in valid_steps:
                snapshots[denoise_step] = noisy_target.detach().cpu()
                if len(snapshots) == len(valid_steps):
                    break

    return snapshots


def _log_preview_figure(
    *,
    mlflow_repo: MLFlowRepository,
    epoch_index: int,
    global_step: int,
    clean_targets: torch.Tensor,
    snapshots: dict[int, torch.Tensor],
    preview_steps: list[int],
) -> None:
    steps_to_plot = [step for step in preview_steps if step in snapshots]
    if clean_targets.numel() == 0 or not steps_to_plot:
        return

    rows = clean_targets.shape[0]
    cols = 1 + len(steps_to_plot)
    figure, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for row_index in range(rows):
        axes[row_index][0].imshow(_latent_to_display_image(clean_targets[row_index]))
        axes[row_index][0].set_axis_off()
        if row_index == 0:
            axes[row_index][0].set_title("original", fontsize=10)
        for col_index, step in enumerate(steps_to_plot, start=1):
            axes[row_index][col_index].imshow(_latent_to_display_image(snapshots[step][row_index]))
            axes[row_index][col_index].set_axis_off()
            if row_index == 0:
                axes[row_index][col_index].set_title(f"step {step}", fontsize=10)
    figure.tight_layout()
    mlflow_repo.mlflow.log_figure(
        figure,
        artifact_file=f"preview/epoch_{epoch_index + 1:04d}_global_step_{global_step:08d}.png",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    set_seed(settings.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    else:
        device = torch.device("cpu")
        use_cuda = False
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else settings.datasets_folder
    if dataset_dir is None:
        raise ValueError("Не задан путь к датасету. Укажите DATASETS_FOLDER или --dataset-dir.")

    train_dataloader = create_sft_dataloader(
        dataset_dir=dataset_dir,
        condition_key=settings.condition_key,
        target_key=settings.target_key,
        batch_size=settings.train_batch_size,
        num_workers=settings.train_num_workers,
        latent_channels=settings.latent_channels,
        condition_height=settings.condition_height,
        condition_width=settings.condition_width,
        target_height=settings.query_height,
        target_width=settings.query_width,
        pin_memory=False,
        split="train",
        train_ratio=0.9,
        split_seed=settings.seed,
    )
    val_dataloader = create_sft_dataloader(
        dataset_dir=dataset_dir,
        condition_key=settings.condition_key,
        target_key=settings.target_key,
        batch_size=settings.train_batch_size,
        num_workers=settings.train_num_workers,
        latent_channels=settings.latent_channels,
        condition_height=settings.condition_height,
        condition_width=settings.condition_width,
        target_height=settings.query_height,
        target_width=settings.query_width,
        pin_memory=False,
        split="val",
        train_ratio=0.9,
        split_seed=settings.seed,
    )

    model_config = build_config_from_settings(settings, architecture_name=args.architecture)
    model = build_model(model_config).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    max_steps = settings.train_max_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    cosine_eta_min = settings.learning_rate * 0.1
    if max_steps > 0:
        lr_total_steps = max_steps
        lr_step_mode = "per_step"
    else:
        try:
            steps_per_epoch = len(train_dataloader)
            if steps_per_epoch <= 0:
                raise ValueError("len(train_dataloader) должен быть > 0 для cosine decay.")
            lr_total_steps = settings.train_num_epochs * steps_per_epoch
            lr_step_mode = "per_step"
        except (TypeError, ValueError):
            lr_total_steps = settings.train_num_epochs
            lr_step_mode = "per_epoch"
            _log(
                "[lr] DataLoader не поддерживает len(), cosine decay применяется по эпохам.",
            )
    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=max(1, lr_total_steps),
        eta_min=cosine_eta_min,
    )
    scheduler = LinearNoiseScheduler(
        num_train_timesteps=settings.num_train_timesteps,
        beta_start=settings.beta_start,
        beta_end=settings.beta_end,
        device=device,
    )
    mlflow_repo = MLFlowRepository(
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
    )

    global_step = 0
    preview_interval_epochs = max(args.preview_every_n_epochs, 0)
    preview_images_count = max(args.preview_images_count, 0)
    preview_steps = [10, 100, 500, 1000]

    if max_steps > 0:
        _log(f"Training progress mode: by global steps ({max_steps} steps = 100%).")
    else:
        _log(f"Training progress mode: by epochs ({settings.train_num_epochs} epochs = 100%).")

    with mlflow_repo.start_run(run_name=args.run_name, tags={"pipeline": "dit-training"}):
        mlflow_repo.log_params(settings.mlflow_param_dict())
        mlflow_repo.log_params(
            {
                "architecture_name": model_config.architecture_name,
                "model_params_total": total_params,
                "model_params_trainable": trainable_params,
                "train_val_split_ratio": "0.9/0.1",
                "lr_scheduler": "cosine_decay",
                "lr_scheduler_mode": lr_step_mode,
                "lr_scheduler_total_steps": lr_total_steps,
                "lr_scheduler_eta_min": cosine_eta_min,
                "preview_every_n_epochs": preview_interval_epochs,
                "preview_images_count": preview_images_count,
                "preview_steps": ",".join(str(step) for step in preview_steps),
            }
        )
        mlflow_repo.log_config(
            {
                "run_name": args.run_name,
                "dataset_dir": str(dataset_dir),
                "device": str(device),
                "architecture_name": model_config.architecture_name,
                "model_params_total": total_params,
                "model_params_trainable": trainable_params,
                "preview_every_n_epochs": preview_interval_epochs,
                "preview_images_count": preview_images_count,
                "preview_steps": preview_steps,
                **settings.mlflow_param_dict(),
            }
        )

        fixed_preview_batch = _collect_fixed_preview_batch(
            dataloader=val_dataloader,
            max_images=preview_images_count,
        )
        if fixed_preview_batch is None and preview_interval_epochs > 0:
            _log(
                "[preview] Не удалось собрать фиксированный набор изображений, визуализация отключена."
            )
        if fixed_preview_batch is not None and preview_interval_epochs > 0:
            preview_generator = torch.Generator(device="cpu").manual_seed(settings.seed)
            fixed_preview_noise = torch.randn(
                fixed_preview_batch["target"].shape,
                generator=preview_generator,
                dtype=fixed_preview_batch["target"].dtype,
            )
        else:
            fixed_preview_noise = None

        for epoch in range(settings.train_num_epochs):
            for batch in train_dataloader:
                if max_steps > 0 and global_step >= max_steps:
                    break

                step_start_time = time.perf_counter()
                model.train()
                condition = batch["condition"].to(device, non_blocking=use_cuda)
                clean_target = batch["target"].to(device, non_blocking=use_cuda)

                timesteps = scheduler.sample_timesteps(
                    batch_size=clean_target.shape[0], device=device
                )
                noise = torch.randn_like(clean_target)
                noisy_target = scheduler.add_noise(clean_target, noise, timesteps)

                predicted_noise = model(
                    noisy_query_latents=noisy_target,
                    condition_latents=condition,
                    timesteps=timesteps,
                )
                loss = functional.mse_loss(predicted_noise, noise)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), settings.grad_clip_norm
                    ).item()
                )
                optimizer.step()
                if lr_step_mode == "per_step":
                    lr_scheduler.step()

                global_step += 1

                if global_step % settings.log_every_steps == 0:
                    with torch.no_grad():
                        train_metrics = compute_noise_metrics(
                            predicted_noise=predicted_noise,
                            noise=noise,
                            clean_target=clean_target,
                            noisy_target=noisy_target,
                            timesteps=timesteps,
                            scheduler=scheduler,
                        )

                    step_duration = max(time.perf_counter() - step_start_time, 1e-8)
                    train_metrics.update(
                        {
                            "gradient_norm": grad_norm,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "steps_per_second": 1.0 / step_duration,
                            "samples_per_second": clean_target.shape[0] / step_duration,
                        }
                    )
                    if max_steps > 0:
                        train_metrics["progress_percent"] = min(
                            (global_step / max_steps) * 100.0,
                            100.0,
                        )
                    if use_cuda:
                        train_metrics["gpu_memory_allocated_mb"] = float(
                            torch.cuda.memory_allocated(device=device)
                        ) / (1024**2)
                        train_metrics["gpu_memory_reserved_mb"] = float(
                            torch.cuda.memory_reserved(device=device)
                        ) / (1024**2)

                    val_metrics = evaluate_on_validation(
                        model=model,
                        dataloader=val_dataloader,
                        scheduler=scheduler,
                        device=device,
                        use_cuda=use_cuda,
                    )
                    prefixed_train_metrics = {
                        f"train_{key}": float(value) for key, value in train_metrics.items()
                    }
                    prefixed_val_metrics = {
                        f"val_{key}": float(value) for key, value in val_metrics.items()
                    }
                    mlflow_repo.log_metrics(
                        metrics={**prefixed_train_metrics, **prefixed_val_metrics},
                        step=global_step,
                    )
                    if max_steps > 0:
                        _log(
                            f"[train] progress {train_metrics['progress_percent']:.2f}% "
                            f"({global_step}/{max_steps} steps), "
                            f"train_loss={train_metrics['loss_mse']:.6f}, "
                            f"val_loss={val_metrics.get('loss_mse', float('nan')):.6f}"
                        )
                    elif "loss_mse" in val_metrics:
                        _log(
                            f"[train/val] step={global_step} "
                            f"train_loss={train_metrics['loss_mse']:.6f}, "
                            f"val_loss={val_metrics['loss_mse']:.6f}"
                        )

                if global_step % settings.checkpoint_every_steps == 0:
                    mlflow_repo.log_checkpoint(model=model, step=global_step)

                # Раннее освобождение ссылок уменьшает пиковое потребление памяти между шагами.
                del (
                    batch,
                    condition,
                    clean_target,
                    timesteps,
                    noise,
                    noisy_target,
                    predicted_noise,
                    loss,
                    grad_norm,
                )

            if max_steps == 0:
                epoch_progress = ((epoch + 1) / settings.train_num_epochs) * 100.0
                _log(
                    f"[train] progress {epoch_progress:.2f}% "
                    f"({epoch + 1}/{settings.train_num_epochs} epochs)"
                )
            if lr_step_mode == "per_epoch":
                lr_scheduler.step()
            if (
                preview_interval_epochs > 0
                and (epoch + 1) % preview_interval_epochs == 0
                and fixed_preview_batch is not None
                and fixed_preview_noise is not None
            ):
                preview_snapshots = _run_preview_sampling(
                    model=model,
                    scheduler=scheduler,
                    condition_latents=fixed_preview_batch["condition"],
                    initial_noise=fixed_preview_noise,
                    device=device,
                    use_cuda=use_cuda,
                    preview_steps=preview_steps,
                )
                _log_preview_figure(
                    mlflow_repo=mlflow_repo,
                    epoch_index=epoch,
                    global_step=global_step,
                    clean_targets=fixed_preview_batch["target"],
                    snapshots=preview_snapshots,
                    preview_steps=preview_steps,
                )
                _log(
                    f"[preview] logged denoising figure for epoch={epoch + 1}, "
                    f"images={fixed_preview_batch['target'].shape[0]}"
                )
            if max_steps > 0 and global_step >= max_steps:
                break

        mlflow_repo.log_model_weights(model=model, artifact_relpath="weights/final_model.pt")
        registration = mlflow_repo.register_final_model(
            model=model,
            registered_model_name=args.registered_model_name,
            artifact_path="models/final",
        )
        mlflow_repo.log_params(
            {
                "registered_model_name": registration["name"],
                "registered_model_version": registration["version"],
                "registered_model_uri": registration["model_uri"],
            }
        )
        mlflow_repo.log_metrics(metrics={"final_step": float(global_step)}, step=global_step)

    _log(
        "Registered model in MLflow Registry: "
        f"{registration['name']} v{registration['version']} ({registration['model_uri']})"
    )
    _log(f"Training finished. Global steps: {global_step}")


if __name__ == "__main__":
    main()
