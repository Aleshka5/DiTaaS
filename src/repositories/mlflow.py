from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from src.config import get_settings
from src.utils.dit_model import BaseModelConfig, DiTModelConfig
from src.utils.dit_v2_model import DiTV2ModelConfig
from src.utils.model_archive import build_model, get_model_entry


class DiTPyFuncModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc-обертка для DiT с именованными входами."""

    def load_context(self, context: Any) -> None:
        config_path = Path(context.artifacts["model_config"])
        weights_path = Path(context.artifacts["weights"])
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        architecture_name = config_payload.get("architecture_name", "dit")
        archive_entry = get_model_entry(architecture_name)
        model_config = archive_entry.config_cls(**config_payload)
        self.model = build_model(model_config).to("cpu")
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @staticmethod
    def _to_numpy_payload(model_input: Any) -> dict[str, np.ndarray]:
        if isinstance(model_input, dict):
            payload = model_input
        elif hasattr(model_input, "to_dict"):
            payload = model_input.to_dict(orient="list")
        else:
            raise TypeError(
                "model_input должен быть dict или DataFrame с ключами "
                "'noisy_query_latents', 'condition_latents', 'timesteps'."
            )

        normalized: dict[str, np.ndarray] = {}
        for key in ("noisy_query_latents", "condition_latents", "timesteps"):
            if key not in payload:
                raise KeyError(f"В model_input отсутствует обязательный ключ '{key}'.")
            value = payload[key]
            if isinstance(value, list) and len(value) == 1 and not np.isscalar(value[0]):
                value = value[0]
            normalized[key] = np.asarray(value)
        return normalized

    def predict(self, context: Any, model_input: Any, params: dict[str, Any] | None = None) -> np.ndarray:
        payload = self._to_numpy_payload(model_input)
        noisy_query_latents = torch.as_tensor(payload["noisy_query_latents"], dtype=torch.float32)
        condition_latents = torch.as_tensor(payload["condition_latents"], dtype=torch.float32)
        timesteps = torch.as_tensor(payload["timesteps"], dtype=torch.long)

        if noisy_query_latents.ndim == 3:
            noisy_query_latents = noisy_query_latents.unsqueeze(0)
        if condition_latents.ndim == 3:
            condition_latents = condition_latents.unsqueeze(0)
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)

        with torch.no_grad():
            predicted_noise = self.model(
                noisy_query_latents=noisy_query_latents,
                condition_latents=condition_latents,
                timesteps=timesteps,
            )
        return predicted_noise.detach().cpu().numpy()


class MLFlowRepository:
    """Обертка над MLflow для обучения/сохранения DiT и выгрузки датасетов."""

    def __init__(self, tracking_uri: str | None = None, registry_uri: str | None = None) -> None:
        import mlflow

        self.mlflow = mlflow
        self.settings = get_settings()
        self.tracking_uri = tracking_uri or self.settings.mlflow_tracking_uri
        self.registry_uri = registry_uri or self.settings.mlflow_registry_uri

        if not self.tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI не задан. Укажите его в .env или CLI.")

        self._apply_environment()
        self.mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            self.mlflow.set_registry_uri(self.registry_uri)

        self.experiment_name = self.settings.mlflow_experiment_name
        self.experiment_id = self._ensure_experiment_id()
        self.client = self.mlflow.tracking.MlflowClient()

    def _apply_environment(self) -> None:
        env_overrides = {
            "AWS_ACCESS_KEY_ID": self.settings.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.settings.aws_secret_access_key,
            "AWS_DEFAULT_REGION": self.settings.aws_default_region,
            "S3_ENDPOINT_URL": self.settings.s3_endpoint_url,
            "MLFLOW_S3_ENDPOINT_URL": self.settings.mlflow_s3_endpoint_url,
        }
        for key, value in env_overrides.items():
            if value and not os.getenv(key):
                os.environ[key] = value

    def _ensure_experiment_id(self) -> str:
        experiment = self.mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        return self.mlflow.create_experiment(self.experiment_name)

    @contextmanager
    def start_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
    ) -> Iterator[Any]:
        with self.mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name) as run:
            if tags:
                self.mlflow.set_tags(tags)
            yield run

    def log_params(self, params: dict[str, Any]) -> None:
        normalized = {key: str(value) for key, value in params.items()}
        self.mlflow.log_params(normalized)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        clean_metrics = {key: float(value) for key, value in metrics.items()}
        self.mlflow.log_metrics(clean_metrics, step=step)

    def log_config(self, config: dict[str, Any], artifact_path: str = "configs") -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "train_config.json"
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
            self.mlflow.log_artifact(str(config_path), artifact_path=artifact_path)

    def log_model_weights(self, model: torch.nn.Module, artifact_relpath: str) -> None:
        self.log_weights_state_dict(model.state_dict(), artifact_relpath=artifact_relpath)

    def log_weights_state_dict(self, state_dict: dict[str, torch.Tensor], artifact_relpath: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            local_path = temp_root / artifact_relpath
            local_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, local_path)
            artifact_root = str(Path(artifact_relpath).parent).replace("\\", "/")
            if artifact_root == ".":
                artifact_root = ""
            self.mlflow.log_artifact(str(local_path), artifact_path=artifact_root or None)

    def log_checkpoint(self, model: torch.nn.Module, step: int, artifact_dir: str = "checkpoints") -> None:
        checkpoint_name = f"step_{step:08d}.pt"
        self.log_model_weights(
            model=model,
            artifact_relpath=f"{artifact_dir}/{checkpoint_name}",
        )

    def _build_dit_signature(
        self, model_config: DiTModelConfig | DiTV2ModelConfig
    ) -> tuple[ModelSignature, dict[str, np.ndarray]]:
        signature = ModelSignature(
            inputs=Schema(
                [
                    TensorSpec(
                        np.dtype(np.float32),
                        (-1, model_config.latent_channels, model_config.query_height, model_config.query_width),
                        name="noisy_query_latents",
                    ),
                    TensorSpec(
                        np.dtype(np.float32),
                        (
                            -1,
                            model_config.latent_channels,
                            model_config.condition_height,
                            model_config.condition_width,
                        ),
                        name="condition_latents",
                    ),
                    TensorSpec(np.dtype(np.int64), (-1,), name="timesteps"),
                ]
            ),
            outputs=Schema(
                [
                    TensorSpec(
                        np.dtype(np.float32),
                        (-1, model_config.latent_channels, model_config.query_height, model_config.query_width),
                        name="predicted_noise",
                    )
                ]
            ),
        )
        input_example = {
            "noisy_query_latents": np.zeros(
                (1, model_config.latent_channels, model_config.query_height, model_config.query_width),
                dtype=np.float32,
            ),
            "condition_latents": np.zeros(
                (
                    1,
                    model_config.latent_channels,
                    model_config.condition_height,
                    model_config.condition_width,
                ),
                dtype=np.float32,
            ),
            "timesteps": np.zeros((1,), dtype=np.int64),
        }
        return signature, input_example

    def register_final_model(
        self,
        model: torch.nn.Module,
        registered_model_name: str,
        artifact_path: str = "models/final",
    ) -> dict[str, str]:
        if not registered_model_name:
            raise ValueError("registered_model_name не может быть пустым.")
        active_run = self.mlflow.active_run()
        if active_run is None:
            raise RuntimeError("Нет активного MLflow run для регистрации модели.")
        if not hasattr(model, "config"):
            raise TypeError(
                "Ожидается torch.nn.Module с полем config, "
                f"получено: {type(model).__name__}"
            )
        model_config = model.config
        if not isinstance(model_config, BaseModelConfig):
            raise TypeError(
                "Ожидается BaseModelConfig в model.config, "
                f"получено: {type(model_config).__name__}"
            )

        model.eval()
        if model_config.architecture_name in {"dit", "dit_v2"}:
            if not isinstance(model_config, (DiTModelConfig, DiTV2ModelConfig)):
                raise TypeError(
                    "Для architecture_name='dit'/'dit_v2' ожидается DiTModelConfig или DiTV2ModelConfig, "
                    f"получено: {type(model_config).__name__}"
                )
            signature, input_example = self._build_dit_signature(model_config)
        else:
            raise NotImplementedError(
                "Регистрация MLflow signature реализована только для архитектур 'dit' и 'dit_v2'."
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            weights_path = temp_root / "final_model_state_dict.pt"
            config_path = temp_root / "model_config.json"
            torch.save(model.state_dict(), weights_path)
            config_path.write_text(
                json.dumps(asdict(model.config), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            model_info = self.mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=DiTPyFuncModel(),
                artifacts={
                    "weights": str(weights_path),
                    "model_config": str(config_path),
                },
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
                metadata={
                    "framework": "pytorch",
                    "model_type": model.__class__.__name__,
                    "architecture_name": model_config.architecture_name,
                },
            )

        registered_version = getattr(model_info, "registered_model_version", None)
        model_uri = model_info.model_uri
        if not registered_version:
            versions = self.client.search_model_versions(f"name = '{registered_model_name}'")
            matching = [item for item in versions if item.run_id == active_run.info.run_id]
            if matching:
                registered_version = str(max(int(item.version) for item in matching))

        return {
            "name": registered_model_name,
            "version": str(registered_version) if registered_version else "unknown",
            "model_uri": model_uri,
        }

    def download_sft_dataset_parts(
        self,
        run_ids: list[str],
        output_dir: str | Path,
        artifact_subpath: str = "encoded_sft",
    ) -> list[Path]:
        if not run_ids:
            raise ValueError("Список run_ids пуст.")

        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)

        collected_files: list[Path] = []
        for run_id in run_ids:
            with tempfile.TemporaryDirectory() as temp_dir:
                downloaded_path = self.mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path=artifact_subpath,
                    dst_path=temp_dir,
                )
                for sft_file in sorted(Path(downloaded_path).rglob("*.sft")):
                    target = destination / sft_file.name
                    if target.exists():
                        target.unlink()
                    shutil.copy2(sft_file, target)
                    collected_files.append(target)
        return collected_files
