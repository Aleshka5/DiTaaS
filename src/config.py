from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    movies_bucket_name: str = Field(default="movies", alias="MOVIES_BUCKET_NAME")
    mlflow_bucket_name: str = Field(default="mlflow", alias="MLFLOW_BUCKET_NAME")

    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field(default="ru-1", alias="AWS_DEFAULT_REGION")
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    mlflow_s3_endpoint_url: str | None = Field(default=None, alias="MLFLOW_S3_ENDPOINT_URL")

    gpu_support: bool = Field(default=True, alias="GPU_SUPPORT")
    datasets_folder: Path | None = Field(default=None, alias="DATASETS_FOLDER")

    mlflow_tracking_uri: str | None = Field(default=None, alias="MLFLOW_TRACKING_URI")
    mlflow_registry_uri: str | None = Field(default=None, alias="MLFLOW_REGISTRY_URI")
    mlflow_experiment_name: str = Field(default="DiT", alias="MLFLOW_EXPERIMENT_NAME")
    mlflow_registered_model_name: str = Field(
        default="DiTModel",
        alias="MLFLOW_REGISTERED_MODEL_NAME",
    )
    dataset_download_dir: Path = Field(
        default=PROJECT_ROOT / "dataset", alias="DATASET_DOWNLOAD_DIR"
    )
    mlflow_dataset_artifact_path: str = Field(
        default="encoded_sft",
        alias="MLFLOW_DATASET_ARTIFACT_PATH",
    )
    mlflow_dataset_run_ids: str = Field(
        default="2bafa11693594d2b8483ef9cf9ba2fe1,0bfac512ad5045689a0f21a3b8b34b63,ef5ae334f1d94ce2b7f9dc508572a2a9,8ed387a85cfb42ffa43cc29a77d1e7b5,123eee7e782b45e3907ce825d7235674,70b9e115d904459e93f0a1a447774196,7e7fc71c21f84156875244fbcbe69a7d,694ba9ba86384c158413f371b1a69236,f1cde66a02a04e6fa64424a0f777cd10,63a5f5bcc4eb4d01bf18fa67d5d13ddb",
        alias="MLFLOW_DATASET_RUN_IDS",
    )

    latent_channels: int = Field(default=16, alias="LATENT_CHANNELS")
    model_architecture: str = Field(default="dit", alias="MODEL_ARCHITECTURE")
    condition_key: str = Field(default="latents_54x30", alias="CONDITION_KEY")
    target_key: str = Field(default="latents_16x30", alias="TARGET_KEY")
    condition_height: int = Field(default=54, alias="CONDITION_HEIGHT")
    condition_width: int = Field(default=30, alias="CONDITION_WIDTH")
    query_height: int = Field(default=16, alias="QUERY_HEIGHT")
    query_width: int = Field(default=30, alias="QUERY_WIDTH")
    hidden_size: int = Field(default=256, alias="HIDDEN_SIZE")
    num_attention_heads: int = Field(default=8, alias="NUM_ATTENTION_HEADS")
    num_transformer_blocks: int = Field(default=6, alias="NUM_TRANSFORMER_BLOCKS")
    mlp_ratio: float = Field(default=4.0, alias="MLP_RATIO")
    dropout: float = Field(default=0.0, alias="DROPOUT")
    dit_v2_query_patch_size: int = Field(default=2, alias="DIT_V2_QUERY_PATCH_SIZE")
    dit_v2_condition_patch_size: int = Field(default=2, alias="DIT_V2_CONDITION_PATCH_SIZE")

    train_batch_size: int = Field(default=8, alias="TRAIN_BATCH_SIZE")
    train_num_workers: int = Field(default=0, alias="TRAIN_NUM_WORKERS")
    train_num_epochs: int = Field(default=5, alias="TRAIN_NUM_EPOCHS")
    train_max_steps: int = Field(default=0, alias="TRAIN_MAX_STEPS")
    learning_rate: float = Field(default=1e-4, alias="LEARNING_RATE")
    weight_decay: float = Field(default=1e-2, alias="WEIGHT_DECAY")
    grad_clip_norm: float = Field(default=1.0, alias="GRAD_CLIP_NORM")
    log_every_steps: int = Field(default=10, alias="LOG_EVERY_STEPS")
    checkpoint_every_steps: int = Field(default=1000, alias="CHECKPOINT_EVERY_STEPS")
    seed: int = Field(default=42, alias="SEED")

    num_train_timesteps: int = Field(default=1000, alias="NUM_TRAIN_TIMESTEPS")
    beta_start: float = Field(default=1e-4, alias="BETA_START")
    beta_end: float = Field(default=2e-2, alias="BETA_END")

    @property
    def default_device(self) -> str:
        return "cuda" if self.gpu_support else "cpu"

    @property
    def dataset_run_ids(self) -> list[str]:
        return [
            run_id.strip() for run_id in self.mlflow_dataset_run_ids.split(",") if run_id.strip()
        ]

    def mlflow_param_dict(self) -> dict[str, Any]:
        return {
            "latent_channels": self.latent_channels,
            "architecture_name": self.model_architecture,
            "condition_key": self.condition_key,
            "target_key": self.target_key,
            "condition_height": self.condition_height,
            "condition_width": self.condition_width,
            "query_height": self.query_height,
            "query_width": self.query_width,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_transformer_blocks": self.num_transformer_blocks,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "query_patch_size": self.dit_v2_query_patch_size,
            "condition_patch_size": self.dit_v2_condition_patch_size,
            "train_batch_size": self.train_batch_size,
            "train_num_workers": self.train_num_workers,
            "train_num_epochs": self.train_num_epochs,
            "train_max_steps": self.train_max_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
            "log_every_steps": self.log_every_steps,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "seed": self.seed,
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "mlflow_registered_model_name": self.mlflow_registered_model_name,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
