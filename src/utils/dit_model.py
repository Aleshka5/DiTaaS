from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class BaseModelConfig:
    architecture_name: str = "dit"
    latent_channels: int = 16
    max_timestep: int = 100


@dataclass(slots=True)
class DiTModelConfig(BaseModelConfig):
    condition_height: int = 54
    condition_width: int = 30
    query_height: int = 16
    query_width: int = 30
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_transformer_blocks: int = 1
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @classmethod
    def from_settings(cls, settings, architecture_name: str = "dit") -> "DiTModelConfig":
        return cls(
            architecture_name=architecture_name,
            latent_channels=settings.latent_channels,
            condition_height=settings.condition_height,
            condition_width=settings.condition_width,
            query_height=settings.query_height,
            query_width=settings.query_width,
            hidden_size=settings.hidden_size,
            num_attention_heads=settings.num_attention_heads,
            num_transformer_blocks=settings.num_transformer_blocks,
            mlp_ratio=settings.mlp_ratio,
            dropout=settings.dropout,
            max_timestep=settings.num_train_timesteps,
        )

    @property
    def condition_tokens(self) -> int:
        return self.condition_height * self.condition_width

    @property
    def query_tokens(self) -> int:
        return self.query_height * self.query_width


class TransformerCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, query_states: Tensor, condition_states: Tensor) -> Tensor:
        residual = query_states
        normalized = self.self_norm(query_states)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        query_states = residual + attended

        residual = query_states
        normalized = self.cross_norm(query_states)
        attended, _ = self.cross_attention(
            normalized,
            condition_states,
            condition_states,
            need_weights=False,
        )
        query_states = residual + attended

        residual = query_states
        query_states = residual + self.mlp(self.mlp_norm(query_states))
        return query_states


class DiTModel(nn.Module):
    """Базовый DiT-подобный CAT для латентов 16-канального пространства."""

    def __init__(self, config: DiTModelConfig) -> None:
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(config.latent_channels, config.hidden_size)
        self.condition_projection = nn.Linear(config.latent_channels, config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.latent_channels)

        self.query_positional_embedding = nn.Parameter(
            torch.zeros(1, config.query_tokens, config.hidden_size)
        )
        self.condition_positional_embedding = nn.Parameter(
            torch.zeros(1, config.condition_tokens, config.hidden_size)
        )
        self.timestep_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                TransformerCrossAttentionBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_transformer_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.query_positional_embedding, std=0.02)
        nn.init.normal_(self.condition_positional_embedding, std=0.02)

    def _to_tokens(self, latents: Tensor, *, expected_height: int, expected_width: int) -> Tensor:
        if latents.ndim != 4:
            raise ValueError(
                f"Ожидается тензор вида [B, C, H, W] или [B, H, W, C], получено: {tuple(latents.shape)}"
            )

        if latents.shape[1] == self.config.latent_channels:
            channels_first = latents
        elif latents.shape[-1] == self.config.latent_channels:
            channels_first = latents.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(
                f"Не удалось определить положение каналов в тензоре: {tuple(latents.shape)}"
            )

        if channels_first.shape[2] != expected_height or channels_first.shape[3] != expected_width:
            if channels_first.shape[2] == expected_width and channels_first.shape[3] == expected_height:
                channels_first = channels_first.transpose(2, 3).contiguous()
            else:
                raise ValueError(
                    "Некорректный spatial shape латентов. "
                    f"Ожидается [{expected_height}, {expected_width}], получено "
                    f"[{channels_first.shape[2]}, {channels_first.shape[3]}]."
                )

        return channels_first.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _sinusoidal_timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
        half_dim = embedding_dim // 2
        exponent = -torch.log(torch.tensor(10000.0, device=timesteps.device)) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return embedding

    def forward(
        self, noisy_query_latents: Tensor, condition_latents: Tensor, timesteps: Tensor
    ) -> Tensor:
        query_tokens = self._to_tokens(
            noisy_query_latents,
            expected_height=self.config.query_height,
            expected_width=self.config.query_width,
        )
        condition_tokens = self._to_tokens(
            condition_latents,
            expected_height=self.config.condition_height,
            expected_width=self.config.condition_width,
        )

        query_states = self.input_projection(query_tokens) + self.query_positional_embedding
        condition_states = (
            self.condition_projection(condition_tokens) + self.condition_positional_embedding
        )

        timestep_embedding = self._sinusoidal_timestep_embedding(
            timesteps=timesteps,
            embedding_dim=self.config.hidden_size,
        )
        timestep_embedding = self.timestep_mlp(timestep_embedding).unsqueeze(1)
        query_states = query_states + timestep_embedding

        for block in self.blocks:
            query_states = block(query_states, condition_states)

        query_states = self.final_norm(query_states)
        predicted_noise = self.output_projection(query_states)
        predicted_noise = predicted_noise.transpose(1, 2).reshape(
            noisy_query_latents.shape[0],
            self.config.latent_channels,
            self.config.query_height,
            self.config.query_width,
        )
        return predicted_noise
