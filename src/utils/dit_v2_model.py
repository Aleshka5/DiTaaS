from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.utils.dit_model import BaseModelConfig


def _modulate(normalized: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@dataclass(slots=True)
class DiTV2ModelConfig(BaseModelConfig):
    condition_height: int = 54
    condition_width: int = 30
    query_height: int = 16
    query_width: int = 30
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_transformer_blocks: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    query_patch_size: int = 2
    condition_patch_size: int = 2

    @classmethod
    def from_settings(cls, settings, architecture_name: str = "dit_v2") -> "DiTV2ModelConfig":
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
            query_patch_size=getattr(settings, "dit_v2_query_patch_size", 2),
            condition_patch_size=getattr(settings, "dit_v2_condition_patch_size", 2),
            max_timestep=settings.num_train_timesteps,
        )


class DiTV2Block(nn.Module):
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
        self.ada_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 9),
        )
        nn.init.zeros_(self.ada_modulation[1].weight)
        nn.init.zeros_(self.ada_modulation[1].bias)

    def forward(self, query_states: Tensor, condition_states: Tensor, timestep_states: Tensor) -> Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_xa,
            scale_xa,
            gate_xa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.ada_modulation(timestep_states).chunk(9, dim=1)

        normalized = _modulate(self.self_norm(query_states), shift_msa, scale_msa)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        query_states = query_states + gate_msa.unsqueeze(1) * attended

        normalized = _modulate(self.cross_norm(query_states), shift_xa, scale_xa)
        attended, _ = self.cross_attention(
            normalized,
            condition_states,
            condition_states,
            need_weights=False,
        )
        query_states = query_states + gate_xa.unsqueeze(1) * attended

        normalized = _modulate(self.mlp_norm(query_states), shift_mlp, scale_mlp)
        query_states = query_states + gate_mlp.unsqueeze(1) * self.mlp(normalized)
        return query_states


class DiTV2Model(nn.Module):
    """Улучшенный DiT: patching + 2D sin-cos pos embed + AdaLN-Zero."""

    def __init__(self, config: DiTV2ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.query_patch_embedding = nn.Conv2d(
            in_channels=config.latent_channels,
            out_channels=config.hidden_size,
            kernel_size=config.query_patch_size,
            stride=config.query_patch_size,
        )
        self.condition_patch_embedding = nn.Conv2d(
            in_channels=config.latent_channels,
            out_channels=config.hidden_size,
            kernel_size=config.condition_patch_size,
            stride=config.condition_patch_size,
        )
        self.timestep_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                DiTV2Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_transformer_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size * 2),
        )
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.latent_channels * (config.query_patch_size**2),
        )

    @staticmethod
    def _to_channels_first(latents: Tensor, *, latent_channels: int, source: str) -> Tensor:
        if latents.ndim != 4:
            raise ValueError(
                f"{source}: ожидается тензор [B, C, H, W] или [B, H, W, C], получено: {tuple(latents.shape)}"
            )
        if latents.shape[1] == latent_channels:
            return latents
        if latents.shape[-1] == latent_channels:
            return latents.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"{source}: не удалось определить положение каналов для latent_channels={latent_channels}, "
            f"shape={tuple(latents.shape)}."
        )

    @staticmethod
    def _sinusoidal_timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
        half_dim = embedding_dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent
        )
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return embedding

    @staticmethod
    def _sincos_1d(positions: Tensor, dim: int) -> Tensor:
        if dim <= 0:
            raise ValueError(f"Ожидается положительная размерность для sin-cos, получено: {dim}")
        half_dim = dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=positions.device, dtype=torch.float32) * exponent
        )
        args = positions.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return embedding

    @classmethod
    def _sincos_2d(cls, height: int, width: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if dim % 2 != 0:
            raise ValueError(f"hidden_size для 2D pos embedding должен быть четным, получено: {dim}")
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        half = dim // 2
        y_embed = cls._sincos_1d(y.reshape(-1), half)
        x_embed = cls._sincos_1d(x.reshape(-1), half)
        return torch.cat([y_embed, x_embed], dim=1).unsqueeze(0).to(dtype=dtype)

    def _patchify(self, latents: Tensor, *, patch_embedding: nn.Conv2d, patch_size: int, source: str) -> Tensor:
        height, width = latents.shape[2], latents.shape[3]
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"{source}: spatial shape [{height}, {width}] не делится на patch_size={patch_size}."
            )
        return patch_embedding(latents)

    def _tokens_from_patches(self, patches: Tensor) -> Tensor:
        return patches.flatten(2).transpose(1, 2).contiguous()

    def _unpatchify(self, tokens: Tensor, *, output_height: int, output_width: int) -> Tensor:
        batch_size = tokens.shape[0]
        query_patch_size = self.config.query_patch_size
        channels = self.config.latent_channels
        patches_h = output_height // query_patch_size
        patches_w = output_width // query_patch_size
        expected_tokens = patches_h * patches_w
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                "Некорректное число query tokens для восстановления пространственной карты: "
                f"ожидается {expected_tokens}, получено {tokens.shape[1]}."
            )
        projected = self.output_projection(tokens)
        projected = projected.view(
            batch_size,
            patches_h,
            patches_w,
            channels,
            query_patch_size,
            query_patch_size,
        )
        return projected.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, output_height, output_width)

    def forward(
        self, noisy_query_latents: Tensor, condition_latents: Tensor, timesteps: Tensor
    ) -> Tensor:
        query_channels_first = self._to_channels_first(
            noisy_query_latents,
            latent_channels=self.config.latent_channels,
            source="noisy_query_latents",
        )
        condition_channels_first = self._to_channels_first(
            condition_latents,
            latent_channels=self.config.latent_channels,
            source="condition_latents",
        )

        query_patches = self._patchify(
            query_channels_first,
            patch_embedding=self.query_patch_embedding,
            patch_size=self.config.query_patch_size,
            source="noisy_query_latents",
        )
        condition_patches = self._patchify(
            condition_channels_first,
            patch_embedding=self.condition_patch_embedding,
            patch_size=self.config.condition_patch_size,
            source="condition_latents",
        )

        query_states = self._tokens_from_patches(query_patches)
        condition_states = self._tokens_from_patches(condition_patches)

        query_positional = self._sincos_2d(
            query_patches.shape[2],
            query_patches.shape[3],
            self.config.hidden_size,
            device=query_states.device,
            dtype=query_states.dtype,
        )
        condition_positional = self._sincos_2d(
            condition_patches.shape[2],
            condition_patches.shape[3],
            self.config.hidden_size,
            device=condition_states.device,
            dtype=condition_states.dtype,
        )
        query_states = query_states + query_positional
        condition_states = condition_states + condition_positional

        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.shape[0] == 1 and query_states.shape[0] > 1:
            timesteps = timesteps.expand(query_states.shape[0])
        if timesteps.shape[0] != query_states.shape[0]:
            raise ValueError(
                "Размер timesteps должен совпадать с batch size: "
                f"timesteps={timesteps.shape[0]}, batch={query_states.shape[0]}."
            )

        timestep_states = self._sinusoidal_timestep_embedding(
            timesteps=timesteps,
            embedding_dim=self.config.hidden_size,
        ).to(dtype=query_states.dtype)
        timestep_states = self.timestep_mlp(timestep_states)

        for block in self.blocks:
            query_states = block(query_states, condition_states, timestep_states)

        shift, scale = self.final_modulation(timestep_states).chunk(2, dim=1)
        query_states = _modulate(self.final_norm(query_states), shift, scale)
        return self._unpatchify(
            query_states,
            output_height=query_channels_first.shape[2],
            output_width=query_channels_first.shape[3],
        )
