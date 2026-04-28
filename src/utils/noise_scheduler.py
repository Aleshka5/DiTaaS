from __future__ import annotations

import torch
from torch import Tensor


class LinearNoiseScheduler:
    """Линейный scheduler зашумления для diffusion-процесса."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps должен быть > 0.")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("Ожидается 0 < beta_start < beta_end < 1.")

        self.num_train_timesteps = num_train_timesteps
        self.device = torch.device(device)
        self.betas = torch.linspace(
            beta_start,
            beta_end,
            steps=num_train_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample_timesteps(self, batch_size: int, device: str | torch.device) -> Tensor:
        return torch.randint(
            low=0,
            high=self.num_train_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def add_noise(self, clean_latents: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        if clean_latents.shape != noise.shape:
            raise ValueError(
                "clean_latents и noise должны иметь одинаковую форму, "
                f"получено: {tuple(clean_latents.shape)} и {tuple(noise.shape)}."
            )
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * clean_latents + sqrt_one_minus_alpha * noise
