from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class PLMGuidedAutoencoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, config: dict[str, Any]) -> None:
        super().__init__()
        ae_cfg = config["autoencoder"]
        data_cfg = config["data"]
        plm_dim = int(config["plm"]["embedding_dim"])
        latent_dim = int(ae_cfg["latent_dim"])
        hidden_dim = int(ae_cfg["hidden_dim"])
        max_length = int(data_cfg["max_length"])
        max_position = max_length + 2
        self.pad_id = pad_id
        self.max_length = max_length
        self.latent_dim = latent_dim

        self.plm_to_latent = nn.Sequential(
            nn.LayerNorm(plm_dim),
            nn.Linear(plm_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(ae_cfg["dropout"])),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.latent_to_plm = nn.Linear(latent_dim, plm_dim)
        self.latent_to_memory = nn.Linear(latent_dim, hidden_dim)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_position, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=int(ae_cfg["attention_heads"]),
            dim_feedforward=int(ae_cfg["feedforward_dim"]),
            dropout=float(ae_cfg["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(ae_cfg["decoder_layers"]))
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.sequence_projector = nn.Linear(latent_dim, hidden_dim)
        self.taste_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(ae_cfg["dropout"])),
            nn.Linear(hidden_dim, 2),
        )

    def encode_plm(self, plm_embeddings: torch.Tensor) -> torch.Tensor:
        return self.plm_to_latent(plm_embeddings)

    def decode(self, decoder_input_ids: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)
        target = (
            self.token_embedding(decoder_input_ids)
            + self.position_embedding(positions).unsqueeze(0)
        )
        memory = self.latent_to_memory(latents)
        causal = nn.Transformer.generate_square_subsequent_mask(
            decoder_input_ids.shape[1],
            device=decoder_input_ids.device,
        )
        hidden = self.decoder(tgt=target, memory=memory, tgt_mask=causal)
        return self.output_projection(hidden)

    def sequence_embedding(self, latents: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = sequence_mask(lengths, latents.shape[1]).unsqueeze(-1).to(latents.dtype)
        pooled = (latents * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.sequence_projector(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        plm_embeddings: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latents = self.encode_plm(plm_embeddings)
        logits = self.decode(input_ids[:, :-1], latents)
        seq_embedding = self.sequence_embedding(latents, lengths)
        return {
            "latents": latents,
            "logits": logits,
            "reconstructed_plm": self.latent_to_plm(latents),
            "sequence_embedding": seq_embedding,
            "taste_logits": self.taste_classifier(seq_embedding),
        }


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(1, half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale
        )
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class LatentDiffusionDenoiser(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        diff_cfg = config["diffusion"]
        data_cfg = config["data"]
        latent_dim = int(diff_cfg["latent_dim"])
        hidden_dim = int(diff_cfg["hidden_dim"])
        time_dim = int(diff_cfg["time_embedding_dim"])
        max_length = int(data_cfg["max_length"])
        self.max_length = max_length
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.length_embedding = nn.Embedding(max_length + 1, hidden_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=float(diff_cfg["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(diff_cfg["layers"]))
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, noisy_latents: torch.Tensor, timesteps: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.clamp(min=1, max=self.max_length)
        hidden = self.input_projection(noisy_latents)
        condition = self.length_embedding(lengths) + self.time_mlp(timesteps)
        hidden = hidden + condition.unsqueeze(1)
        padding_mask = ~sequence_mask(lengths, noisy_latents.shape[1])
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.output_projection(encoded)


class DiffusionSchedule:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        diff_cfg = config["diffusion"]
        self.timesteps = int(diff_cfg["timesteps"])
        betas = torch.linspace(
            float(diff_cfg["beta_start"]),
            float(diff_cfg["beta_end"]),
            self.timesteps,
            device=device,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def add_noise(self, latents: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        scale_signal = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1)
        scale_noise = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1)
        return scale_signal * latents + scale_noise * noise

    @torch.no_grad()
    def sample(
        self,
        denoiser: LatentDiffusionDenoiser,
        shape: tuple[int, int, int],
        lengths: torch.Tensor,
        steps: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        device = lengths.device
        latents = torch.randn(shape, device=device) * float(temperature)
        indices = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        for timestep in indices:
            batch_t = torch.full((shape[0],), int(timestep.item()), dtype=torch.long, device=device)
            predicted_noise = denoiser(latents, batch_t, lengths)
            alpha_t = self.alphas[batch_t].view(-1, 1, 1)
            alpha_bar_t = self.alpha_bars[batch_t].view(-1, 1, 1)
            beta_t = self.betas[batch_t].view(-1, 1, 1)
            mean = (latents - beta_t * predicted_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
            if timestep.item() > 0:
                latents = mean + torch.sqrt(beta_t) * torch.randn_like(latents)
            else:
                latents = mean
        mask = sequence_mask(lengths, shape[1]).unsqueeze(-1).to(latents.dtype)
        return latents * mask


def sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if embeddings.shape[0] <= 1:
        return embeddings.new_tensor(0.0)
    embeddings = F.normalize(embeddings, dim=1)
    labels = labels.view(-1, 1)
    positive = torch.eq(labels, labels.T).float()
    self_mask = torch.eye(labels.shape[0], device=embeddings.device)
    positive = positive * (1.0 - self_mask)
    valid = positive.sum(dim=1) > 0
    if not torch.any(valid):
        return embeddings.new_tensor(0.0)
    logits = torch.matmul(embeddings, embeddings.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    loss = -(positive * log_prob).sum(dim=1) / positive.sum(dim=1).clamp_min(1.0)
    return loss[valid].mean()
