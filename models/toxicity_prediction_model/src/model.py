from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class BranchEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(nn.Module):
                                                                        

    def __init__(self, branch_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(branch_count * hidden_dim, branch_count)

    def forward(self, branch_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(branch_outputs, dim=1)
        gate_input = torch.cat(branch_outputs, dim=1)
        weights = torch.softmax(self.gate(gate_input), dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: int, dropout: float) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.adapter(x))


class MotifGuidedResidualFusion(nn.Module):
    def __init__(self, hidden_dim: int, motif_alpha: float = 1.0) -> None:
        super().__init__()
        self.motif_alpha = float(motif_alpha)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        plm_output: torch.Tensor,
        motif_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(torch.cat([plm_output, motif_output], dim=1))
        fused = self.norm(plm_output + self.motif_alpha * gate * motif_output)
        gate_summary = torch.stack(
            [
                1.0 - gate.mean(dim=1),
                gate.mean(dim=1),
            ],
            dim=1,
        )
        return fused, gate_summary


class FeatureMasking(nn.Module):
    def __init__(self, mask_prob: float) -> None:
        super().__init__()
        if mask_prob < 0.0 or mask_prob >= 1.0:
            raise ValueError("Feature masking probability must be in [0, 1).")
        self.mask_prob = float(mask_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.mask_prob
        mask = torch.empty_like(x).bernoulli_(keep_prob)
        return x * mask / keep_prob


class PeptideToxicityClassifier(nn.Module):
    def __init__(self, feature_dims: dict[str, int], config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        hidden_dim = int(model_cfg["branch_hidden_dim"])
        branch_dropout = float(model_cfg["branch_dropout"])
        latent_dim = int(model_cfg["latent_dim"])
        classifier_hidden_dim = int(model_cfg["classifier_hidden_dim"])
        classifier_dropout = float(model_cfg["classifier_dropout"])
        self.use_esm_residual_adapter = bool(
            model_cfg.get("esm_residual_adapter", False)
        )
        self.use_motif_guided_fusion = bool(
            model_cfg.get("motif_guided_residual_fusion", False)
        )
        self.multi_sample_dropout = bool(model_cfg.get("multi_sample_dropout", False))
        self.dropout_samples = int(model_cfg.get("dropout_samples", 4))
        self.motif_fusion_alpha = float(model_cfg.get("motif_fusion_alpha", 1.0))
        self.feature_mask_probs = {
            "plm": float(model_cfg.get("esm_feature_mask_prob", 0.0)),
            "motif": float(model_cfg.get("motif_feature_mask_prob", 0.0)),
            "physicochemical": float(
                model_cfg.get("physicochemical_feature_mask_prob", 0.0)
            ),
        }

        active_dims = {name: dim for name, dim in feature_dims.items() if dim > 0}
        if not active_dims:
            raise ValueError("At least one feature branch must be active.")

        self.branch_names = list(active_dims.keys())
        self.encoders = nn.ModuleDict(
            {
                name: BranchEncoder(dim, hidden_dim, branch_dropout)
                for name, dim in active_dims.items()
            }
        )
        self.feature_maskers = nn.ModuleDict(
            {
                name: FeatureMasking(self.feature_mask_probs.get(name, 0.0))
                for name in active_dims
            }
        )
        self.esm_adapter = (
            ResidualAdapter(
                hidden_dim,
                int(model_cfg.get("esm_adapter_hidden_dim", hidden_dim)),
                branch_dropout,
            )
            if self.use_esm_residual_adapter and "plm" in active_dims
            else None
        )
        self.motif_fusion = (
            MotifGuidedResidualFusion(hidden_dim, self.motif_fusion_alpha)
            if self.use_motif_guided_fusion
            and "plm" in active_dims
            and "motif" in active_dims
            else None
        )
        self.fusion = GatedFusion(len(self.branch_names), hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, latent_dim),
        )
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded: dict[str, torch.Tensor] = {}
        for name in self.branch_names:
            if name not in features:
                raise KeyError(f"Missing feature branch in batch: {name}")
            masked_features = self.feature_maskers[name](features[name])
            encoded[name] = self.encoders[name](masked_features)

        if self.esm_adapter is not None and "plm" in encoded:
            encoded["plm"] = self.esm_adapter(encoded["plm"])

        if self.motif_fusion is not None and "plm" in encoded and "motif" in encoded:
            fused, gate_weights = self.motif_fusion(encoded["plm"], encoded["motif"])
        else:
            branch_outputs = [encoded[name] for name in self.branch_names]
            fused, gate_weights = self.fusion(branch_outputs)
        latent = self.projector(fused)
        if self.multi_sample_dropout and self.training and self.dropout_samples > 1:
            logits = [
                self.classifier(self.classifier_dropout(latent)).squeeze(-1)
                for _ in range(self.dropout_samples)
            ]
            logit = torch.stack(logits, dim=0).mean(dim=0)
        else:
            logit = self.classifier(latent).squeeze(-1)
        probability = torch.sigmoid(logit)
        return {
            "logit": logit,
            "probability": probability,
            "latent": latent,
            "gate_weights": gate_weights,
        }


def supervised_contrastive_loss(
    latents: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    motif_features: torch.Tensor | None = None,
    motif_positive_weight: float = 1.0,
    motif_hard_negative_weight: float = 0.5,
) -> torch.Tensor:
                                                                            

                                                                            
                                                                              
                                                                               
                                                                                
       

    if latents.shape[0] <= 1:
        return latents.new_tensor(0.0)

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float()
    self_mask = torch.eye(labels.shape[0], device=latents.device)
    positive_mask = positive_mask * (1.0 - self_mask)
    negative_mask = (1.0 - torch.eq(labels, labels.T).float()) * (1.0 - self_mask)

    positive_counts = positive_mask.sum(dim=1)
    valid_rows = positive_counts > 0
    if not torch.any(valid_rows):
        return latents.new_tensor(0.0)

    latents = F.normalize(latents, dim=1)
    logits = torch.matmul(latents, latents.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = 1.0 - self_mask

    positive_weights = positive_mask
    denominator_weights = logits_mask
    if motif_features is not None and motif_features.shape[0] == latents.shape[0]:
        motif_features = F.normalize(motif_features.float(), dim=1)
        motif_similarity = torch.matmul(motif_features, motif_features.T)
        motif_similarity = ((motif_similarity + 1.0) / 2.0).clamp(0.0, 1.0)
        motif_similarity = motif_similarity * logits_mask
        positive_weights = positive_mask * (
            1.0 + float(motif_positive_weight) * motif_similarity
        )
        denominator_weights = logits_mask * (
            1.0 + float(motif_hard_negative_weight) * motif_similarity * negative_mask
        )

    positive_weight_sums = positive_weights.sum(dim=1)
    valid_rows = positive_weight_sums > 0
    if not torch.any(valid_rows):
        return latents.new_tensor(0.0)

    exp_logits = torch.exp(logits) * denominator_weights
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (
        positive_weights * log_prob
    ).sum(dim=1) / positive_weight_sums.clamp_min(1e-12)
    return -mean_log_prob_pos[valid_rows].mean()


def pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.0,
    max_pairs: int | None = None,
) -> torch.Tensor:
                                                                       

    positive_scores = logits[labels > 0.5]
    negative_scores = logits[labels <= 0.5]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return logits.new_tensor(0.0)

    differences = positive_scores[:, None] - negative_scores[None, :] - float(margin)
    flat = differences.reshape(-1)
    if max_pairs is not None and max_pairs > 0 and flat.numel() > max_pairs:
        indices = torch.linspace(
            0,
            flat.numel() - 1,
            steps=max_pairs,
            device=flat.device,
        ).long()
        flat = flat.index_select(0, indices)
    return F.softplus(-flat).mean()


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    labels = batch["label"].to(device)
    features = {
        key: value.to(device)
        for key, value in batch.items()
        if torch.is_tensor(value) and key != "label"
    }
    return features, labels
