from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

RESIDUE_MASS = {
    "A": 89.09,
    "C": 121.16,
    "D": 133.10,
    "E": 147.13,
    "F": 165.19,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "K": 146.19,
    "L": 131.17,
    "M": 149.21,
    "N": 132.12,
    "P": 115.13,
    "Q": 146.15,
    "R": 174.20,
    "S": 105.09,
    "T": 119.12,
    "V": 117.15,
    "W": 204.23,
    "Y": 181.19,
}

GROUPS = {
    "hydrophobic_ratio": set("AILMFWVY"),
    "polar_ratio": set("STNQCY"),
    "acidic_ratio": set("DE"),
    "basic_ratio": set("KRH"),
    "aromatic_ratio": set("FWY"),
    "small_ratio": set("AGSTCP"),
    "proline_ratio": set("P"),
    "glycine_ratio": set("G"),
}

CODON_COUNTS = {
    "A": 4,
    "C": 2,
    "D": 2,
    "E": 2,
    "F": 2,
    "G": 4,
    "H": 2,
    "I": 3,
    "K": 2,
    "L": 6,
    "M": 1,
    "N": 2,
    "P": 4,
    "Q": 2,
    "R": 6,
    "S": 6,
    "T": 4,
    "V": 4,
    "W": 1,
    "Y": 2,
}

CTD_GROUPS = {
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "normalized_vdw": ("GASTPDC", "NVEQIL", "MHKFRYW"),
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondary_structure": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    "solvent_accessibility": ("ALFCGIVW", "RKQEND", "MPSTHY"),
}


@dataclass
class Standardizer:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer must be fitted before transform.")
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class FeatureBuilder:
    config: dict[str, Any]
    physicochemical_names: list[str] = field(default_factory=list)
    motif_vocab: list[str] = field(default_factory=list)
    motif_scores: dict[str, float] = field(default_factory=dict)
    plm_columns: list[str] = field(default_factory=list)
    phys_standardizer: Standardizer = field(default_factory=Standardizer)
    motif_standardizer: Standardizer = field(default_factory=Standardizer)
    plm_standardizer: Standardizer = field(default_factory=Standardizer)
    plm_table: pd.DataFrame | None = None
    plm_cache: dict[str, np.ndarray] = field(default_factory=dict)
    project_root: Path | None = None
    _plm_tokenizer: Any = field(default=None, init=False, repr=False)
    _plm_model: Any = field(default=None, init=False, repr=False)
    _plm_device: Any = field(default=None, init=False, repr=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_plm_tokenizer"] = None
        state["_plm_model"] = None
        state["_plm_device"] = None
        return state

    def fit(self, df: pd.DataFrame, project_root: Path) -> "FeatureBuilder":
        self.project_root = project_root
        data_cfg = self.config["data"]
        feature_cfg = self.config["features"]
        sequences = df[data_cfg["sequence_column"]].astype(str).tolist()
        labels = df[data_cfg["label_column"]].astype(int).tolist()

        if feature_cfg.get("use_physicochemical", True):
            phys = self._physicochemical_matrix(sequences)
            self.phys_standardizer.fit(phys)

        if feature_cfg.get("use_motif", True):
            self.motif_vocab = self._build_motif_vocab(sequences, labels)
            motif = self._motif_matrix(sequences)
            self.motif_standardizer.fit(motif)

        if feature_cfg.get("use_plm", False):
            self.plm_table = maybe_load_plm_embedding_table(
                project_root,
                feature_cfg.get("plm_embedding_path", ""),
                self.config.get("outputs", {}).get("feature_dir", "outputs/features"),
                feature_cfg.get("plm_model_name", "facebook/esm2_t33_650M_UR50D"),
            )
            plm = self._plm_matrix(df)
            self.plm_standardizer.fit(plm)

        return self

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        data_cfg = self.config["data"]
        feature_cfg = self.config["features"]
        sequences = df[data_cfg["sequence_column"]].astype(str).tolist()
        out: dict[str, np.ndarray] = {}

        if feature_cfg.get("use_physicochemical", True):
            phys = self._physicochemical_matrix(sequences)
            out["physicochemical"] = self.phys_standardizer.transform(phys).astype(np.float32)

        if feature_cfg.get("use_motif", True):
            motif = self._motif_matrix(sequences)
            out["motif"] = self.motif_standardizer.transform(motif).astype(np.float32)

        if feature_cfg.get("use_plm", False):
            plm = self._plm_matrix(df)
            out["plm"] = self.plm_standardizer.transform(plm).astype(np.float32)

        return out

    def feature_dims(self) -> dict[str, int]:
        dims = {}
        if self.config["features"].get("use_physicochemical", True):
            dims["physicochemical"] = len(self.physicochemical_names)
        if self.config["features"].get("use_motif", True):
            dims["motif"] = len(self.motif_vocab)
        if self.config["features"].get("use_plm", False):
            dims["plm"] = len(self.plm_columns)
        return dims

    def _physicochemical_matrix(self, sequences: list[str]) -> np.ndarray:
        feature_cfg = self.config.get("features", {})
        rows = [physicochemical_features(sequence, feature_cfg) for sequence in sequences]
        if not self.physicochemical_names:
            self.physicochemical_names = physicochemical_feature_names(feature_cfg)
        return np.asarray(rows, dtype=np.float64)

    def _build_motif_vocab(self, sequences: list[str], labels: list[int]) -> list[str]:
        feature_cfg = self.config["features"]
        min_n = int(feature_cfg["motif_min_n"])
        max_n = int(feature_cfg["motif_max_n"])
        min_count = int(feature_cfg["motif_min_count"])
        top_k = int(feature_cfg["motif_top_k"])
        include_terminal = bool(feature_cfg.get("include_terminal_motifs", True))
        selection = str(feature_cfg.get("motif_selection", "discriminative"))

        counts: Counter[str] = Counter()
        per_label_counts = {
            int(feature_cfg.get("motif_positive_label", 1)): Counter(),
            int(feature_cfg.get("motif_negative_label", 0)): Counter(),
        }
        for sequence, label in zip(sequences, labels):
            motif_counts = Counter(extract_motifs(sequence, min_n, max_n, include_terminal))
            counts.update(motif_counts)
            if label in per_label_counts:
                per_label_counts[label].update(motif_counts)

        if selection == "frequency":
            motifs = [
                motif
                for motif, count in counts.most_common()
                if count >= min_count
            ]
            self.motif_scores = {motif: float(counts[motif]) for motif in motifs[:top_k]}
            return motifs[:top_k]

        scored = discriminative_motif_scores(
            counts,
            per_label_counts[int(feature_cfg.get("motif_positive_label", 1))],
            per_label_counts[int(feature_cfg.get("motif_negative_label", 0))],
            min_count=min_count,
            alpha=float(feature_cfg.get("motif_log_odds_alpha", 0.5)),
        )
        self.motif_scores = {motif: score for motif, score, _ in scored[:top_k]}
        return [motif for motif, _, _ in scored[:top_k]]

    def _motif_matrix(self, sequences: list[str]) -> np.ndarray:
        rows = []
        for sequence in sequences:
            counts = Counter(extract_motifs(
                sequence,
                int(self.config["features"]["motif_min_n"]),
                int(self.config["features"]["motif_max_n"]),
                bool(self.config["features"].get("include_terminal_motifs", True)),
            ))
            denom = max(1, sum(counts.values()))
            rows.append([counts[motif] / denom for motif in self.motif_vocab])
        return np.asarray(rows, dtype=np.float64)

    def _plm_matrix(self, df: pd.DataFrame) -> np.ndarray:
        data_cfg = self.config["data"]
        key_column = data_cfg["id_column"]
        fallback_key = data_cfg["sequence_column"]

        if self.plm_table is None:
            sequences = df[fallback_key].astype(str).tolist()
            matrix = self._generate_plm_embeddings(sequences)
            if not self.plm_columns:
                self.plm_columns = [f"emb_{idx}" for idx in range(matrix.shape[1])]
            return matrix

        table = self.plm_table
        if not self.plm_columns:
            self.plm_columns = [
                column for column in table.columns if column.startswith("emb_")
            ]
            if not self.plm_columns:
                raise ValueError("PLM embedding table must contain columns named emb_0, emb_1, ...")
        if key_column in table.columns:
            merged = df[[key_column]].merge(table, on=key_column, how="left")
        elif fallback_key in table.columns:
            merged = df[[fallback_key]].merge(table, on=fallback_key, how="left")
        else:
            raise ValueError(
                f"PLM embedding table must contain either {key_column!r} or {fallback_key!r}."
            )

        missing_mask = merged[self.plm_columns].isna().any(axis=1).to_numpy()
        matrix = merged[self.plm_columns].to_numpy(dtype=np.float64)
        if missing_mask.any():
            sequences = df[fallback_key].astype(str).to_numpy()[missing_mask].tolist()
            generated = self._generate_plm_embeddings(sequences)
            if generated.shape[1] != len(self.plm_columns):
                raise ValueError(
                    "Generated PLM embedding dimension does not match the saved table."
                )
            matrix[missing_mask] = generated
        return matrix

    def _generate_plm_embeddings(self, sequences: list[str]) -> np.ndarray:
        self._ensure_plm_model_loaded()
        missing = [sequence for sequence in sequences if sequence not in self.plm_cache]
        if missing:
            generated = generate_plm_embeddings(
                missing,
                self._plm_tokenizer,
                self._plm_model,
                self._plm_device,
                batch_size=int(self.config["features"].get("plm_batch_size", 16)),
                max_length=int(self.config["features"].get("plm_max_length", 128)),
                pooling=str(self.config["features"].get("plm_pooling", "mean")),
            )
            for sequence, embedding in zip(missing, generated):
                self.plm_cache[sequence] = embedding
        return np.asarray([self.plm_cache[sequence] for sequence in sequences], dtype=np.float64)

    def _ensure_plm_model_loaded(self) -> None:
        if self._plm_tokenizer is not None and self._plm_model is not None:
            return

        feature_cfg = self.config["features"]
        local_model_path = str(feature_cfg.get("plm_local_model_path", "")).strip()
        local_files_only = bool(feature_cfg.get("plm_local_files_only", False))
        if local_model_path:
            model_source = Path(local_model_path).expanduser()
            if not model_source.is_absolute() and self.project_root is not None:
                model_source = self.project_root / model_source
            if not model_source.exists():
                raise FileNotFoundError(f"Local PLM directory not found: {model_source}")
            validate_local_plm_directory(
                model_source,
                use_safetensors=bool(feature_cfg.get("plm_use_safetensors", False)),
            )
            model_name_or_path: str | Path = model_source
            local_files_only = True
        else:
            model_name_or_path = str(
                feature_cfg.get("plm_model_name", "facebook/esm2_t33_650M_UR50D")
            )

        hf_endpoint = str(feature_cfg.get("hf_endpoint", "")).strip()
        if hf_endpoint and not local_files_only:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(
            int(feature_cfg.get("hf_download_timeout", 1800))
        )
        if bool(feature_cfg.get("hf_disable_symlink_warning", True)):
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "PLM branch is enabled, but torch/transformers are not available. "
                "Install requirements.txt or set features.use_plm=false."
            ) from exc

        requested_device = str(feature_cfg.get("plm_device", "auto"))
        if requested_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(requested_device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        try:
            model = AutoModel.from_pretrained(
                model_name_or_path,
                add_pooling_layer=False,
                use_safetensors=bool(feature_cfg.get("plm_use_safetensors", False)),
                local_files_only=local_files_only,
            )
        except OSError as exc:
            raise OSError(
                "Failed to load PLM weights. The Hugging Face cache may be incomplete, "
                "or the selected mirror may not expose the requested weight format. "
                "The most reliable fix is a complete local model directory configured "
                "with plm_local_model_path and plm_local_files_only=true. "
                f"Original error: {exc}"
            ) from exc
        model.to(device)
        model.eval()

        self._plm_tokenizer = tokenizer
        self._plm_model = model
        self._plm_device = device

    def release_plm_model(self) -> None:
        self._plm_tokenizer = None
        self._plm_model = None
        self._plm_device = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def save_plm_cache(self, path: Path | None = None) -> Path | None:
        if not self.plm_cache:
            return None
        if path is None:
            if self.project_root is None:
                raise RuntimeError("project_root is not set; call fit before save_plm_cache.")
            feature_dir = self.config.get("outputs", {}).get("feature_dir", "outputs/features")
            model_name = self.config["features"].get("plm_model_name", "facebook/esm2_t33_650M_UR50D")
            path = self.project_root / feature_dir / plm_cache_filename(model_name)
        if not self.plm_columns:
            first_embedding = next(iter(self.plm_cache.values()))
            self.plm_columns = [f"emb_{idx}" for idx in range(len(first_embedding))]
        rows = []
        for sequence, embedding in sorted(self.plm_cache.items()):
            row = {"sequence": sequence}
            row.update({column: float(value) for column, value in zip(self.plm_columns, embedding)})
            rows.append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path


def physicochemical_feature_names(feature_cfg: dict[str, Any] | None = None) -> list[str]:
    feature_cfg = feature_cfg or {}
    names: list[str] = []
    if bool(feature_cfg.get("descriptor_include_basic_properties", True)):
        names.extend(
            [
                "length",
                "log_length",
                "molecular_weight",
                "gravy",
                "net_charge_per_residue",
            ]
        )
        names.extend(GROUPS.keys())
    if bool(feature_cfg.get("descriptor_include_aac", True)):
        names.extend([f"aac_{aa}" for aa in AMINO_ACIDS])
    if bool(feature_cfg.get("descriptor_include_dpc", True)):
        names.extend([f"dpc_{aa1}{aa2}" for aa1 in AMINO_ACIDS for aa2 in AMINO_ACIDS])
    if bool(feature_cfg.get("descriptor_include_dde", True)):
        names.extend([f"dde_{aa1}{aa2}" for aa1 in AMINO_ACIDS for aa2 in AMINO_ACIDS])
    if bool(feature_cfg.get("descriptor_include_ctd", True)):
        names.extend(ctd_feature_names())
    if bool(feature_cfg.get("descriptor_include_terminal", True)):
        names.extend([f"n_terminal_{aa}" for aa in AMINO_ACIDS])
        names.extend([f"c_terminal_{aa}" for aa in AMINO_ACIDS])
    if bool(feature_cfg.get("descriptor_include_opf", True)):
        positions = int(feature_cfg.get("descriptor_opf_positions", 5))
        names.extend(opf_feature_names(positions))
    return names


def physicochemical_features(
    sequence: str,
    feature_cfg: dict[str, Any] | None = None,
) -> list[float]:
    feature_cfg = feature_cfg or {}
    length = len(sequence)
    counts = Counter(sequence)
    values: list[float] = []
    if bool(feature_cfg.get("descriptor_include_basic_properties", True)):
        molecular_weight = sum(RESIDUE_MASS[aa] for aa in sequence)
        if length > 1:
            molecular_weight -= 18.015 * (length - 1)
        gravy = sum(HYDROPATHY[aa] for aa in sequence) / length
        net_charge = (
            counts["K"] + counts["R"] + 0.1 * counts["H"] - counts["D"] - counts["E"]
        ) / length
        values.extend(
            [
                float(length),
                float(np.log1p(length)),
                float(molecular_weight),
                float(gravy),
                float(net_charge),
            ]
        )
        values.extend(sum(counts[aa] for aa in group) / length for group in GROUPS.values())
    if bool(feature_cfg.get("descriptor_include_aac", True)):
        values.extend(aac_features(sequence))
    if bool(feature_cfg.get("descriptor_include_dpc", True)):
        values.extend(dpc_features(sequence))
    if bool(feature_cfg.get("descriptor_include_dde", True)):
        values.extend(dde_features(sequence))
    if bool(feature_cfg.get("descriptor_include_ctd", True)):
        values.extend(ctd_features(sequence))
    if bool(feature_cfg.get("descriptor_include_terminal", True)):
        values.extend(1.0 if sequence[0] == aa else 0.0 for aa in AMINO_ACIDS)
        values.extend(1.0 if sequence[-1] == aa else 0.0 for aa in AMINO_ACIDS)
    if bool(feature_cfg.get("descriptor_include_opf", True)):
        positions = int(feature_cfg.get("descriptor_opf_positions", 5))
        values.extend(opf_features(sequence, positions))
    return values


def aac_features(sequence: str) -> list[float]:
    counts = Counter(sequence)
    length = max(1, len(sequence))
    return [counts[aa] / length for aa in AMINO_ACIDS]


def dpc_features(sequence: str) -> list[float]:
    counts = Counter(sequence[idx : idx + 2] for idx in range(max(0, len(sequence) - 1)))
    denom = max(1, len(sequence) - 1)
    return [
        counts[f"{aa1}{aa2}"] / denom
        for aa1 in AMINO_ACIDS
        for aa2 in AMINO_ACIDS
    ]


def dde_features(sequence: str) -> list[float]:
    observed = dpc_features(sequence)
    pair_count = max(1, len(sequence) - 1)
    values: list[float] = []
    for observed_value, aa1, aa2 in zip(
        observed,
        [aa1 for aa1 in AMINO_ACIDS for _ in AMINO_ACIDS],
        [aa2 for _ in AMINO_ACIDS for aa2 in AMINO_ACIDS],
    ):
        expected = (CODON_COUNTS[aa1] / 61.0) * (CODON_COUNTS[aa2] / 61.0)
        variance = expected * (1.0 - expected) / pair_count
        values.append((observed_value - expected) / np.sqrt(variance + 1e-12))
    return values


def ctd_feature_names() -> list[str]:
    names: list[str] = []
    for property_name in CTD_GROUPS:
        names.extend([f"ctd_{property_name}_composition_g{idx}" for idx in range(1, 4)])
        names.extend(
            [
                f"ctd_{property_name}_transition_g1g2",
                f"ctd_{property_name}_transition_g1g3",
                f"ctd_{property_name}_transition_g2g3",
            ]
        )
        for group_idx in range(1, 4):
            names.extend(
                [
                    f"ctd_{property_name}_distribution_g{group_idx}_first",
                    f"ctd_{property_name}_distribution_g{group_idx}_25",
                    f"ctd_{property_name}_distribution_g{group_idx}_50",
                    f"ctd_{property_name}_distribution_g{group_idx}_75",
                    f"ctd_{property_name}_distribution_g{group_idx}_100",
                ]
            )
    return names


def ctd_features(sequence: str) -> list[float]:
    values: list[float] = []
    length = max(1, len(sequence))
    for groups in CTD_GROUPS.values():
        labels = []
        for residue in sequence:
            group_idx = next(
                idx
                for idx, residues in enumerate(groups, start=1)
                if residue in residues
            )
            labels.append(group_idx)

        for group_idx in range(1, 4):
            values.append(labels.count(group_idx) / length)

        transition_counts = {(1, 2): 0, (1, 3): 0, (2, 3): 0}
        for left, right in zip(labels, labels[1:]):
            if left == right:
                continue
            pair = tuple(sorted((left, right)))
            transition_counts[pair] += 1
        denom = max(1, len(sequence) - 1)
        values.extend(
            [
                transition_counts[(1, 2)] / denom,
                transition_counts[(1, 3)] / denom,
                transition_counts[(2, 3)] / denom,
            ]
        )

        for group_idx in range(1, 4):
            positions = [
                idx + 1
                for idx, label in enumerate(labels)
                if label == group_idx
            ]
            values.extend(distribution_positions(positions, length))
    return values


def distribution_positions(positions: list[int], sequence_length: int) -> list[float]:
    if not positions:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    count = len(positions)
    percentile_indices = [
        0,
        int(np.ceil(0.25 * count)) - 1,
        int(np.ceil(0.50 * count)) - 1,
        int(np.ceil(0.75 * count)) - 1,
        count - 1,
    ]
    return [
        positions[max(0, min(index, count - 1))] / sequence_length
        for index in percentile_indices
    ]


def opf_feature_names(positions: int) -> list[str]:
    names: list[str] = []
    for pos in range(1, positions + 1):
        names.extend([f"opf_npos_{pos}_{aa}" for aa in AMINO_ACIDS])
    for pos in range(1, positions + 1):
        names.extend([f"opf_cpos_{pos}_{aa}" for aa in AMINO_ACIDS])
    return names


def opf_features(sequence: str, positions: int) -> list[float]:
    values: list[float] = []
    for pos in range(positions):
        residue = sequence[pos] if pos < len(sequence) else ""
        values.extend(1.0 if residue == aa else 0.0 for aa in AMINO_ACIDS)
    for pos in range(positions):
        residue = sequence[-(pos + 1)] if pos < len(sequence) else ""
        values.extend(1.0 if residue == aa else 0.0 for aa in AMINO_ACIDS)
    return values


def discriminative_motif_scores(
    total_counts: Counter[str],
    positive_counts: Counter[str],
    negative_counts: Counter[str],
    min_count: int,
    alpha: float = 0.5,
) -> list[tuple[str, float, int]]:
    candidates = [
        motif
        for motif, count in total_counts.items()
        if count >= min_count
    ]
    vocab_size = max(1, len(candidates))
    positive_total = sum(positive_counts[motif] for motif in candidates)
    negative_total = sum(negative_counts[motif] for motif in candidates)
    scored: list[tuple[str, float, int]] = []
    for motif in candidates:
        pos = positive_counts[motif]
        neg = negative_counts[motif]
        pos_prob = (pos + alpha) / (positive_total + alpha * vocab_size)
        neg_prob = (neg + alpha) / (negative_total + alpha * vocab_size)
        score = abs(float(np.log(pos_prob / neg_prob)))
        scored.append((motif, score, total_counts[motif]))
    return sorted(scored, key=lambda item: (item[1], item[2], item[0]), reverse=True)


def extract_motifs(
    sequence: str,
    min_n: int,
    max_n: int,
    include_terminal: bool = True,
) -> list[str]:
    motifs: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(sequence) < n:
            continue
        motifs.extend(sequence[i : i + n] for i in range(len(sequence) - n + 1))
    if include_terminal:
        motifs.append(f"N:{sequence[0]}")
        motifs.append(f"C:{sequence[-1]}")
        for n in range(2, min(max_n, len(sequence)) + 1):
            motifs.append(f"N{n}:{sequence[:n]}")
            motifs.append(f"C{n}:{sequence[-n:]}")
    return motifs


def maybe_load_plm_embedding_table(
    project_root: Path,
    path_value: str,
    feature_dir: str = "outputs/features",
    model_name: str = "facebook/esm2_t33_650M_UR50D",
) -> pd.DataFrame | None:
    if not path_value:
        default_path = project_root / feature_dir / plm_cache_filename(model_name)
        if default_path.exists():
            return pd.read_csv(default_path)
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def plm_cache_filename(model_name: str) -> str:
    safe_name = model_name.replace("/", "__").replace(":", "_")
    return f"generated_plm_embeddings__{safe_name}.csv"


def validate_local_plm_directory(path: Path, use_safetensors: bool) -> None:
    weight_filename = "model.safetensors" if use_safetensors else "pytorch_model.bin"
    required_files = {
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        weight_filename,
    }
    missing = sorted(name for name in required_files if not (path / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Local PLM directory is incomplete: {path}. Missing files: {missing}"
        )
    weight_path = path / weight_filename
    if weight_path.stat().st_size < 100_000_000:
        raise ValueError(
            f"Local PLM weight file appears incomplete: {weight_path} "
            f"({weight_path.stat().st_size} bytes)."
        )


def load_plm_embedding_table(project_root: Path, path_value: str) -> pd.DataFrame:
    if not path_value:
        raise ValueError("features.use_plm=true requires features.plm_embedding_path.")
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"PLM embedding file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def generate_plm_embeddings(
    sequences: list[str],
    tokenizer: Any,
    model: Any,
    device: Any,
    batch_size: int,
    max_length: int,
    pooling: str,
) -> np.ndarray:
    import torch

    embeddings: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        encoded = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = encoded.pop("special_tokens_mask")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooling = pooling.strip().lower().replace("-", "_")
        if pooling == "cls":
            pooled = hidden[:, 0, :]
        elif pooling == "mean":
            residue_mask = attention_mask * (
                1.0 - special_tokens_mask.to(device).unsqueeze(-1).float()
            )
            pooled = (hidden * residue_mask).sum(dim=1) / residue_mask.sum(dim=1).clamp_min(1.0)
        else:
            raise ValueError("Unsupported PLM pooling method. Use 'mean' or 'cls'.")
        embeddings.extend(pooled.detach().cpu().numpy())
    return np.asarray(embeddings, dtype=np.float64)
