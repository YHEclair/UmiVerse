from __future__ import annotations

from dataclasses import dataclass

import torch


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<MASK>")


@dataclass(frozen=True)
class PeptideTokenizer:
    amino_acids: str = AMINO_ACIDS

    def __post_init__(self) -> None:
        tokens = list(SPECIAL_TOKENS) + list(self.amino_acids)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "token_to_id", {token: idx for idx, token in enumerate(tokens)})
        object.__setattr__(self, "id_to_token", {idx: token for idx, token in enumerate(tokens)})

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def mask_id(self) -> int:
        return self.token_to_id["<MASK>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def validate(self, sequence: str) -> None:
        invalid = sorted(set(sequence) - set(self.amino_acids))
        if invalid:
            raise ValueError(f"Sequence contains invalid residues: {invalid}")

    def encode(self, sequence: str, max_residues: int) -> list[int]:
        sequence = sequence.strip().upper()
        self.validate(sequence)
        if len(sequence) > max_residues:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds max_residues={max_residues}"
            )
        ids = [self.bos_id]
        ids.extend(self.token_to_id[residue] for residue in sequence)
        ids.append(self.eos_id)
        target_length = max_residues + 2
        ids.extend([self.pad_id] * (target_length - len(ids)))
        return ids

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        residues: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token[int(token_id)]
            if token == "<EOS>":
                break
            if token in SPECIAL_TOKENS:
                continue
            residues.append(token)
        return "".join(residues)

