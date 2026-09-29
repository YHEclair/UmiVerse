from __future__ import annotations

from pathlib import Path

import pandas as pd


VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).strip().upper().split())


def validate_sequence(sequence: str, max_length: int = 20) -> tuple[bool, str]:
    normalized = normalize_sequence(sequence)
    if not normalized:
        return False, "Sequence is empty."
    invalid = sorted(set(normalized) - VALID_AMINO_ACIDS)
    if invalid:
        return False, f"Invalid amino acid residues: {', '.join(invalid)}"
    if len(normalized) > max_length:
        return False, f"Sequence length {len(normalized)} exceeds {max_length} aa."
    return True, ""


def read_fasta(path: Path) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if current_id is not None:
                    sequence = normalize_sequence("".join(chunks))
                    rows.append(
                        {
                            "peptide_id": current_id,
                            "sequence": sequence,
                            "length": len(sequence),
                        }
                    )
                current_id = text[1:].strip() or f"seq_{len(rows) + 1}"
                chunks = []
            else:
                chunks.append(text)
    if current_id is not None:
        sequence = normalize_sequence("".join(chunks))
        rows.append({"peptide_id": current_id, "sequence": sequence, "length": len(sequence)})
    return pd.DataFrame(rows)


def read_sequence_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa", ".fas"}:
        return read_fasta(path)
    frame = pd.read_csv(path)
    if "sequence" not in frame.columns:
        candidates = [name for name in frame.columns if "seq" in name.lower() or "peptide" in name.lower()]
        if not candidates:
            raise ValueError("CSV file must contain a sequence column.")
        frame = frame.rename(columns={candidates[0]: "sequence"})
    if "peptide_id" not in frame.columns:
        frame.insert(0, "peptide_id", [f"seq_{idx + 1}" for idx in range(len(frame))])
    frame["sequence"] = frame["sequence"].map(normalize_sequence)
    frame["length"] = frame["sequence"].str.len()
    return frame


def write_fasta(frame: pd.DataFrame, path: Path, sequence_column: str = "sequence") -> None:
    id_column = "peptide_id" if "peptide_id" in frame.columns else None
    with path.open("w", encoding="utf-8") as handle:
        for idx, row in frame.reset_index(drop=True).iterrows():
            peptide_id = str(row[id_column]) if id_column else f"seq_{idx + 1}"
            sequence = normalize_sequence(str(row[sequence_column]))
            handle.write(f">{peptide_id}\n{sequence}\n")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
