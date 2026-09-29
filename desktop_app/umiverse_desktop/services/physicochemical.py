from __future__ import annotations

from collections import Counter


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


def estimate_pi(sequence: str) -> float:
    acidic = sequence.count("D") + sequence.count("E")
    basic = sequence.count("K") + sequence.count("R") + sequence.count("H")
    if basic > acidic:
        return 8.5
    if acidic > basic:
        return 4.5
    return 6.5


def physicochemical_properties(sequence: str) -> dict[str, float | int]:
    length = len(sequence)
    counts = Counter(sequence)
    molecular_weight = sum(RESIDUE_MASS.get(residue, 0.0) for residue in sequence)
    if length > 1:
        molecular_weight -= 18.015 * (length - 1)
    hydrophobicity = (
        sum(HYDROPATHY.get(residue, 0.0) for residue in sequence) / length if length else 0.0
    )
    return {
        "length": length,
        "molecular_weight": round(molecular_weight, 3),
        "isoelectric_point_est": estimate_pi(sequence),
        "hydrophobicity": round(hydrophobicity, 3),
        "hydrophobic_ratio": round(sum(counts[aa] for aa in "AILMFWVY") / max(1, length), 4),
        "acidic_ratio": round(sum(counts[aa] for aa in "DE") / max(1, length), 4),
        "basic_ratio": round(sum(counts[aa] for aa in "KRH") / max(1, length), 4),
    }
