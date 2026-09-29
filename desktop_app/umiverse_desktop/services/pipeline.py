from __future__ import annotations

from pathlib import Path

import pandas as pd

from .generator import UmamiGenerator
from .io_utils import read_sequence_file, validate_sequence
from .predictors import PredictionSuite


class UmiVersePipeline:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.predictors = PredictionSuite(project_root)
        self.generator = UmamiGenerator(project_root)

    def predict_single(self, sequence: str) -> pd.DataFrame:
        valid, message = validate_sequence(sequence)
        if not valid:
            raise ValueError(message)
        return self.predictors.predict([sequence])

    def predict_file(self, path: Path, include_umami: bool, include_toxicity: bool) -> pd.DataFrame:
        frame = read_sequence_file(path)
        sequences = frame["sequence"].astype(str).tolist()
        validity = [validate_sequence(sequence) for sequence in sequences]
        valid_mask = [item[0] for item in validity]
        result = frame.copy()
        result["valid_sequence"] = valid_mask
        result["validation_message"] = [item[1] for item in validity]
        if any(valid_mask):
            predictions = self.predictors.predict(
                [sequence for sequence, valid in zip(sequences, valid_mask) if valid],
                include_umami=include_umami,
                include_toxicity=include_toxicity,
            )
            result = result.merge(predictions, on="sequence", how="left")
        return result

    def generate_candidates(
        self,
        mode: str,
        count: int,
        fixed_length: int,
        min_length: int,
        max_length: int,
        sampling_steps: int,
        temperature: float,
        top_p: float,
    ) -> pd.DataFrame:
        return self.generator.generate(
            mode=mode,
            count=count,
            fixed_length=fixed_length,
            min_length=min_length,
            max_length=max_length,
            sampling_steps=sampling_steps,
            temperature=temperature,
            top_p=top_p,
        )

    def screen_generated(
        self,
        mode: str,
        count: int,
        fixed_length: int,
        min_length: int,
        max_length: int,
        sampling_steps: int,
        temperature: float,
        top_p: float,
        min_umami: float,
        max_toxic: float,
    ) -> pd.DataFrame:
        generated = self.generate_candidates(
            mode=mode,
            count=count,
            fixed_length=fixed_length,
            min_length=min_length,
            max_length=max_length,
            sampling_steps=sampling_steps,
            temperature=temperature,
            top_p=top_p,
        )
        screened = self.predictors.screen(
            generated["sequence"].astype(str).tolist(),
            min_umami=min_umami,
            max_toxic=max_toxic,
        )
        return generated.merge(screened, on="sequence", how="left")
