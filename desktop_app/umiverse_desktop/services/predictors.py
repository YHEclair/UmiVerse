from __future__ import annotations

from pathlib import Path

import pandas as pd

from .unpublished import research_assets_unavailable


class PredictionSuite:
                                                                              

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def predict(
        self,
        sequences: list[str],
        include_umami: bool = True,
        include_toxicity: bool = True,
    ) -> pd.DataFrame:
        research_assets_unavailable()

    def screen(
        self,
        sequences: list[str],
        min_umami: float,
        max_toxic: float,
    ) -> pd.DataFrame:
        research_assets_unavailable()
