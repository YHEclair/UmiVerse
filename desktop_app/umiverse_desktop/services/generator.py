from __future__ import annotations

from pathlib import Path

import pandas as pd

from .unpublished import research_assets_unavailable


class UmamiGenerator:
                                                                              

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def generate(
        self,
        mode: str,
        count: int,
        fixed_length: int = 8,
        min_length: int = 2,
        max_length: int = 20,
        sampling_steps: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> pd.DataFrame:
        research_assets_unavailable()
