from __future__ import annotations

from pathlib import Path
from typing import Any

from .unpublished import research_assets_unavailable


class UmamiExplainer:
                                                                              

    def __init__(self, project_root: Path, fold: int = 1) -> None:
        self.project_root = project_root
        self.fold = fold

    def explain_sequence(
        self,
        sequence: str,
        strategy: str = "ag_replacement",
    ) -> dict[str, Any]:
        research_assets_unavailable()
