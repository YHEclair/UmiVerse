from __future__ import annotations

from typing import NoReturn


class ResearchAssetsUnavailableError(RuntimeError):
    pass                                                         


def research_assets_unavailable() -> NoReturn:
    raise ResearchAssetsUnavailableError(
        "Model-dependent functionality is not included in this pre-publication "
        "source preview. Versioned research assets and reproduction instructions "
        "will be released with the associated manuscript."
    )
