from __future__ import annotations

from pathlib import Path
from typing import Any


def sanitize_rel(rel: str) -> str:
    """Replace characters that upset to_hetero; keeps only underscores as separators."""
    return rel.replace("-", "_")


def load_biokg_pickle(root: Path | None = None) -> Any:
    """Load ogbl-biokg raw dataset from a pickle file.

    Expected file name: ogbl_biokg_raw.pkl at project root.
    """
    project_root = root or Path(__file__).resolve().parents[1]
    path = project_root / "ogbl_biokg_raw.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Create it with your preprocessing or place the file at the project root."
        )
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)
