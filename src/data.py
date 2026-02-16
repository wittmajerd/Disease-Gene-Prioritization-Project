from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle


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
    
    with path.open("rb") as f:
        return pickle.load(f)


# Betöltés
def load_raw_data(path) -> Any                          # pickle load

# Gráf építés — a config-ban lévő node/edge típusok alapján
def build_hetero_graph(raw, graph_cfg) -> HeteroData     # KULCS: csak azt veszi be ami a config-ban van
def get_graph_summary(data) -> dict                      # EDA

# Split — a target edge-re, message-passing aware
@dataclass EdgeSplits: train, val, test, msg_passing_edge_index_dict
def split_target_edges(data, target_edge, split_cfg) -> EdgeSplits

# Negative sampling — kiszűri a valódi éleket
def negative_sampling(num_src, num_dst, num_samples, positive_edges, device) -> Tensor