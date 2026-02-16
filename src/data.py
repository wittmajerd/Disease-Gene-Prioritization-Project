"""Data loading, heterogeneous graph construction, splitting and negative sampling.

Responsibilities
────────────────
1. Load raw BioKG pickle.
2. Build a ``HeteroData`` driven purely by ``GraphConfig`` (node/edge types).
3. Split the *target* edge into train / val / test with proper message-passing
   edge separation (no information leakage).
4. Provide a negative sampler that filters out true positives.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.config import GraphConfig, SplitConfig


# ── Helpers ─────────────────────────────────────────────────────────


def sanitize_rel(rel: str) -> str:
    """Replace characters that ``to_hetero`` cannot handle (e.g. ``-``)."""
    return rel.replace("-", "_")


# ── Raw data loading ────────────────────────────────────────────────


def load_raw_data(data_path: str | Path) -> Any:
    """Load the raw BioKG object from a pickle file.

    Parameters
    ----------
    data_path:
        Absolute *or* project-root-relative path to the ``.pkl`` file.

    Returns
    -------
    The unpickled object (expected: an OGB-style HeteroData with
    ``edge_index_dict`` and ``num_nodes_dict``).
    """
    ...


# ── Heterogeneous graph construction ───────────────────────────────


def build_hetero_graph(raw: Any, graph_cfg: GraphConfig) -> HeteroData:
    """Construct a ``HeteroData`` object containing **only** the node/edge
    types listed in *graph_cfg*.

    Steps
    -----
    1. Set ``num_nodes`` for every requested node type.
    2. Copy the ``edge_index`` for every requested edge type (sanitized
       relation names so ``to_hetero`` works).
    3. Optionally add reverse edges for each edge type.

    Parameters
    ----------
    raw:
        The raw BioKG object returned by :func:`load_raw_data`.
    graph_cfg:
        Describes which node/edge types to include.

    Returns
    -------
    A ``HeteroData`` ready for GNN training.
    """
    ...


def build_full_homogeneous_graph(raw: Any) -> tuple[Tensor, int]:
    """Flatten the *entire* BioKG into a single homogeneous graph.

    Used exclusively by ``Node2VecFeaturizer`` when
    ``node2vec.use_full_graph = True``.

    Returns
    -------
    edge_index:
        ``[2, num_edges]`` tensor with globally re-indexed nodes.
    total_num_nodes:
        Sum of all node type counts.

    Notes
    -----
    Node ordering: the types are sorted alphabetically and concatenated.
    The offset map is needed later to slice embeddings back per type.
    """
    ...


def get_node_offsets(raw: Any) -> dict[str, tuple[int, int]]:
    """Return ``{node_type: (start_idx, count)}`` for the full homogeneous
    indexing.  Matches the order used by :func:`build_full_homogeneous_graph`.
    """
    ...


# ── Graph summary (EDA) ────────────────────────────────────────────


def get_graph_summary(data: HeteroData) -> Dict[str, Any]:
    """Return a JSON-friendly summary: node counts and edge counts per type."""
    ...


# ── Edge splitting ──────────────────────────────────────────────────


@dataclass
class EdgeSplits:
    """Result of splitting the target edge type.

    Attributes
    ----------
    train_edges:
        Positive supervision edges for training.
    val_edges:
        Positive edges held out for validation.
    test_edges:
        Positive edges held out for final evaluation.
    msg_edge_index_dict:
        A *copy* of all ``edge_index_dict`` entries where the target edge
        type is replaced by ``train_edges`` only.  This is what the GNN
        uses for message passing during training — prevents leaking
        val/test supervision edges into the aggregation.
    """

    train_edges: Tensor
    val_edges: Tensor
    test_edges: Tensor
    msg_edge_index_dict: dict[tuple[str, str, str], Tensor]


def split_target_edges(
    data: HeteroData,
    target_edge: list[str] | tuple[str, str, str],
    split_cfg: SplitConfig,
    seed: int = 42,
) -> EdgeSplits:
    """Split the target edge type into train / val / test.

    The message-passing ``edge_index_dict`` is adjusted so that the GNN
    only ever aggregates over *training* supervision edges (plus all
    non-target auxiliary edges).

    Parameters
    ----------
    data:
        The full ``HeteroData``.
    target_edge:
        ``[src_type, relation, dst_type]`` identifying the target.
    split_cfg:
        Contains ``train_ratio`` and ``val_ratio``.
    seed:
        For reproducibility.

    Returns
    -------
    An ``EdgeSplits`` instance.
    """
    ...


# ── Negative sampling ──────────────────────────────────────────────


def negative_sampling(
    num_src: int,
    num_dst: int,
    num_samples: int,
    positive_edges: Tensor,
    device: torch.device,
) -> Tensor:
    """Sample negative edges that are **not** in the positive set.

    Uses rejection sampling: draw random (src, dst) pairs and discard
    any that appear in ``positive_edges``.

    Parameters
    ----------
    num_src:
        Number of source-type nodes (e.g. diseases).
    num_dst:
        Number of destination-type nodes (e.g. proteins).
    num_samples:
        How many negatives to return.
    positive_edges:
        ``[2, E]`` tensor of known positive edges (used for filtering).
    device:
        Target device.

    Returns
    -------
    ``[2, num_samples]`` tensor of negative edges.
    """
    ...