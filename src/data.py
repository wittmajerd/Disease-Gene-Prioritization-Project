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
from src.utils import project_root


# ── Helpers ─────────────────────────────────────────────────────────


def sanitize_rel(rel: str) -> str:
    """Replace characters that ``to_hetero`` cannot handle (e.g. ``-``)."""
    return rel.replace("-", "_")


def _as_edge_key(edge: list[str] | tuple[str, str, str]) -> tuple[str, str, str]:
    if len(edge) != 3:
        raise ValueError(f"Edge type must have exactly 3 elements, got: {edge}")
    src, rel, dst = edge
    return str(src), str(rel), str(dst)


def _raw_edge_dict(raw: Any) -> dict[tuple[str, str, str], Tensor]:
    if not hasattr(raw, "edge_index_dict"):
        raise ValueError("Raw object does not contain edge_index_dict.")
    return raw.edge_index_dict


def _raw_num_nodes_dict(raw: Any) -> dict[str, int]:
    if not hasattr(raw, "num_nodes_dict"):
        raise ValueError("Raw object does not contain num_nodes_dict.")
    return {str(k): int(v) for k, v in raw.num_nodes_dict.items()}


def _resolve_raw_edge_key(raw: Any, edge_type: tuple[str, str, str]) -> tuple[str, str, str]:
    edge_dict = _raw_edge_dict(raw)
    src, rel, dst = edge_type

    if (src, rel, dst) in edge_dict:
        return (src, rel, dst)

    rel_sanitized = sanitize_rel(rel)
    for key in edge_dict.keys():
        if key[0] == src and key[2] == dst and sanitize_rel(str(key[1])) == rel_sanitized:
            return key

    raise KeyError(f"Missing edge type in raw data: {(src, rel, dst)}")


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
    path = Path(data_path)
    if not path.is_absolute():
        path = project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Raw BioKG pickle not found: {path}")

    with path.open("rb") as f:
        raw = pickle.load(f)

    _ = _raw_edge_dict(raw)
    _ = _raw_num_nodes_dict(raw)
    return raw


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
    node_counts = _raw_num_nodes_dict(raw)
    data = HeteroData()

    for ntype in graph_cfg.node_types:
        if ntype not in node_counts:
            raise KeyError(f"Node type '{ntype}' is missing from raw BioKG")
        data[ntype].num_nodes = int(node_counts[ntype])

    sanitized_edge_types: list[tuple[str, str, str]] = []
    for edge in graph_cfg.edge_types:
        src, rel, dst = _as_edge_key(edge)
        if src not in graph_cfg.node_types or dst not in graph_cfg.node_types:
            raise ValueError(
                f"Edge {(src, rel, dst)} references node types not listed in graph.node_types"
            )

        raw_key = _resolve_raw_edge_key(raw, (src, rel, dst))
        rel_sanitized = sanitize_rel(rel)
        edge_tensor = _raw_edge_dict(raw)[raw_key]
        data[(src, rel_sanitized, dst)].edge_index = edge_tensor
        sanitized_edge_types.append((src, rel_sanitized, dst))

    if graph_cfg.add_reverse_edges:
        for src, rel, dst in sanitized_edge_types:
            rev_key = (dst, f"rev_{rel}", src)
            if rev_key in data.edge_types:
                continue
            data[rev_key].edge_index = data[(src, rel, dst)].edge_index.flip(0)

    # Ensure every node type receives messages at least from one relation.
    # This is required for stable `to_hetero` conversion when reverse edges
    # are disabled (otherwise some node types never get updated).
    incoming_by_type = {ntype: 0 for ntype in data.node_types}
    for _, _, dst in data.edge_types:
        incoming_by_type[dst] += 1

    for ntype, incoming_count in incoming_by_type.items():
        if incoming_count > 0:
            continue
        num_nodes = int(data[ntype].num_nodes)
        idx = torch.arange(num_nodes, dtype=torch.long)
        self_key = (ntype, "self", ntype)
        if self_key not in data.edge_types:
            data[self_key].edge_index = torch.stack([idx, idx], dim=0)

    target_key = _as_edge_key(graph_cfg.target_edge)
    target_sanitized = (target_key[0], sanitize_rel(target_key[1]), target_key[2])
    if target_sanitized not in data.edge_types:
        raise ValueError(
            f"graph.target_edge {target_key} is not present after graph construction. "
            "Ensure it is included in graph.edge_types."
        )

    return data


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
    offsets = get_node_offsets(raw)
    edge_parts: list[Tensor] = []

    for (src, _, dst), edge_index in _raw_edge_dict(raw).items():
        src_offset = offsets[src][0]
        dst_offset = offsets[dst][0]

        shifted = edge_index.clone().long()
        shifted[0] += src_offset
        shifted[1] += dst_offset
        edge_parts.append(shifted)

    if edge_parts:
        edge_index = torch.cat(edge_parts, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    total_num_nodes = sum(count for _, count in offsets.values())
    return edge_index, int(total_num_nodes)


def get_node_offsets_from_hetero(
    data: HeteroData,
    node_type_order: list[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """Return ``{node_type: (start_idx, count)}`` for a constructed subgraph.

    Parameters
    ----------
    data:
        Input heterogeneous graph.
    node_type_order:
        Optional node type order. If omitted, uses ``data.node_types`` order.
    """
    order = list(node_type_order) if node_type_order is not None else list(data.node_types)
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for ntype in order:
        if ntype not in data.node_types:
            raise KeyError(f"Node type '{ntype}' missing from HeteroData")
        count = int(data[ntype].num_nodes)
        offsets[ntype] = (start, count)
        start += count
    return offsets


def build_homogeneous_from_hetero(
    data: HeteroData,
    node_type_order: list[str] | None = None,
) -> tuple[Tensor, int, dict[str, tuple[int, int]]]:
    """Flatten a ``HeteroData`` graph into a homogeneous ``edge_index``.

    Useful for Node2Vec subgraph mode.

    Parameters
    ----------
    data:
        Constructed heterogeneous graph.
    node_type_order:
        Optional ordering for global node indexing.

    Returns
    -------
    edge_index:
        ``[2, num_edges]`` shifted homogeneous edge tensor.
    total_num_nodes:
        Total nodes across included node types.
    node_offsets:
        ``{node_type: (start_idx, count)}`` used for slicing embeddings back.
    """
    offsets = get_node_offsets_from_hetero(data, node_type_order=node_type_order)
    edge_parts: list[Tensor] = []

    for (src, _, dst), edge_index in data.edge_index_dict.items():
        src_offset = offsets[src][0]
        dst_offset = offsets[dst][0]

        shifted = edge_index.clone().long()
        shifted[0] += src_offset
        shifted[1] += dst_offset
        edge_parts.append(shifted)

    if edge_parts:
        merged = torch.cat(edge_parts, dim=1)
    else:
        merged = torch.empty((2, 0), dtype=torch.long)

    total_num_nodes = sum(count for _, count in offsets.values())
    return merged, int(total_num_nodes), offsets


def get_node_offsets(raw: Any) -> dict[str, tuple[int, int]]:
    """Return ``{node_type: (start_idx, count)}`` for the full homogeneous
    indexing.  Matches the order used by :func:`build_full_homogeneous_graph`.
    """
    node_counts = _raw_num_nodes_dict(raw)
    offsets: dict[str, tuple[int, int]] = {}

    start = 0
    for node_type in sorted(node_counts.keys()):
        count = int(node_counts[node_type])
        offsets[node_type] = (start, count)
        start += count

    return offsets


# ── Graph summary (EDA) ────────────────────────────────────────────


def get_graph_summary(data: HeteroData) -> Dict[str, Any]:
    """Return a JSON-friendly summary: node counts and edge counts per type."""
    node_counts = {ntype: int(data[ntype].num_nodes) for ntype in data.node_types}
    edge_counts = {
        "::".join([src, rel, dst]): int(data[(src, rel, dst)].edge_index.size(1))
        for src, rel, dst in data.edge_types
    }
    return {
        "nodes": node_counts,
        "edges": edge_counts,
    }


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
    train_ratio = float(split_cfg.train_ratio)
    val_ratio = float(split_cfg.val_ratio)
    if train_ratio <= 0 or val_ratio < 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("Invalid split ratios. Require: train>0, val>=0, train+val<1")

    src, rel, dst = _as_edge_key(target_edge)
    rel = sanitize_rel(rel)
    target_key = (src, rel, dst)
    if target_key not in data.edge_types:
        raise KeyError(f"Target edge type not found in graph: {target_key}")

    edge_index = data[target_key].edge_index
    num_edges = int(edge_index.size(1))
    if num_edges < 3:
        raise ValueError("Target edge set is too small for train/val/test split.")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(num_edges, generator=generator)

    train_end = int(train_ratio * num_edges)
    val_end = int((train_ratio + val_ratio) * num_edges)

    if train_end <= 0 or val_end <= train_end or val_end >= num_edges:
        raise ValueError("Split produced empty train/val/test. Adjust ratios or data size.")

    train_edges = edge_index[:, perm[:train_end]].clone()
    val_edges = edge_index[:, perm[train_end:val_end]].clone()
    test_edges = edge_index[:, perm[val_end:]].clone()

    msg_edge_index_dict = {k: v.clone() for k, v in data.edge_index_dict.items()}
    msg_edge_index_dict[target_key] = train_edges

    reverse_key = (dst, f"rev_{rel}", src)
    if reverse_key in msg_edge_index_dict:
        msg_edge_index_dict[reverse_key] = train_edges.flip(0)

    return EdgeSplits(
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        msg_edge_index_dict=msg_edge_index_dict,
    )


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
    if num_samples <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    pos_src = positive_edges[0].long().cpu()
    pos_dst = positive_edges[1].long().cpu()
    pos_ids = (pos_src * int(num_dst) + pos_dst).tolist()
    pos_set = set(pos_ids)

    total_pairs = int(num_src) * int(num_dst)
    max_negatives = total_pairs - len(pos_set)
    if num_samples > max_negatives:
        raise ValueError(
            f"Requested {num_samples} negatives, but only {max_negatives} are possible."
        )

    chosen_src: list[int] = []
    chosen_dst: list[int] = []
    chosen_ids: set[int] = set()

    while len(chosen_src) < num_samples:
        remaining = num_samples - len(chosen_src)
        batch_size = max(1024, remaining * 4)

        cand_src = torch.randint(0, num_src, (batch_size,), device="cpu")
        cand_dst = torch.randint(0, num_dst, (batch_size,), device="cpu")
        cand_ids = cand_src * int(num_dst) + cand_dst

        for src_i, dst_i, id_i in zip(cand_src.tolist(), cand_dst.tolist(), cand_ids.tolist()):
            if id_i in pos_set or id_i in chosen_ids:
                continue
            chosen_src.append(src_i)
            chosen_dst.append(dst_i)
            chosen_ids.add(id_i)
            if len(chosen_src) >= num_samples:
                break

    neg_edges = torch.tensor([chosen_src, chosen_dst], dtype=torch.long, device=device)
    return neg_edges