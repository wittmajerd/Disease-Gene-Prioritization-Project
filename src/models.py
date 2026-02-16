"""GNN-based link prediction models for heterogeneous graphs.

Architecture overview
─────────────────────
``HeteroLinkPredictor``  (top-level model used by Trainer)
    ├── node embeddings   (``nn.ParameterDict``, one per node type)
    ├── encoder           (``GNNEncoder`` | ``HGTEncoder`` — via ``build_encoder``)
    └── decoder           (``DotProductDecoder`` | ``MLPDecoder`` | ``DistMultDecoder`` — via ``build_decoder``)

Optional:
    ``Node2VecFeaturizer`` — trains Node2Vec on the (full) graph and
    returns per-node-type embedding tensors to initialise the learnable
    embeddings above.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GraphConv,
    HGTConv,
    Node2Vec,
    SAGEConv,
    TransformerConv,
    to_hetero,
)

from src.config import ModelConfig, Node2VecConfig


# ═══════════════════════════════════════════════════════════════════
# Convolution layer factory
# ═══════════════════════════════════════════════════════════════════


def _build_conv(kind: str, in_ch: int, out_ch: int, heads: int = 4, dropout: float = 0.0) -> nn.Module:
    """Return a single GNN convolution layer.

    Parameters
    ----------
    kind:
        One of ``sage``, ``gcn``, ``gat``, ``transformer``.
    in_ch:
        Input feature dimension (``-1`` lets PyG infer per-type).
    out_ch:
        Output feature dimension.
    heads:
        Number of attention heads (GAT / Transformer only).
    dropout:
        Dropout probability inside attention layers.
    """
    ...


# ═══════════════════════════════════════════════════════════════════
# Encoders
# ═══════════════════════════════════════════════════════════════════


class GNNEncoder(nn.Module):
    """Stack of *N* homogeneous GNN layers wrapped by ``to_hetero``.

    Works with SAGE, GCN, GAT, Transformer — anything supported by
    ``to_hetero`` auto-conversion.

    Parameters
    ----------
    layer_kind:
        Convolution type string passed to ``_build_conv``.
    num_layers:
        How many conv layers to stack.
    hidden_dim:
        Width of all hidden layers.
    output_dim:
        Width of the final layer output.
    heads:
        Attention heads (relevant for gat / transformer).
    dropout:
        Applied between layers.
    """

    def __init__(
        self,
        layer_kind: str,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        ...

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Run message passing.  After ``to_hetero`` wrapping this
        becomes ``forward(x_dict, edge_index_dict)``."""
        ...


class HGTEncoder(nn.Module):
    """Native heterogeneous encoder using ``HGTConv``.

    Unlike the ``to_hetero`` wrappers this directly consumes
    ``(x_dict, edge_index_dict)`` and therefore needs the graph
    ``metadata`` at construction time.

    Parameters
    ----------
    metadata:
        ``(node_types, edge_types)`` as returned by ``HeteroData.metadata()``.
    num_layers:
        Number of ``HGTConv`` layers.
    hidden_dim:
        Hidden dimension.
    output_dim:
        Output dimension of the last layer.
    heads:
        Number of attention heads per layer.
    """

    def __init__(
        self,
        metadata: tuple,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        heads: int = 4,
    ) -> None:
        super().__init__()
        ...

    def forward(self, x_dict: dict[str, Tensor], edge_index_dict: dict) -> dict[str, Tensor]:
        ...


def build_encoder(metadata: tuple, model_cfg: ModelConfig) -> nn.Module:
    """Factory: return the appropriate encoder (already ``to_hetero``
    wrapped if applicable).

    Parameters
    ----------
    metadata:
        From ``HeteroData.metadata()``.
    model_cfg:
        Contains ``encoder_type``, dimensions, etc.

    Returns
    -------
    An ``nn.Module`` whose forward is
    ``(x_dict, edge_index_dict) → z_dict``.
    """
    ...


# ═══════════════════════════════════════════════════════════════════
# Decoders
# ═══════════════════════════════════════════════════════════════════


class DotProductDecoder(nn.Module):
    """Score = ∑(z_src ⊙ z_dst) — simple, parameter-free."""

    def forward(
        self,
        z_src: Tensor,
        z_dst: Tensor,
        edge_label_index: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        z_src:
            ``[N_src, D]`` node embeddings of source type.
        z_dst:
            ``[N_dst, D]`` node embeddings of destination type.
        edge_label_index:
            ``[2, E]`` — row-0 indexes into z_src, row-1 into z_dst.

        Returns
        -------
        ``[E]`` scalar scores (logits).
        """
        ...


class MLPDecoder(nn.Module):
    """Score = MLP(z_src ∥ z_dst).  Learnable, more expressive."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        ...

    def forward(self, z_src: Tensor, z_dst: Tensor, edge_label_index: Tensor) -> Tensor:
        ...


class DistMultDecoder(nn.Module):
    """Score = ∑(z_src ⊙ R ⊙ z_dst) with a learnable diagonal relation matrix."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        ...

    def forward(self, z_src: Tensor, z_dst: Tensor, edge_label_index: Tensor) -> Tensor:
        ...


def build_decoder(model_cfg: ModelConfig) -> nn.Module:
    """Factory: instantiate the requested decoder."""
    ...


# ═══════════════════════════════════════════════════════════════════
# Composite model
# ═══════════════════════════════════════════════════════════════════


class HeteroLinkPredictor(nn.Module):
    """End-to-end heterogeneous link prediction model.

    Components
    ----------
    emb : ``nn.ParameterDict``
        Learnable node embeddings, one ``[N_type, hidden_dim]`` matrix
        per node type.  Optionally initialised from Node2Vec.
    encoder : ``nn.Module``
        GNN encoder (output: ``dict[str, Tensor]``).
    decoder : ``nn.Module``
        Scores candidate edges from the encoded representations.

    Parameters
    ----------
    metadata:
        ``(node_types, edge_types)`` from the ``HeteroData``.
    node_counts:
        ``{node_type: num_nodes}`` dict.
    model_cfg:
        Architecture hyper-parameters.
    init_embeddings:
        Optional ``{node_type: Tensor[N, D]}`` from Node2Vec.
    """

    def __init__(
        self,
        metadata: tuple,
        node_counts: dict[str, int],
        model_cfg: ModelConfig,
        init_embeddings: dict[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        ...

    # ------------------------------------------------------------------

    def encode(self, edge_index_dict: dict) -> dict[str, Tensor]:
        """Run the GNN encoder over the full graph.

        Parameters
        ----------
        edge_index_dict:
            Message-passing edges (typically ``splits.msg_edge_index_dict``
            during training, full ``data.edge_index_dict`` at test time).

        Returns
        -------
        ``{node_type: [N, output_dim]}`` encoded representations.
        """
        ...

    def decode(
        self,
        z_dict: dict[str, Tensor],
        edge_label_index: Tensor,
        src_type: str,
        dst_type: str,
    ) -> Tensor:
        """Score a batch of candidate edges.

        Parameters
        ----------
        z_dict:
            Output of ``encode``.
        edge_label_index:
            ``[2, E]`` indices (row-0 = src, row-1 = dst).
        src_type, dst_type:
            Node type strings to index into ``z_dict``.

        Returns
        -------
        ``[E]`` logits.
        """
        ...

    def forward(
        self,
        edge_index_dict: dict,
        edge_label_index: Tensor,
        src_type: str,
        dst_type: str,
    ) -> Tensor:
        """Convenience: encode → decode in one call."""
        z_dict = self.encode(edge_index_dict)
        return self.decode(z_dict, edge_label_index, src_type, dst_type)


# ═══════════════════════════════════════════════════════════════════
# Node2Vec featurizer
# ═══════════════════════════════════════════════════════════════════


class Node2VecFeaturizer:
    """Train Node2Vec and produce per-node-type embedding tensors.

    Two modes (controlled by ``node2vec_cfg.use_full_graph``):
    * **full-graph** — train on *every* BioKG node/edge (drug, function,
      sideeffect …), then slice out only the requested node types.
      Captures richer structural signal.
    * **subgraph** — train only on the nodes/edges present in the
      downstream ``HeteroData``.

    In both cases the returned dict maps ``{node_type: [N, D]}`` and
    can be fed directly to ``HeteroLinkPredictor(init_embeddings=…)``.
    """

    def __init__(self, node2vec_cfg: Node2VecConfig) -> None:
        self.cfg = node2vec_cfg
        self._model: Node2Vec | None = None

    def fit(
        self,
        edge_index: Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> "Node2VecFeaturizer":
        """Train Node2Vec on the provided (homogeneous) edge index.

        Parameters
        ----------
        edge_index:
            ``[2, E]`` globally re-indexed edge tensor.
        num_nodes:
            Total node count in the homogeneous graph.
        device:
            CUDA / CPU.
        """
        ...

    def get_embeddings(
        self,
        node_offsets: dict[str, tuple[int, int]],
        node_types: list[str],
    ) -> dict[str, Tensor]:
        """Slice the trained embedding table by node type.

        Parameters
        ----------
        node_offsets:
            ``{type: (start_idx, count)}`` — produced by
            ``data.get_node_offsets``.
        node_types:
            Which types to return (e.g. ``["disease", "protein"]``).

        Returns
        -------
        ``{node_type: Tensor[count, embedding_dim]}``
        """
        ...

    def fit_and_get(
        self,
        edge_index: Tensor,
        num_nodes: int,
        node_offsets: dict[str, tuple[int, int]],
        node_types: list[str],
        device: torch.device,
    ) -> dict[str, Tensor]:
        """Convenience: fit + get_embeddings in one call."""
        self.fit(edge_index, num_nodes, device)
        return self.get_embeddings(node_offsets, node_types)