from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GraphConv,
    HGTConv,
    RGCNConv,
    SAGEConv,
    to_hetero,
)


class _TwoLayerBase(nn.Module):
    """Two-layer wrapper to be heterogenized via to_hetero."""

    def __init__(self, conv_cls, in_ch, hid_ch, out_ch, **conv_kwargs):
        super().__init__()
        self.conv1 = conv_cls(in_ch, hid_ch, **conv_kwargs)
        self.conv2 = conv_cls(hid_ch, out_ch, **conv_kwargs)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def hetero_sage(metadata, hidden: int = 128, out: int = 128):
    # Use in_ch = -1 so PyG infers channels per type; avoids tuple issues on older PyG.
    base = _TwoLayerBase(SAGEConv, -1, hidden, out)
    return to_hetero(base, metadata=metadata, aggr="sum")


def hetero_gcn(metadata, hidden: int = 128, out: int = 128):
    """GraphConv fallback because GCNConv lacks bipartite support in this PyG version."""
    # Older GraphConv may not accept add_self_loops kwarg; rely on its default behavior.
    base = _TwoLayerBase(GraphConv, -1, hidden, out)
    return to_hetero(base, metadata=metadata, aggr="sum")


def hetero_gat(metadata, hidden: int = 128, out: int = 128, heads: int = 4):
    # Use concat=False to keep feature dims consistent across layers; disable self-loops.
    base = _TwoLayerBase(GATConv, -1, hidden, out, heads=heads, concat=False, add_self_loops=False)
    return to_hetero(base, metadata=metadata, aggr="sum")


def hetero_rgcn(num_relations: int, hidden: int = 128, out: int = 128, num_bases: int = 30):
    # RGCNConv expects relation type ids; use relational data with edge_type passed at call time.
    class RGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = RGCNConv((-1, -1), hidden, num_relations=num_relations, num_bases=num_bases)
            self.conv2 = RGCNConv(hidden, out, num_relations=num_relations, num_bases=num_bases)

        def forward(self, x, edge_index, edge_type):
            x = self.conv1(x, edge_index, edge_type).relu()
            x = self.conv2(x, edge_index, edge_type)
            return x

    return RGCN()


def hetero_hgt(metadata, hidden: int = 128, out: int = 128, heads: int = 4):
    class HGTWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = HGTConv(hidden, hidden, metadata=metadata, heads=heads)
            self.conv2 = HGTConv(hidden, out, metadata=metadata, heads=heads)

        def forward(self, x_dict, edge_index_dict):
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {k: v.relu() for k, v in x_dict.items()}
            x_dict = self.conv2(x_dict, edge_index_dict)
            return x_dict

    return HGTWrapper()


def build_encoder(kind: str, metadata=None, num_relations: int = None, hidden: int = 128, out: int = 128, heads: int = 4, num_bases: int = 30):
    kind = kind.lower()
    if kind == "sage":
        return hetero_sage(metadata, hidden, out)
    if kind == "gcn":
        return hetero_gcn(metadata, hidden, out)
    if kind == "gat":
        return hetero_gat(metadata, hidden, out, heads=heads)
    if kind == "rgcn":
        if num_relations is None:
            raise ValueError("num_relations is required for rgcn")
        return hetero_rgcn(num_relations, hidden, out, num_bases=num_bases)
    if kind == "hgt":
        return hetero_hgt(metadata, hidden, out, heads=heads)
    raise ValueError(f"Unknown encoder kind: {kind}")
