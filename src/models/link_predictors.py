from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.nn import SAGEConv

from ..hetero_models import build_encoder


class LinkPredictorSAGE(torch.nn.Module):
    def __init__(self, num_nodes: int, hidden: int = 128, out: int = 128):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, out)

    def encode(self, edge_index: Tensor) -> Tensor:
        x = self.embedding.weight
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(self, edge_index: Tensor, edge_label_index: Tensor) -> Tensor:
        z = self.encode(edge_index)
        return self.decode(z, edge_label_index)


class HeteroLinkPredictor(torch.nn.Module):
    def __init__(
        self,
        metadata,
        num_diseases: int,
        num_proteins: int,
        hidden: int = 128,
        out: int = 128,
        model_kind: str = "sage",
        heads: int = 4,
        num_bases: int = 30,
    ):
        super().__init__()
        self.emb = torch.nn.ParameterDict(
            {
                "disease": torch.nn.Parameter(torch.randn(num_diseases, hidden)),
                "protein": torch.nn.Parameter(torch.randn(num_proteins, hidden)),
            }
        )
        if model_kind == "rgcn":
            raise ValueError("rgcn requires integer edge_type; not wired for BioKG hetero dict yet.")
        self.encoder = build_encoder(
            model_kind, metadata=metadata, hidden=hidden, out=out, heads=heads, num_bases=num_bases
        )

    def forward(self, data, edge_label_index: Tensor) -> Tensor:
        x_dict = {ntype: self.emb[ntype] for ntype in self.emb}
        z_dict = self.encoder(x_dict, data.edge_index_dict)
        return (z_dict["disease"][edge_label_index[0]] * z_dict["protein"][edge_label_index[1]]).sum(
            dim=-1
        )
