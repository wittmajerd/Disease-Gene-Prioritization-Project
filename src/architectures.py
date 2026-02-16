from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.nn import GATConv, GraphConv, Node2Vec, SAGEConv, TransformerConv, to_hetero


def _build_conv(kind: str, in_ch: int, out_ch: int, heads: int = 4):
    kind = kind.lower()
    if kind == "sage":
        return SAGEConv(in_ch, out_ch)
    if kind == "gcn":
        return GraphConv(in_ch, out_ch)
    if kind == "gat":
        return GATConv(in_ch, out_ch, heads=heads, concat=False, add_self_loops=False)
    if kind in {"transformer", "graph_transformer"}:
        return TransformerConv(in_ch, out_ch, heads=heads, concat=False)
    raise ValueError(f"Unknown layer type: {kind}")


class _GNNBase(torch.nn.Module):
    def __init__(self, layer_kind: str, layer_sizes: list[int], heads: int = 4, first_in: int | None = None):
        super().__init__()
        if not layer_sizes:
            raise ValueError("layer_sizes must not be empty")

        self.convs = torch.nn.ModuleList()
        for i, out_ch in enumerate(layer_sizes):
            if i == 0:
                in_ch = first_in if first_in is not None else layer_sizes[0]
            else:
                in_ch = layer_sizes[i - 1]
            self.convs.append(_build_conv(layer_kind, in_ch, out_ch, heads=heads))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
        return x


class GNNLinkPredictor(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        layer_kind: str,
        layer_sizes: list[int],
        heads: int = 4,
        init_features: Tensor | None = None,
    ):
        super().__init__()
        if not layer_sizes:
            raise ValueError("layer_sizes must not be empty")

        if init_features is not None:
            if init_features.dim() != 2 or init_features.size(0) != num_nodes:
                raise ValueError("init_features must be [num_nodes, feature_dim]")
            if init_features.size(1) != layer_sizes[0]:
                raise ValueError("init_features dim must match layer_sizes[0]")
            self.embedding = torch.nn.Embedding.from_pretrained(init_features, freeze=False)
        else:
            self.embedding = torch.nn.Embedding(num_nodes, layer_sizes[0])

        self.encoder = _GNNBase(layer_kind=layer_kind, layer_sizes=layer_sizes, heads=heads)

    def encode(self, edge_index: Tensor) -> Tensor:
        x = self.embedding.weight
        return self.encoder(x, edge_index)

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
        layer_kind: str,
        layer_sizes: list[int],
        heads: int = 4,
        disease_features: Tensor | None = None,
        protein_features: Tensor | None = None,
    ):
        super().__init__()
        if not layer_sizes:
            raise ValueError("layer_sizes must not be empty")

        def _init_param(features: Tensor | None, num_nodes: int) -> torch.nn.Parameter:
            if features is not None:
                if features.dim() != 2 or features.size(0) != num_nodes:
                    raise ValueError("init features must be [num_nodes, feature_dim]")
                if features.size(1) != layer_sizes[0]:
                    raise ValueError("init feature dim must match layer_sizes[0]")
                return torch.nn.Parameter(features)
            return torch.nn.Parameter(torch.randn(num_nodes, layer_sizes[0]))

        self.emb = torch.nn.ParameterDict(
            {
                "disease": _init_param(disease_features, num_diseases),
                "protein": _init_param(protein_features, num_proteins),
            }
        )

        base = _GNNBase(layer_kind=layer_kind, layer_sizes=layer_sizes, heads=heads, first_in=-1)
        self.encoder = to_hetero(base, metadata=metadata, aggr="sum")

    def forward(self, data, edge_label_index: Tensor) -> Tensor:
        x_dict = {ntype: self.emb[ntype] for ntype in self.emb}
        z_dict = self.encoder(x_dict, data.edge_index_dict)
        return (z_dict["disease"][edge_label_index[0]] * z_dict["protein"][edge_label_index[1]]).sum(dim=-1)


class Node2VecFeaturizer:
    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 20,
        context_size: int = 10,
        walks_per_node: int = 10,
        num_negative_samples: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        sparse: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.p = p
        self.q = q
        self.sparse = sparse
        self.model: Node2Vec | None = None

    def fit(
        self,
        edge_index: Tensor,
        num_nodes: int | None = None,
        device: torch.device | None = None,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 0.01,
    ) -> "Node2VecFeaturizer":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge_index = edge_index.to(device)
        self.model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            num_negative_samples=self.num_negative_samples,
            p=self.p,
            q=self.q,
            sparse=self.sparse,
            num_nodes=num_nodes,
        ).to(device)

        loader = self.model.loader(batch_size=batch_size, shuffle=True, num_workers=0)
        optimizer_cls = torch.optim.SparseAdam if self.sparse else torch.optim.Adam
        optimizer = optimizer_cls(list(self.model.parameters()), lr=lr)

        for _ in range(epochs):
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

        return self

    def get_embeddings(self) -> Tensor:
        if self.model is None:
            raise RuntimeError("Call fit() before get_embeddings().")
        return self.model.embedding.weight.detach().cpu()

    def fit_transform(
        self,
        edge_index: Tensor,
        num_nodes: int | None = None,
        device: torch.device | None = None,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 0.01,
    ) -> Tensor:
        self.fit(edge_index, num_nodes=num_nodes, device=device, epochs=epochs, batch_size=batch_size, lr=lr)
        return self.get_embeddings()
