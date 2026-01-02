# %%
from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import to_undirected

try:
    from src.hetero_models import build_encoder
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from hetero_models import build_encoder

# Suppress torch.load weights_only FutureWarning emitted by OGB loader
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch.load` with `weights_only=False`",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sanitize_rel(rel: str) -> str:
    """Replace characters that upset to_hetero; keeps only underscores as separators."""
    return rel.replace("-", "_")


def load_biokg() -> Tuple[Data, Tensor, Tensor]:
    """Load ogbl-biokg and return bipartite (disease, protein) graph and counts.

    Returns:
        graph: homogeneous Data with edge_index only (disease+protein).
        num_diseases: tensor with single int.
        num_proteins: tensor with single int.
    """
    dataset = PygLinkPropPredDataset(name="ogbl-biokg")
    hetero = dataset[0]

    dp_edges = hetero.edge_index_dict[("disease", "disease-protein", "protein")]
    num_diseases = torch.tensor(hetero.num_nodes_dict["disease"])
    num_proteins = torch.tensor(hetero.num_nodes_dict["protein"])

    # Shift protein node indices so diseases [0, num_diseases-1], proteins [num_diseases, ...]
    edge_index = dp_edges.clone()
    edge_index[1] += num_diseases

    graph = Data(edge_index=edge_index)
    graph.num_nodes = int(num_diseases + num_proteins)
    graph.node_type = torch.cat(
        [torch.zeros(num_diseases, dtype=torch.long), torch.ones(num_proteins, dtype=torch.long)]
    )
    return graph, num_diseases, num_proteins


def load_biokg_hetero() -> Tuple[HeteroData, Tensor, Tensor]:
    """Load ogbl-biokg and keep disease–protein plus optional protein–protein edges."""
    dataset = PygLinkPropPredDataset(name="ogbl-biokg")
    hetero = dataset[0]

    data = HeteroData()
    num_diseases = hetero.num_nodes_dict["disease"]
    num_proteins = hetero.num_nodes_dict["protein"]
    data["disease"].num_nodes = num_diseases
    data["protein"].num_nodes = num_proteins

    dp_rel = _sanitize_rel("disease-protein")
    pd_rel = _sanitize_rel("protein-disease")
    dp_edges = hetero.edge_index_dict[("disease", "disease-protein", "protein")]
    data["disease", dp_rel, "protein"].edge_index = dp_edges
    # Add reverse edges so both node types receive messages
    data["protein", pd_rel, "disease"].edge_index = dp_edges.flip(0)

    ppi_types = [
        "protein-protein_activation",
        "protein-protein_binding",
        "protein-protein_catalysis",
        "protein-protein_expression",
        "protein-protein_inhibition",
        "protein-protein_ptmod",
        "protein-protein_reaction",
    ]
    for et in ppi_types:
        safe_rel = _sanitize_rel(et)
        data["protein", safe_rel, "protein"].edge_index = hetero.edge_index_dict[("protein", et, "protein")]

    return data, torch.tensor(num_diseases), torch.tensor(num_proteins)


def summarize_hetero(data: HeteroData) -> Dict[str, Dict[str, int]]:
    """Return counts of nodes and edges per type for quick EDA."""
    node_counts = {ntype: int(data[ntype].num_nodes) for ntype in data.node_types}
    edge_counts = {
        "::".join([src, rel, dst]): data[(src, rel, dst)].edge_index.size(1)
        for src, rel, dst in data.edge_types
    }
    return {"nodes": node_counts, "edges": edge_counts}


def summarize_bipartite(graph: Data, num_diseases: int, num_proteins: int) -> Dict[str, int]:
    """Return basic counts for the flattened bipartite view."""
    return {
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.edge_index.size(1)),
        "num_diseases": int(num_diseases),
        "num_proteins": int(num_proteins),
    }


def add_ppi_edges(graph: Data, num_diseases: int, hetero) -> Data:
    """Augment with protein–protein edges from BioKG.

    The protein indices are already shifted by num_diseases.
    """
    ppi_types = [
        "protein-protein_activation",
        "protein-protein_binding",
        "protein-protein_catalysis",
        "protein-protein_expression",
        "protein-protein_inhibition",
        "protein-protein_ptmod",
        "protein-protein_reaction",
    ]
    ppi_list = []
    for et in ppi_types:
        ppi = hetero.edge_index_dict[("protein", et, "protein")].clone()
        # shift both ends by num_diseases
        ppi += num_diseases
        ppi_list.append(ppi)
    if ppi_list:
        ppi_edges = torch.cat(ppi_list, dim=1)
        # undirected to propagate both ways
        ppi_edges = to_undirected(ppi_edges)
        graph.edge_index = torch.cat([graph.edge_index, ppi_edges], dim=1)
    return graph


def random_split_edges(edge_index: Tensor, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges, generator=torch.Generator().manual_seed(seed))
    train_end = int(train_ratio * num_edges)
    val_end = int((train_ratio + val_ratio) * num_edges)
    train_edges = edge_index[:, perm[:train_end]]
    val_edges = edge_index[:, perm[train_end:val_end]]
    test_edges = edge_index[:, perm[val_end:]]
    return train_edges, val_edges, test_edges


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
    def __init__(self, metadata, num_diseases: int, num_proteins: int, hidden: int = 128, out: int = 128, model_kind: str = "sage", heads: int = 4, num_bases: int = 30):
        super().__init__()
        self.emb = torch.nn.ParameterDict(
            {
                "disease": torch.nn.Parameter(torch.randn(num_diseases, hidden)),
                "protein": torch.nn.Parameter(torch.randn(num_proteins, hidden)),
            }
        )
        if model_kind == "rgcn":
            raise ValueError("rgcn requires integer edge_type; not wired for BioKG hetero dict yet.")
        self.encoder = build_encoder(model_kind, metadata=metadata, hidden=hidden, out=out, heads=heads, num_bases=num_bases)

    def forward(self, data: HeteroData, edge_label_index: Tensor) -> Tensor:
        x_dict = {ntype: self.emb[ntype] for ntype in self.emb}
        z_dict = self.encoder(x_dict, data.edge_index_dict)
        return (z_dict["disease"][edge_label_index[0]] * z_dict["protein"][edge_label_index[1]]).sum(dim=-1)


def negative_sampling(num_nodes: int, num_samples: int, device) -> Tensor:
    return torch.randint(0, num_nodes, (2, num_samples), device=device)


def _classification_metrics(logits: Tensor, labels: Tensor) -> Dict[str, float]:
    """Compute F1 (primary), precision, recall, accuracy, and AUC on CPU."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    labels_int = labels.long()

    tp = ((preds == 1) & (labels_int == 1)).sum().item()
    fp = ((preds == 1) & (labels_int == 0)).sum().item()
    fn = ((preds == 0) & (labels_int == 1)).sum().item()
    tn = ((preds == 0) & (labels_int == 0)).sum().item()

    def _safe_div(num, den):
        return num / den if den != 0 else 0.0

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    try:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(labels_int.cpu(), probs.cpu())
    except Exception:
        auc = float("nan")

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "auc": float(auc),
    }


def train_one_epoch(model, optimizer, criterion, edge_index, pos_edges, device):
    model.train()
    optimizer.zero_grad()
    pos_out = model(edge_index, pos_edges)
    pos_loss = criterion(pos_out, torch.ones(pos_out.size(0), device=device))
    neg_edges = negative_sampling(model.embedding.num_embeddings, pos_edges.size(1), device)
    neg_out = model(edge_index, neg_edges)
    neg_loss = criterion(neg_out, torch.zeros(neg_out.size(0), device=device))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_epoch_hetero(model, optimizer, criterion, data: HeteroData, pos_edges: Tensor, device: torch.device):
    model.train()
    optimizer.zero_grad()
    pos_out = model(data, pos_edges)
    pos_loss = criterion(pos_out, torch.ones(pos_out.size(0), device=device))
    # Draw negatives within valid ranges for each node type
    neg_d = torch.randint(0, model.emb["disease"].size(0), (pos_edges.size(1),), device=device)
    neg_p = torch.randint(0, model.emb["protein"].size(0), (pos_edges.size(1),), device=device)
    neg_edges = torch.stack([neg_d, neg_p], dim=0)
    neg_out = model(data, neg_edges)
    neg_loss = criterion(neg_out, torch.zeros(neg_out.size(0), device=device))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, edge_index, pos_edges, device):
    model.eval()
    with torch.no_grad():
        pos_out = model(edge_index, pos_edges)
        neg_edges = negative_sampling(model.embedding.num_embeddings, pos_edges.size(1), device)
        neg_out = model(edge_index, neg_edges)
        scores = torch.cat([pos_out, neg_out]).cpu()
        labels = torch.cat(
            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]
        ).cpu()
    return _classification_metrics(scores, labels)


def evaluate_hetero(model, data: HeteroData, pos_edges: Tensor, device: torch.device):
    model.eval()
    with torch.no_grad():
        pos_out = model(data, pos_edges)
        neg_d = torch.randint(0, model.emb["disease"].size(0), (pos_edges.size(1),), device=device)
        neg_p = torch.randint(0, model.emb["protein"].size(0), (pos_edges.size(1),), device=device)
        neg_edges = torch.stack([neg_d, neg_p], dim=0)
        neg_out = model(data, neg_edges)
        scores = torch.cat([pos_out, neg_out]).cpu()
        labels = torch.cat(
            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]
        ).cpu()
    return _classification_metrics(scores, labels)


class EarlyStopper:
    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False
        improve = metric > self.best if self.mode == "max" else metric < self.best
        if improve:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _save_ckpt(model: torch.nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def pretty(summary: Dict) -> str:
    """Nicely format a summary dict as JSON."""
    return json.dumps(summary, indent=2)

# %%
defaults = {
    "add_ppi": True,
    "model_kind": "gcn",  # sage|gcn|gat|hgt   |rgcn
    "hidden": 128,
    "out": 128,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "epochs": 50,
    "eval_every": 1,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "seed": 42,
    "patience": 10,
    "ckpt_path": None,
    "heads": 4,
    "num_bases": 30,
}
args = SimpleNamespace(**defaults)
print(pretty(defaults))

seed_everything(args.seed)

data, num_diseases, num_proteins = load_biokg_hetero()
train_e, val_e, test_e = random_split_edges(
    data["disease", _sanitize_rel("disease-protein"), "protein"].edge_index, args.train_ratio, args.val_ratio, args.seed
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
train_e, val_e, test_e = train_e.to(device), val_e.to(device), test_e.to(device)
model = HeteroLinkPredictor(
    metadata=data.metadata(),
    num_diseases=int(num_diseases),
    num_proteins=int(num_proteins),
    hidden=args.hidden,
    out=args.out,
    model_kind=args.model_kind,
    heads=getattr(args, "heads", 4),
    num_bases=getattr(args, "num_bases", 30),
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = BCEWithLogitsLoss()
stopper = EarlyStopper(patience=args.patience, mode="max")
best_state = None
best_val = float("-inf")
ckpt_path = Path(args.ckpt_path) if args.ckpt_path else Path(f"hetero_{args.model_kind}_best_model.pth")

for epoch in range(1, args.epochs + 1):
    loss = train_one_epoch_hetero(model, optimizer, criterion, data, train_e, device)
    if epoch % args.eval_every == 0:
        val_metrics = evaluate_hetero(model, data, val_e, device)
        improved = val_metrics["f1"] > best_val
        if improved:
            best_val = val_metrics["f1"]
            best_state = model.state_dict()
            if ckpt_path:
                _save_ckpt(model, ckpt_path)
        print(
            f"Epoch {epoch:03d} | loss {loss:.4f} | val f1 {val_metrics['f1']:.4f} | "
            f"precision {val_metrics['precision']:.4f} | recall {val_metrics['recall']:.4f} | auc {val_metrics['auc']:.4f}"
        )
        if stopper.step(val_metrics["f1"]):
            print("Early stopping")
            break

if best_state is not None:
    model.load_state_dict(best_state)
test_metrics = evaluate_hetero(model, data, test_e, device)
print(
    f"Test f1 {test_metrics['f1']:.4f} | precision {test_metrics['precision']:.4f} | "
    f"recall {test_metrics['recall']:.4f} | acc {test_metrics['accuracy']:.4f} | auc {test_metrics['auc']:.4f}"
)

# %% else block for homogeneous model
defaults = {
    "add_ppi": False,
    "model_kind": "sage",  # sage|gcn|gat|hgt   |rgcn
    "hidden": 128,
    "out": 128,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "epochs": 100,
    "eval_every": 5,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "seed": 42,
    "patience": 20,
    "ckpt_path": None,
    "heads": 4,
    "num_bases": 30,
}
args = SimpleNamespace(**defaults)
print(pretty(defaults))

seed_everything(args.seed)

dataset = PygLinkPropPredDataset(name="ogbl-biokg")
hetero = dataset[0]
graph, num_diseases, num_proteins = load_biokg()
if args.add_ppi:
    graph = add_ppi_edges(graph, int(num_diseases), hetero)

train_e, val_e, test_e = random_split_edges(graph.edge_index, args.train_ratio, args.val_ratio, args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LinkPredictorSAGE(num_nodes=graph.num_nodes, hidden=args.hidden, out=args.out).to(device)
graph = graph.to(device)
train_e, val_e, test_e = train_e.to(device), val_e.to(device), test_e.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = BCEWithLogitsLoss()
stopper = EarlyStopper(patience=args.patience, mode="max")
best_state = None
best_val = float("-inf")
ckpt_path = Path(args.ckpt_path) if args.ckpt_path else Path(f"homogen_{args.model_kind}_best_model.pth")

for epoch in range(1, args.epochs + 1):
    loss = train_one_epoch(model, optimizer, criterion, graph.edge_index, train_e, device)
    if epoch % args.eval_every == 0:
        val_metrics = evaluate(model, graph.edge_index, val_e, device)
        improved = val_metrics["f1"] > best_val
        if improved:
            best_val = val_metrics["f1"]
            best_state = model.state_dict()
            if ckpt_path:
                _save_ckpt(model, ckpt_path)
        print(
            f"Epoch {epoch:03d} | loss {loss:.4f} | val f1 {val_metrics['f1']:.4f} | "
            f"precision {val_metrics['precision']:.4f} | recall {val_metrics['recall']:.4f} | auc {val_metrics['auc']:.4f}"
        )
        if stopper.step(val_metrics["f1"]):
            print("Early stopping")
            break

if best_state is not None:
    model.load_state_dict(best_state)
test_metrics = evaluate(model, graph.edge_index, test_e, device)
print(
    f"Test f1 {test_metrics['f1']:.4f} | precision {test_metrics['precision']:.4f} | "
    f"recall {test_metrics['recall']:.4f} | acc {test_metrics['accuracy']:.4f} | auc {test_metrics['auc']:.4f}"
)

# %%


data, nd, np = load_biokg_hetero()
print(pretty(summarize_hetero(data)))

graph, nd, np = load_biokg()
print(pretty(summarize_bipartite(graph, nd, np)))

# %%
import pickle
from ogb.linkproppred import PygLinkPropPredDataset

dataset = PygLinkPropPredDataset(name="ogbl-biokg")

# with open("ogbl_biokg_raw.pkl", "wb") as f:
#     pickle.dump(dataset[0], f)

# %%
