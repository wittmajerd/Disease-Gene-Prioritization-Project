from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def random_split_edges(
    edge_index: Tensor, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
) -> Tuple[Tensor, Tensor, Tensor]:
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges, generator=torch.Generator().manual_seed(seed))
    train_end = int(train_ratio * num_edges)
    val_end = int((train_ratio + val_ratio) * num_edges)
    train_edges = edge_index[:, perm[:train_end]]
    val_edges = edge_index[:, perm[train_end:val_end]]
    test_edges = edge_index[:, perm[val_end:]]
    return train_edges, val_edges, test_edges


def negative_sampling_homo(num_nodes: int, num_samples: int, device) -> Tensor:
    return torch.randint(0, num_nodes, (2, num_samples), device=device)


def negative_sampling_hetero(num_diseases: int, num_proteins: int, num_samples: int, device) -> Tensor:
    neg_d = torch.randint(0, num_diseases, (num_samples,), device=device)
    neg_p = torch.randint(0, num_proteins, (num_samples,), device=device)
    return torch.stack([neg_d, neg_p], dim=0)
