from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected

from .data_loading import sanitize_rel


PPI_TYPES = [
    "protein-protein_activation",
    "protein-protein_binding",
    "protein-protein_catalysis",
    "protein-protein_expression",
    "protein-protein_inhibition",
    "protein-protein_ptmod",
    "protein-protein_reaction",
]


def build_bipartite_graph(hetero) -> Tuple[Data, Tensor, Tensor]:
    """Flatten disease and protein nodes into one homogeneous graph.

    Returns:
        graph: homogeneous Data with edge_index only (disease+protein).
        num_diseases: tensor with single int.
        num_proteins: tensor with single int.
    """
    dp_edges = hetero.edge_index_dict[("disease", "disease-protein", "protein")]
    num_diseases = torch.tensor(hetero.num_nodes_dict["disease"])
    num_proteins = torch.tensor(hetero.num_nodes_dict["protein"])

    edge_index = dp_edges.clone()
    edge_index[1] += num_diseases

    graph = Data(edge_index=edge_index)
    graph.num_nodes = int(num_diseases + num_proteins)
    graph.node_type = torch.cat(
        [torch.zeros(num_diseases, dtype=torch.long), torch.ones(num_proteins, dtype=torch.long)]
    )
    return graph, num_diseases, num_proteins


def add_ppi_edges(graph: Data, num_diseases: int, hetero) -> Data:
    """Augment with protein-protein edges from BioKG (homogeneous view)."""
    ppi_list = []
    for et in PPI_TYPES:
        ppi = hetero.edge_index_dict[("protein", et, "protein")].clone()
        ppi += num_diseases
        ppi_list.append(ppi)
    if ppi_list:
        ppi_edges = torch.cat(ppi_list, dim=1)
        ppi_edges = to_undirected(ppi_edges)
        graph.edge_index = torch.cat([graph.edge_index, ppi_edges], dim=1)
    return graph


def build_hetero_graph(hetero, include_ppi: bool = True, add_reverse: bool = True) -> Tuple[HeteroData, Tensor, Tensor]:
    """Build a hetero graph with disease-protein plus optional PPI edges."""
    data = HeteroData()
    num_diseases = hetero.num_nodes_dict["disease"]
    num_proteins = hetero.num_nodes_dict["protein"]
    data["disease"].num_nodes = num_diseases
    data["protein"].num_nodes = num_proteins

    dp_rel = sanitize_rel("disease-protein")
    pd_rel = sanitize_rel("protein-disease")
    dp_edges = hetero.edge_index_dict[("disease", "disease-protein", "protein")]
    data[("disease", dp_rel, "protein")].edge_index = dp_edges
    if add_reverse:
        data[("protein", pd_rel, "disease")].edge_index = dp_edges.flip(0)

    if include_ppi:
        for et in PPI_TYPES:
            safe_rel = sanitize_rel(et)
            data[("protein", safe_rel, "protein")].edge_index = hetero.edge_index_dict[("protein", et, "protein")]

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
