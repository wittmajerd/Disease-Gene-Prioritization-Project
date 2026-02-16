from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected

from src.data_loading import sanitize_rel


PPI_TYPES = [
    "protein-protein_activation",
    "protein-protein_binding",
    "protein-protein_catalysis",
    "protein-protein_expression",
    "protein-protein_inhibition",
    "protein-protein_ptmod",
    "protein-protein_reaction",
]

# there are a lot of types of edges in the original hetero graph, 
# what should we focus on? we can king of use all of specific nodes and all edes of that type 
# but I want to conpare how much these additional edges or even node add to the disease-gene prioritization task. 
# so this should be configurable before run easily.
(disease, disease-protein, protein)=[73547, 1],
    (drug, drug-disease, disease)=[5147, 1],
    (drug, drug-drug_acquired_metabolic_disease, drug)=[63430, 1],
    (drug, drug-drug_bacterial_infectious_disease, drug)=[18554, 1],
    (drug, drug-drug_benign_neoplasm, drug)=[30348, 1],
    (drug, drug-drug_cancer, drug)=[48514, 1],
    (drug, drug-drug_cardiovascular_system_disease, drug)=[94842, 1],
    (drug, drug-drug_chromosomal_disease, drug)=[316, 1],
    (drug, drug-drug_cognitive_disorder, drug)=[34660, 1],
    (drug, drug-drug_cryptorchidism, drug)=[128, 1],
    (drug, drug-drug_developmental_disorder_of_mental_health, drug)=[14314, 1],
    (drug, drug-drug_endocrine_system_disease, drug)=[55994, 1],
    (drug, drug-drug_fungal_infectious_disease, drug)=[36114, 1],
    (drug, drug-drug_gastrointestinal_system_disease, drug)=[83210, 1],
    (drug, drug-drug_hematopoietic_system_disease, drug)=[79202, 1],
    (drug, drug-drug_hematopoietic_system_diseases, drug)=[3006, 1],
    (drug, drug-drug_hypospadias, drug)=[292, 1],
    (drug, drug-drug_immune_system_disease, drug)=[34242, 1],
    (drug, drug-drug_inherited_metabolic_disorder, drug)=[36492, 1],
    (drug, drug-drug_integumentary_system_disease, drug)=[73902, 1],
    (drug, drug-drug_irritable_bowel_syndrome, drug)=[8528, 1],
    (drug, drug-drug_monogenic_disease, drug)=[600, 1],
    (drug, drug-drug_musculoskeletal_system_disease, drug)=[57926, 1],
    (drug, drug-drug_nervous_system_disease, drug)=[80208, 1],
    (drug, drug-drug_orofacial_cleft, drug)=[380, 1],
    (drug, drug-drug_parasitic_infectious_disease, drug)=[1680, 1],
    (drug, drug-drug_personality_disorder, drug)=[972, 1],
    (drug, drug-drug_polycystic_ovary_syndrome, drug)=[514, 1],
    (drug, drug-drug_pre-malignant_neoplasm, drug)=[3224, 1],
    (drug, drug-drug_psoriatic_arthritis, drug)=[2014, 1],
    (drug, drug-drug_reproductive_system_disease, drug)=[17006, 1],
    (drug, drug-drug_respiratory_system_disease, drug)=[82168, 1],
    (drug, drug-drug_sexual_disorder, drug)=[1260, 1],
    (drug, drug-drug_sleep_disorder, drug)=[25860, 1],
    (drug, drug-drug_somatoform_disorder, drug)=[2214, 1],
    (drug, drug-drug_struct_sim, drug)=[26348, 1],
    (drug, drug-drug_substance-related_disorder, drug)=[4392, 1],
    (drug, drug-drug_thoracic_disease, drug)=[4660, 1],
    (drug, drug-drug_urinary_system_disease, drug)=[67326, 1],
    (drug, drug-drug_viral_infectious_disease, drug)=[38846, 1],
    (drug, drug-protein, protein)=[117930, 1],
    (drug, drug-sideeffect, sideeffect)=[157479, 1],
    (function, function-function, function)=[1433230, 1],
    (protein, protein-function, function)=[777577, 1],
    (protein, protein-protein_activation, protein)=[73044, 1],
    (protein, protein-protein_binding, protein)=[292254, 1],
    (protein, protein-protein_catalysis, protein)=[303434, 1],
    (protein, protein-protein_expression, protein)=[1952, 1],
    (protein, protein-protein_inhibition, protein)=[25732, 1],
    (protein, protein-protein_ptmod, protein)=[15120, 1],
    (protein, protein-protein_reaction, protein)=[352546, 1],


@dataclass
class GraphArtifacts:
    graph_data: Data | HeteroData
    edge_index: Tensor
    homo_edge_index: Tensor
    num_diseases: int
    num_proteins: int
    summary: Dict[str, Any]


@dataclass
class EdgeSplits:
    train_edges: Tensor
    val_edges: Tensor
    test_edges: Tensor


def build_bipartite_graph(hetero) -> tuple[Data, int, int]:
    dp_edges = hetero.edge_index_dict[("disease", "disease-protein", "protein")]
    num_diseases = int(hetero.num_nodes_dict["disease"])
    num_proteins = int(hetero.num_nodes_dict["protein"])

    edge_index = dp_edges.clone()
    edge_index[1] += num_diseases

    graph = Data(edge_index=edge_index)
    graph.num_nodes = num_diseases + num_proteins
    graph.node_type = torch.cat(
        [torch.zeros(num_diseases, dtype=torch.long), torch.ones(num_proteins, dtype=torch.long)]
    )
    return graph, num_diseases, num_proteins


def add_ppi_edges(graph: Data, num_diseases: int, hetero) -> Data:
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


def build_hetero_graph(hetero, include_ppi: bool = True, add_reverse: bool = True) -> tuple[HeteroData, int, int]:
    data = HeteroData()
    num_diseases = int(hetero.num_nodes_dict["disease"])
    num_proteins = int(hetero.num_nodes_dict["protein"])
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

    return data, num_diseases, num_proteins


def summarize_hetero(data: HeteroData) -> Dict[str, Dict[str, int]]:
    node_counts = {ntype: int(data[ntype].num_nodes) for ntype in data.node_types}
    edge_counts = {
        "::".join([src, rel, dst]): data[(src, rel, dst)].edge_index.size(1)
        for src, rel, dst in data.edge_types
    }
    return {"nodes": node_counts, "edges": edge_counts}


def summarize_bipartite(graph: Data, num_diseases: int, num_proteins: int) -> Dict[str, int]:
    return {
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.edge_index.size(1)),
        "num_diseases": int(num_diseases),
        "num_proteins": int(num_proteins),
    }


def random_split_edges(
    edge_index: Tensor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> EdgeSplits:
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges, generator=torch.Generator().manual_seed(seed))
    train_end = int(train_ratio * num_edges)
    val_end = int((train_ratio + val_ratio) * num_edges)

    return EdgeSplits(
        train_edges=edge_index[:, perm[:train_end]],
        val_edges=edge_index[:, perm[train_end:val_end]],
        test_edges=edge_index[:, perm[val_end:]],
    )


def negative_sampling_homo(num_nodes: int, num_samples: int, device) -> Tensor:
    return torch.randint(0, num_nodes, (2, num_samples), device=device)


def negative_sampling_hetero(num_diseases: int, num_proteins: int, num_samples: int, device) -> Tensor:
    neg_d = torch.randint(0, num_diseases, (num_samples,), device=device)
    neg_p = torch.randint(0, num_proteins, (num_samples,), device=device)
    return torch.stack([neg_d, neg_p], dim=0)


def prepare_graph_artifacts(raw_data, use_hetero: bool, add_ppi: bool) -> GraphArtifacts:
    homo_graph, num_diseases, num_proteins = build_bipartite_graph(raw_data)
    if add_ppi:
        homo_graph = add_ppi_edges(homo_graph, num_diseases, raw_data)

    if use_hetero:
        hetero_data, _, _ = build_hetero_graph(raw_data, include_ppi=add_ppi)
        return GraphArtifacts(
            graph_data=hetero_data,
            edge_index=hetero_data[("disease", sanitize_rel("disease-protein"), "protein")].edge_index,
            homo_edge_index=homo_graph.edge_index,
            num_diseases=num_diseases,
            num_proteins=num_proteins,
            summary=summarize_hetero(hetero_data),
        )

    return GraphArtifacts(
        graph_data=homo_graph,
        edge_index=homo_graph.edge_index,
        homo_edge_index=homo_graph.edge_index,
        num_diseases=num_diseases,
        num_proteins=num_proteins,
        summary=summarize_bipartite(homo_graph, num_diseases, num_proteins),
    )
