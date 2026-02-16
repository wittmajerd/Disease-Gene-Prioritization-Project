from __future__ import annotations

from pathlib import Path

import torch

from src.architectures import GNNLinkPredictor, HeteroLinkPredictor, Node2VecFeaturizer
from src.preprocessing import GraphArtifacts


def resolve_layers(cfg) -> tuple[str, list[int]]:
    if cfg.layer_sizes:
        layer_sizes = list(cfg.layer_sizes)
    else:
        if cfg.layer_count < 1:
            raise ValueError("layer_count must be >= 1")
        layer_sizes = [cfg.layer_size] * max(cfg.layer_count - 1, 0) + [cfg.output_size]

    return cfg.model_kind, layer_sizes


def build_node2vec_features(cfg, artifacts: GraphArtifacts, layer_sizes: list[int], device, logger=None):
    if not cfg.use_node2vec:
        return None, cfg.node2vec_dim

    node2vec_dim = cfg.node2vec_dim
    if node2vec_dim != layer_sizes[0]:
        if logger is not None:
            logger.info(
                "node2vec_dim (%s) does not match layer_sizes[0] (%s). Using %s.",
                node2vec_dim,
                layer_sizes[0],
                layer_sizes[0],
            )
        node2vec_dim = layer_sizes[0]

    featurizer = Node2VecFeaturizer(embedding_dim=node2vec_dim)
    features = featurizer.fit_transform(
        artifacts.homo_edge_index,
        num_nodes=artifacts.num_diseases + artifacts.num_proteins,
        device=device,
        epochs=cfg.node2vec_epochs,
        lr=cfg.node2vec_lr,
    )
    return features, node2vec_dim


def build_model(
    cfg,
    artifacts: GraphArtifacts,
    layer_kind: str,
    layer_sizes: list[int],
    node2vec_features,
    device,
):
    if cfg.use_hetero:
        disease_features = None
        protein_features = None
        if node2vec_features is not None:
            disease_features = node2vec_features[: artifacts.num_diseases]
            protein_features = node2vec_features[artifacts.num_diseases :]

        model = HeteroLinkPredictor(
            metadata=artifacts.graph_data.metadata(),
            num_diseases=artifacts.num_diseases,
            num_proteins=artifacts.num_proteins,
            layer_kind=layer_kind,
            layer_sizes=layer_sizes,
            heads=cfg.heads,
            disease_features=disease_features,
            protein_features=protein_features,
        )
        return model.to(device)

    model = GNNLinkPredictor(
        num_nodes=artifacts.num_diseases + artifacts.num_proteins,
        layer_kind=layer_kind,
        layer_sizes=layer_sizes,
        heads=cfg.heads,
        init_features=node2vec_features,
    )
    return model.to(device)


def load_model_weights(model: torch.nn.Module, ckpt_path: str | Path, device) -> torch.nn.Module:
    state_dict = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state_dict)
    return model
