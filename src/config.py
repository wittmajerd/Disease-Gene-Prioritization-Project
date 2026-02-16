from __future__ import annotations


@dataclass GraphConfig:     node_types, edge_types, target_edge, add_reverse_edges
@dataclass ModelConfig:     encoder_type, hidden_dim, num_layers, output_dim, heads, dropout, decoder_type
@dataclass TrainingConfig:  lr, weight_decay, epochs, patience, eval_every, neg_ratio
@dataclass SplitConfig:     train_ratio, val_ratio
@dataclass LoggingConfig:   use_wandb, project, run_name
@dataclass PipelineConfig:  graph, model, training, split, logging, seed

def load_config(path) -> PipelineConfig   # YAML → nested dataclasses
