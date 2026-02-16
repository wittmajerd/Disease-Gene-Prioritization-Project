from __future__ import annotations

# GNN alap — to_hetero()-val wrappelve, VAGY natív HGT
class GNNEncoder(nn.Module):     # configurable layers, to_hetero wrapping
class HGTEncoder(nn.Module):     # natív heterogén (nem to_hetero)

def build_encoder(metadata, model_cfg) -> nn.Module    # factory

# Decoder
class DotProductDecoder(nn.Module)
class MLPDecoder(nn.Module)

def build_decoder(model_cfg) -> nn.Module               # factory

# Összetett modell
class HeteroLinkPredictor(nn.Module):
    emb: ParameterDict             # learnable embedding per node type
    encoder: nn.Module             # GNN encoder
    decoder: nn.Module             # decoder
    
    def encode(data) -> dict[str, Tensor]
    def decode(z_dict, edge_label_index, src_type, dst_type) -> Tensor
    def forward(data, edge_label_index, src_type, dst_type) -> Tensor

# Node2Vec featurizer - otioanl
class Node2VecFeaturizer: