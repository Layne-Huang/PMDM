from .cftfm import CFTransformerEncoder
from .edge import *
from .egnn import EGNN_Sparse_Network
from .gin import *
from .schnet import SchNetEncoder_protein, CASchNetEncoder, SchNetEncoder_pure
from .schnet_geo import SchNetEncoder, SchNetEncoder_pocket


def get_encoder(config):
    if config.name == 'schnet':
        return SchNetEncoder(
            hidden_channels=config.hidden_channels,
            num_filters=config.num_filters,
            num_interactions=config.num_interactions,
            edge_channels=config.edge_channels,
            cutoff=config.cutoff,
        )
    elif config.name == 'cftfm':
        return CFTransformerEncoder(
            hidden_channels=config.hidden_channels,
            edge_channels=config.edge_channels,
            key_channels=config.key_channels,
            num_heads=config.num_heads,
            num_interactions=config.num_interactions,
            k=config.knn,
            cutoff=config.cutoff,
        )
    elif config.name == 'egnn':
        return EGNN_Sparse_Network(
            n_layers=config.layer,
            # feats_dim = config.hidden_channels,
            feats_dim=config.feats_dim,
            edge_attr_dim=config.edge_attr_dim,
            m_dim=config.m_dim,
            soft_edge=config.soft_edges,
            norm_coors=config.norm_coors,
            # soft_edges = True
            aggr='sum'
            # cutoff = config.cutoff,
        )
    else:
        raise NotImplementedError('Unknown encoder: %s' % config.name)
