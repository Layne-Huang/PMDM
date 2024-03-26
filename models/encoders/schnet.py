from math import pi as PI

import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding
from torch_geometric.nn import MessagePassing, radius_graph

from .attention import BasicTransformerBlock
from ..common import GaussianSmearing, ShiftedSoftplus


class CFConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_filters, edge_channels, cutoff=10.0, smooth=False):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = Sequential(
            Linear(edge_channels, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )  # Network for generating filter weights
        self.cutoff = cutoff
        self.smooth = smooth

    def forward(self, x, edge_index, edge_length, edge_attr):
        W = self.nn(edge_attr)

        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        # if self.cutoff is not None:
        #     C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
        #     C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)     # Modification: cutoff
        W = W * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(Module):

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth=False):
        super(InteractionBlock, self).__init__()
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, num_gaussians, cutoff, smooth)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)
        # self.bn = BatchNorm(hidden_channels)
        # self.gn = GraphNorm(hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        # x = self.bn(x)
        # x = self.gn(x)
        return x


class SchNetEncoder_protein(Module):

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=64, cutoff=10.0, input_dim=27):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.input_dim = input_dim
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.cutoff = cutoff
        self.emblin = Linear(self.input_dim, hidden_channels)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth=True)
            self.interactions.append(block)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch):
        edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        h = self.emblin(node_attr)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        # batch = batch.squeeze(0)
        # # print(batch.size())
        # # print(h.size())
        # h = scatter_mean(h, batch, dim=0)
        # h = h.index_select(0, batch_ligand)
        return h


class SchNetEncoder(Module):

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False, input_dim=5, time_emb=True,
                 context=False):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.input_dim = input_dim
        self.time_emb = time_emb
        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)
        self.emblin = Linear(self.input_dim, hidden_channels)  # 16 or 8
        self.context = context

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth)
            self.interactions.append(block)
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels)
        if context:
            self.fc1_m = Linear(hidden_channels, 256)
            self.fc2_m = Linear(256, 64)
            self.fc3_m = Linear(64, input_dim)

            # Mapping to [c], cmean
            self.fc1_v = Linear(hidden_channels, 256)
            self.fc2_v = Linear(256, 64)
            self.fc3_v = Linear(64, input_dim)

    def forward(self, z, edge_index, edge_length, edge_attr, embed_node=True):
        if edge_attr is None:
            edge_attr = self.distance_expansion(edge_length)
        if z.dim() == 1 and z.dtype == torch.long:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)

        else:
            # h = z # default
            if self.time_emb:
                z, ptemb = z[:, :self.input_dim], z[:, self.input_dim:]
                h = self.emblin(z) + ptemb
            else:
                h = self.emblin(z)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)

        if self.context:
            m = F.relu(self.fc1_m(h))
            m = F.relu(self.fc2_m(m))
            m = self.fc3_m(m)
            v = F.relu(self.fc1_v(h))
            v = F.relu(self.fc2_v(v))
            v = self.fc3_v(v)
            return m, v
        else:
            return h


class CASchNetEncoder(Module):
    '''
    cross attention schnet encoder
    '''

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False, input_dim=5,
                 n_head=8, d_dim=32, time_emb=True, context=False):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.input_dim = input_dim
        self.time_emb = time_emb
        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)
        self.emblin = Linear(self.input_dim, hidden_channels)  # 16 or 8
        self.context = context

        self.interactions = ModuleList()
        self.crossattns = ModuleList()
        self.atten_layer = BasicTransformerBlock(hidden_channels, n_head, d_dim, 0.1, hidden_channels)
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth)
            self.interactions.append(block)
            # atten_layer = BasicTransformerBlock(hidden_channels,n_head,d_dim,0.1,hidden_channels)
            # self.crossattns.append(atten_layer)

        if context:
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels)
            self.fc1_m = Linear(hidden_channels, 256)
            self.fc2_m = Linear(256, 64)
            self.fc3_m = Linear(64, input_dim)

            # Mapping to [c], cmean
            self.fc1_v = Linear(hidden_channels, 256)
            self.fc2_v = Linear(256, 64)
            self.fc3_v = Linear(64, input_dim)

    def forward(self, z, p_ctx, edge_index, edge_length, edge_attr, embed_node=True):
        if edge_attr is None:
            edge_attr = self.distance_expansion(edge_length)
        if z.dim() == 1 and z.dtype == torch.long:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)

        else:
            # h = z # default
            if self.time_emb:
                z, ptemb = z[:, :self.input_dim], z[:, self.input_dim:]
                h = self.emblin(z) + ptemb
            else:
                h = self.emblin(z)
        # h = self.atten_layer(h,p_ctx)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        # for interaction,crossattn in zip(self.interactions,self.crossattns):
        #     h = crossattn(h,p_ctx)
        #     h = h + interaction(h, edge_index, edge_length, edge_attr)

        if self.context:
            m = F.relu(self.fc1_m(h))
            m = F.relu(self.fc2_m(m))
            m = self.fc3_m(m)
            v = F.relu(self.fc1_v(h))
            v = F.relu(self.fc2_v(v))
            v = self.fc3_v(v)
            return m, v
        else:
            return h


class SchNetEncoder_pure(Module):
    '''
    cross attention schnet encoder
    '''

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False, input_dim=5,
                 time_emb=True, context=False):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.input_dim = input_dim
        self.time_emb = time_emb
        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)
        self.context = context

        self.interactions = ModuleList()

        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth)
            self.interactions.append(block)

        if context:
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels)
            self.fc1_m = Linear(hidden_channels, 256)
            self.fc2_m = Linear(256, 64)
            self.fc3_m = Linear(64, input_dim)

            # Mapping to [c], cmean
            self.fc1_v = Linear(hidden_channels, 256)
            self.fc2_v = Linear(256, 64)
            self.fc3_v = Linear(64, input_dim)

    def forward(self, z, p_ctx, edge_index, edge_length, edge_attr, embed_node=True):
        if edge_attr is None:
            edge_attr = self.distance_expansion(edge_length)
        if z.dim() == 1 and z.dtype == torch.long:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)

        else:
            h = z  # default
            # if self.time_emb:
            #     z, ptemb = z[:,:self.input_dim],z[:,self.input_dim:]
            #     h = self.emblin(z)+ptemb
            # else:
            #     h = self.emblin(z)
        # h = self.atten_layer(h,p_ctx)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        # for interaction,crossattn in zip(self.interactions,self.crossattns):
        #     h = crossattn(h,p_ctx)
        #     h = h + interaction(h, edge_index, edge_length, edge_attr)

        if self.context:
            m = F.relu(self.fc1_m(h))
            m = F.relu(self.fc2_m(m))
            m = self.fc3_m(m)
            v = F.relu(self.fc1_v(h))
            v = F.relu(self.fc2_v(v))
            v = self.fc3_v(v)
            return m, v
        else:
            return h
