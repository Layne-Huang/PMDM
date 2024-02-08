from math import pi as PI

import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding, SiLU, Parameter
from torch_geometric.nn import MessagePassing

from ..geometry import get_distance

Adj = object
Size = object
OptTensor = object
Tensor = object


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


class CoorsNorm(Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AsymmetricSineCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        num_basis_k = num_basis // 2
        num_basis_l = num_basis - num_basis_k
        self.register_buffer('freq_k', torch.arange(1, num_basis_k + 1).float())
        self.register_buffer('freq_l', torch.arange(1, num_basis_l + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0) + self.freq_l.size(0)

    def forward(self, angle):
        # If we don't incorporate `cos`, the embedding of 0-deg and 180-deg will be the
        #  same, which is undesirable.
        s = torch.sin(angle.view(-1, 1) * self.freq_k.view(1, -1))  # (num_angles, num_basis_k)
        c = torch.cos(angle.view(-1, 1) * self.freq_l.view(1, -1))  # (num_angles, num_basis_l)
        return torch.cat([s, c], dim=-1)


class SymmetricCosineSmearing(Module):

    def __init__(self, num_basis=50):
        super().__init__()
        self.register_buffer('freq_k', torch.arange(1, num_basis + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.freq_k.view(1, -1))  # (num_angles, num_basis)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff, smooth):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        # m_ij = self.message(x,W)
        x, m_ij = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x, m_ij

    def message(self, x_j, W):
        return x_j * W

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        #  get messages
        m_ij = self.message(**msg_kwargs)
        m_i = self.aggregate(m_ij, **aggr_kwargs)
        return self.update(m_i, **update_kwargs), m_ij


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth):
        super(InteractionBlock, self).__init__()
        mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, mlp, cutoff, smooth)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels * 2, hidden_channels)
        # self.bn = BatchNorm(hidden_channels)
        # self.gn = GraphNorm(hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        m_i, m_ij = self.conv(x, edge_index, edge_length, edge_attr)
        m_i = self.act(m_i)
        x = self.lin(torch.cat([x, m_i], dim=1))
        # x = self.bn(x)
        # x = self.gn(x)
        return x, m_ij


class CoordBlock(MessagePassing):
    def __init__(self, input_channels, hidden_channels):
        super(CoordBlock, self).__init__(aggr='add')
        layer = Linear(hidden_channels, 3, bias=False)
        self.coord_net = Sequential(
            Linear(input_channels, hidden_channels),
            SiLU(),
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            layer)

    def forward(self, r, h, edge_index, ligand_batch):
        dist = get_distance(r, edge_index).view(-1, 1)
        coord_emb = self.coord_net(h)
        m = self.propagate(edge_index, x=coord_emb, W=dist)
        return m[:len(ligand_batch), :] / 100  # norm value

    def message(self, x_j, W):
        return W * x_j


class EGNNBlock(MessagePassing):
    def __init__(self, input_channels, hidden_channels):
        super(EGNNBlock, self).__init__(aggr='add')
        layer = Linear(hidden_channels, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coords_range = 15
        self.coors_norm = CoorsNorm(scale_init=1e-2)
        self.coord_net = Sequential(
            Linear(input_channels * 1, hidden_channels * 2),
            SiLU(),
            Linear(hidden_channels * 2, hidden_channels),
            SiLU(),
            layer)

    def forward(self, r, m_ij, edge_index, edge_attr, ligand_batch):
        row, col = edge_index
        edge_length, coord_diff = coord2diff(r, edge_index)
        coord_diff = self.coors_norm(coord_diff)
        # coord_emb = self.coord_net(torch.cat([m_i[row],m_i[col]],dim=1))
        coord_emb = self.coord_net(m_ij)
        m = self.propagate(edge_index, x=coord_diff, W=coord_emb)
        return m[:len(ligand_batch), :]  # norm value

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        #  get messages
        m_ij = kwargs['W'] * kwargs['x']
        m_i = self.aggregate(m_ij, **aggr_kwargs)
        return self.update(m_i, **update_kwargs)


# class EGNNBlock(Module):
#     def __init__(self,input_channels,hidden_channels):
#         super(EGNNBlock, self).__init__()
#         layer = Linear(hidden_channels, 3, bias=False)
#         torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
#         self.coords_range = 15
#         self.coord_net = Sequential(
#             Linear(3*input_channels+1, hidden_channels),
#             SiLU(),
#             Linear(hidden_channels, hidden_channels),
#             SiLU(),
#             layer)

#     def forward(self,r,h,edge_index,edge_attr,ligand_batch):
#         row, col = edge_index
#         edge_length, coord_diff = coord2diff(r,edge_index)
#         input_tensor = torch.cat([h[row], h[col], edge_length, edge_attr], dim=1)
#         trans = coord_diff * torch.tanh(self.coord_net(input_tensor))*15
#         agg = unsorted_segment_sum(trans, row, num_segments=r.size(0),
#                                    normalization_factor=100,
#                                    aggregation_method='sum')

#         return agg[:len(ligand_batch),:] # norm value


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
        self.crossattns = ModuleList()
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


class SchNetEncoder_pocket(Module):

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
        # self.emblin_p = Linear(self.input_dim, hidden_channels)
        self.protein_encoder = Sequential(
            Linear(self.input_dim, 128),
            torch.nn.SiLU(),
            Linear(128, hidden_channels)
        )
        self.ligand_encoder = Sequential(
            Linear(self.input_dim, 128),
            torch.nn.SiLU(),
            Linear(128, hidden_channels)
        )
        self.context = context

        self.interactions = ModuleList()
        self.coordlayers = ModuleList()
        self.crossattns = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels,
                                     num_filters, cutoff, smooth)
            self.interactions.append(block)
            coord_block = EGNNBlock(hidden_channels, hidden_channels)
            self.coordlayers.append(coord_block)
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels)
        if context:
            self.fc1_m = Linear(hidden_channels, 256)
            self.fc2_m = Linear(256, 64)
            self.fc3_m = Linear(64, input_dim)

            # Mapping to [c], cmean
            self.fc1_v = Linear(hidden_channels, 256)
            self.fc2_v = Linear(256, 64)
            self.fc3_v = Linear(64, input_dim)

    def forward(self, z, r, edge_index, edge_length, edge_attr, ligand_batch, ctx, embed_node=True):
        if edge_attr is None:
            edge_attr = self.distance_expansion(edge_length)
        if z.dim() == 1 and z.dtype == torch.long:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)

        else:
            # h = z # default
            if self.time_emb:
                # z, ptemb = z[:,:self.input_dim],z[:,self.input_dim:]
                h_ligand = self.ligand_encoder(z[:len(ligand_batch), :])
                h_protein = self.protein_encoder(z[len(ligand_batch):, :])
                h_ligand = h_ligand + ctx
                # h = self.emblin(z)+ptemb
            else:
                # h = self.emblin(z)
                h_ligand = self.ligand_encoder(z[:len(ligand_batch), :])
                h_protein = self.protein_encoder(z[len(ligand_batch):, :])

        h = torch.cat([h_ligand, h_protein], dim=0)
        r_0 = r[:len(ligand_batch), :]
        for interaction, coordblock in zip(self.interactions, self.coordlayers):
            edge_length = get_distance(r, edge_index)
            # edge_length, coord_diff = coord2diff(r, edge_index) #egnn
            h, m_ij = interaction(h, edge_index, edge_length, edge_attr)
            # h = h + m_ij
            r = torch.cat([r[:len(ligand_batch), :] + coordblock(r, m_ij, edge_index, edge_attr, ligand_batch),
                           r[len(ligand_batch):, :]], dim=0)
        h = h[:len(ligand_batch), :]
        r = r[:len(ligand_batch), :]
        r = r - r_0  # egnn

        if self.context:
            m = F.relu(self.fc1_m(h))
            m = F.relu(self.fc2_m(m))
            m = self.fc3_m(m)
            v = F.relu(self.fc1_v(h))
            v = F.relu(self.fc2_v(v))
            v = self.fc3_v(v)
            return m, v
        else:
            return h, r

# class SchNetEncoder_pocket(Module):

#     def __init__(self, hidden_channels=128, num_filters=128,
#                 num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False, input_dim=5, time_emb=True, context=False):
#         super().__init__()

#         self.hidden_channels = hidden_channels
#         self.num_filters = num_filters
#         self.num_interactions = num_interactions
#         self.cutoff = cutoff
#         self.input_dim=input_dim
#         self.time_emb = time_emb
#         self.embedding = Embedding(100, hidden_channels, max_norm=10.0)
#         self.emblin = Linear(self.input_dim, hidden_channels) # 16 or 8
#         self.context = context


#         self.interactions = ModuleList()
#         self.crossattns = ModuleList()
#         for _ in range(num_interactions):
#             block = InteractionBlock(hidden_channels, edge_channels,
#                                      num_filters, cutoff, smooth)
#             self.interactions.append(block)
#         self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels)
#         if context:
#             self.fc1_m = Linear(hidden_channels, 256)
#             self.fc2_m = Linear(256, 64)
#             self.fc3_m = Linear(64, input_dim)

#             # Mapping to [c], cmean
#             self.fc1_v = Linear(hidden_channels, 256)
#             self.fc2_v = Linear(256, 64)
#             self.fc3_v = Linear(64, input_dim)


#     def forward(self, z, edge_index, edge_length, edge_attr, ligand_batch, embed_node=True):
#         if edge_attr is None:
#             edge_attr = self.distance_expansion(edge_length)
#         if z.dim() == 1 and z.dtype == torch.long:
#             assert z.dim() == 1 and z.dtype == torch.long
#             h = self.embedding(z)

#         else:
#             # h = z # default
#             if self.time_emb:
#                 z, ptemb = z[:,:self.input_dim],z[:,self.input_dim:]
#                 h = self.emblin(z)
#                 ligand_emb = h[:len(ligand_batch),:]+ptemb[:len(ligand_batch),:]
#             else:
#                 h = self.emblin(z)
#         protein_emb = h[len(ligand_batch):,:]
#         h = torch.cat([ligand_emb,protein_emb],dim=0)
#         for interaction in self.interactions:
#             h = h + interaction(h, edge_index, edge_length, edge_attr)
#             # h = torch.cat([h[:len(ligand_batch),:],protein_emb],dim=0)

#         if self.context:
#             m = F.relu(self.fc1_m(h))
#             m = F.relu(self.fc2_m(m))
#             m = self.fc3_m(m)
#             v = F.relu(self.fc1_v(h))
#             v = F.relu(self.fc2_v(v))
#             v = self.fc3_v(v)
#             return m,v
#         else:
#             return h
