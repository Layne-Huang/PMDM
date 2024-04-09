import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.AllChem import EmbedMolecule
from torch_geometric.transforms import Compose
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_dense_adj, dense_to_sparse, remove_self_loops, dropout_adj, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import subgraph
from torch_scatter import scatter_add
from torch_sparse import coalesce

from .chem import BOND_TYPES, BOND_NAMES, get_atom_symbol
from .data import ProteinLigandData
from .misc import get_adj_matrix
from .protein_ligand import ATOM_FAMILIES

from evaluation.sascorer import *
from rdkit.Chem.Descriptors import MolLogP, qed

# Crossdock atom element
C_ligand_element = [1, 6, 7, 8, 9, 15, 16, 17]
C_protein_element = [1, 6, 7, 8, 16, 34]

# PDBind atom element
P_ligand_element = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 23, 26, 27, 29, 33, 34, 35, 44, 51, 53, 78]
P_ligand_element_100 = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 23, 26, 29, 33, 34, 35, 44, 51, 53, 78]
P_ligand_element_filter = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
# P_ligand_element_filter = [1, 35, 5, 6, 7, 8, 9, 15, 16, 17, 53]
P_protein_element = [1, 6, 7, 8, 16]


class FeaturizeProteinAtom(object):

    def __init__(self, dataset='crossdock', pocket=False):
        super().__init__()
        if dataset == 'crossdock' or 'pdbind':
            if not pocket:
                self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])  # crossdock H, C, N, O, S, Se
            else:
                self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17, 34, 119])  # crossdock pocket
        # if dataset == 'pdbind':
        #     self.atomic_numbers = torch.LongTensor(P_protein_element)
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data: ProteinLigandData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)  # full_feature
        data.protein_atom_feature_full = x
        data.protein_atom_feature = element
        return data


class FeaturizeLigandAtom(object):

    def __init__(self, dataset='crossdock', pocket=False, pdbind_filter='original'):
        super().__init__()
        if dataset == 'crossdock' or 'pdbind':
            if not pocket:
                self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])
            else:
                self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17, 34, 119])
        # if dataset == 'pdbind':
        #     if pdbind_filter == 'original':
        #         self.atomic_numbers = torch.LongTensor(P_ligand_element)
        #     if pdbind_filter == 'length_100':
        #         self.atomic_numbers = torch.LongTensor(P_ligand_element_100)
        #     if pdbind_filter == 'filter':
        #         self.atomic_numbers = torch.LongTensor(P_ligand_element_filter)

    @property
    def num_properties(self):
        return len(ATOM_FAMILIES)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + len(ATOM_FAMILIES)

    def __call__(self, data: ProteinLigandData):
        element = data.ligand_element.view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        x_full = torch.cat([element, data.ligand_atom_feature], dim=-1)  # full_feature
        data.ligand_atom_feature_full = x_full
        data.ligand_atom_feature = element  # .float()+torch.randn_like(element.float()).to(element.device)
        data.ligand_atom_feature_charge = torch.cat([data.ligand_atom_feature_full,
                                                     data.ligand_element.unsqueeze(1)], dim=1)
        return data


class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1, num_classes=3)  # (1,2,3) to (0,1,2)-onehot
        return data


class LigandCountNeighbors(object):

    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand_num_neighbors = self.count_neighbors(
            data.ligand_bond_index,
            symmetry=True,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_valence = self.count_neighbors(
            data.ligand_bond_index,
            symmetry=True,
            valence=data.ligand_bond_type,
            num_nodes=data.ligand_element.size(0),
        )
        return data


class LigandRandomMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked

    def __call__(self, data: ProteinLigandData):
        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = int(num_atoms * ratio)

        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        idx = np.arange(num_atoms)
        np.random.shuffle(idx)
        idx = torch.LongTensor(idx)
        masked_idx = idx[:num_masked]
        context_idx = idx[num_masked:]

        data.ligand_masked_element = data.ligand_element[masked_idx]
        data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]  # For Prediction
        data.ligand_masked_pos = data.ligand_pos[masked_idx]

        data.ligand_context_element = data.ligand_element[context_idx]
        data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]  # For Input
        data.ligand_context_pos = data.ligand_pos[context_idx]

        data.ligand_context_bond_index, data.ligand_context_bond_feature = subgraph(
            context_idx,
            data.ligand_bond_index,
            edge_attr=data.ligand_bond_feature,
            relabel_nodes=True,
        )
        data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            num_nodes=context_idx.size(0),
        )

        # print(context_idx)
        # print(data.ligand_context_bond_index)

        # mask = torch.logical_and(
        #     (data.ligand_bond_index[0].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        #     (data.ligand_bond_index[1].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        # )
        # print(data.ligand_bond_index[:, mask])

        # print(data.ligand_context_num_neighbors)
        # print(data.ligand_num_neighbors[context_idx])

        data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]

        data._mask = 'random'

        return data


class LigandBFSMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, inverse=False):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        self.inverse = inverse

    @staticmethod
    def get_bfs_perm(nbh_list):
        num_nodes = len(nbh_list)
        num_neighbors = torch.LongTensor([len(nbh_list[i]) for i in range(num_nodes)])

        bfs_queue = [random.randint(0, num_nodes - 1)]
        bfs_perm = []
        num_remains = [num_neighbors.clone()]
        bfs_next_list = {}
        visited = {bfs_queue[0]}

        num_nbh_remain = num_neighbors.clone()

        while len(bfs_queue) > 0:
            current = bfs_queue.pop(0)
            for nbh in nbh_list[current]:
                num_nbh_remain[nbh] -= 1
            bfs_perm.append(current)
            num_remains.append(num_nbh_remain.clone())
            next_candid = []
            for nxt in nbh_list[current]:
                if nxt in visited: continue
                next_candid.append(nxt)
                visited.add(nxt)

            random.shuffle(next_candid)
            bfs_queue += next_candid
            bfs_next_list[current] = copy.copy(bfs_queue)

        return torch.LongTensor(bfs_perm), bfs_next_list, num_remains

    def __call__(self, data):
        bfs_perm, bfs_next_list, num_remaining_nbs = self.get_bfs_perm(data.ligand_nbh_list)

        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = int(num_atoms * ratio)
        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        if self.inverse:
            masked_idx = bfs_perm[:num_masked]
            context_idx = bfs_perm[num_masked:]
        else:
            masked_idx = bfs_perm[-num_masked:]
            context_idx = bfs_perm[:-num_masked]

        data.ligand_masked_element = data.ligand_element[masked_idx]
        data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]  # For Prediction
        data.ligand_masked_pos = data.ligand_pos[masked_idx]

        data.ligand_context_element = data.ligand_element[context_idx]
        data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]  # For Input
        data.ligand_context_pos = data.ligand_pos[context_idx]

        data.ligand_context_bond_index, data.ligand_context_bond_feature = subgraph(
            context_idx,
            data.ligand_bond_index,
            edge_attr=data.ligand_bond_feature,
            relabel_nodes=True,
        )
        data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            num_nodes=context_idx.size(0),
        )

        # print(context_idx)
        # print(data.ligand_context_bond_index)

        # mask = torch.logical_and(
        #     (data.ligand_bond_index[0].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        #     (data.ligand_bond_index[1].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        # )
        # print(data.ligand_bond_index[:, mask])

        # print(data.ligand_context_num_neighbors)
        # print(data.ligand_num_neighbors[context_idx])

        data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]

        data._mask = 'invbfs' if self.inverse else 'bfs'

        return data


class LigandMaskAll(LigandRandomMask):

    def __init__(self):
        super().__init__(min_ratio=1.0)


class LigandMixedMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, p_random=0.5, p_bfs=0.25,
                 p_invbfs=0.25):
        super().__init__()

        self.t = [
            LigandRandomMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=False),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=True),
        ]
        self.p = [p_random, p_bfs, p_invbfs]

    def __call__(self, data):
        f = random.choices(self.t, k=1, weights=self.p)[0]
        return f(data)


def get_mask(cfg):
    if cfg.type == 'bfs':
        return LigandBFSMask(
            min_ratio=cfg.min_ratio,
            max_ratio=cfg.max_ratio,
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
        )
    elif cfg.type == 'random':
        return LigandRandomMask(
            min_ratio=cfg.min_ratio,
            max_ratio=cfg.max_ratio,
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
        )
    elif cfg.type == 'mixed':
        return LigandMixedMask(
            min_ratio=cfg.min_ratio,
            max_ratio=cfg.max_ratio,
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
            p_random=cfg.p_random,
            p_bfs=cfg.p_bfs,
            p_invbfs=cfg.p_invbfs,
        )
    elif cfg.type == 'all':
        return LigandMaskAll()
    else:
        raise NotImplementedError('Unknown mask: %s' % cfg.type)


class ContrastiveSample(object):

    def __init__(self, num_real=50, num_fake=50, pos_real_std=0.05, pos_fake_std=2.0, elements=None):
        super().__init__()
        self.num_real = num_real
        self.num_fake = num_fake
        self.pos_real_std = pos_real_std
        self.pos_fake_std = pos_fake_std
        if elements is None:
            # elements = torch.LongTensor([
            #     1, 3, 5, 6, 7, 8, 9, 
            #     12, 13, 14, 15, 16, 17, 
            #     21, 23, 24, 26, 27, 29, 33, 34, 35, 
            #     39, 42, 44, 50, 53, 74, 79, 80
            # ])
            elements = [1, 6, 7, 8, 9, 15, 16, 17]
        self.elements = torch.LongTensor(elements)

    @property
    def num_elements(self):
        return self.elements.size(0)

    def __call__(self, data: ProteinLigandData):
        # Positive samples
        pos_real_mode = data.ligand_masked_pos
        element_real = data.ligand_masked_element
        ind_real = data.ligand_masked_feature
        cls_real = data.ligand_masked_element.view(-1, 1) == self.elements.view(1, -1)
        if not (cls_real.sum(-1) > 0).all():
            print(data.ligand_element)
        assert (cls_real.sum(-1) > 0).all(), 'Unexpected elements.'

        real_sample_idx = np.random.choice(np.arange(pos_real_mode.size(0)), size=self.num_real)
        data.pos_real = pos_real_mode[real_sample_idx]
        data.pos_real += torch.randn_like(data.pos_real) * self.pos_real_std
        data.element_real = element_real[real_sample_idx]
        data.cls_real = cls_real[real_sample_idx]
        data.ind_real = ind_real[real_sample_idx]

        # Negative samples
        pos_fake_mode = torch.cat([data.ligand_context_pos, data.protein_pos], dim=0)
        fake_sample_idx = np.random.choice(np.arange(pos_fake_mode.size(0)), size=self.num_fake)
        data.pos_fake = pos_fake_mode[fake_sample_idx]
        data.pos_fake += torch.randn_like(data.pos_fake) * self.pos_fake_std

        return data


class Pos2Distance(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        # Positive samples
        protein_pos = data.protein_pos
        ligand_pos = data.ligand_pos
        protein_pos_mean = torch.mean(protein_pos, 0)
        # protein_pos_x_mean = protein_pos_mean[0]
        # protein_pos_y_mean = protein_pos_mean[1]
        # protein_pos_z_mean = protein_pos_mean[2]_

        ligand_pos_dis = ligand_pos - protein_pos_mean
        data.ligand_pos_dis = ligand_pos_dis
        # data.ligand_pos = ligand_pos_dis/20
        data.protein_pos_mean = protein_pos_mean

        return data


class RemoveMean(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        # Positive samples
        ligand_pos = data.ligand_pos
        ligand_pos_mean = torch.mean(ligand_pos, 0)
        # protein_pos_x_mean = protein_pos_mean[0]
        # protein_pos_y_mean = protein_pos_mean[1]
        # protein_pos_z_mean = protein_pos_mean[2]_

        ligand_pos_com = ligand_pos - ligand_pos_mean
        data.ligand_pos = ligand_pos_com
        data.ligand_pos_mean = ligand_pos_mean

        return data


def get_contrastive_sampler(cfg):
    return ContrastiveSample(
        num_real=cfg.num_real,
        num_fake=cfg.num_fake,
        pos_real_std=cfg.pos_real_std,
        pos_fake_std=cfg.pos_fake_std,
    )


class AddHigherOrderEdges(object):

    def __init__(self, order, num_types=len(BOND_TYPES)):
        super().__init__()
        self.order = order
        self.num_types = num_types

    def binarize(self, x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(self, adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        """
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    self.binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

        for i in range(2, order + 1):
            adj_mats.append(self.binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    def __call__(self, data: Data):
        N = data.ligand_nodes
        print(data.ligand_bond_index.size())
        adj = to_dense_adj(data.ligand_bond_index).squeeze(0)
        adj_order = self.get_higher_order_adj_matrix(adj, self.order)  # (N, N)

        type_mat = to_dense_adj(data.ligand_bond_index, edge_attr=data.ligand_bond_type).squeeze(0)  # (N, N)
        type_highorder = torch.where(adj_order > 1, self.num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.ligand_bond_edge_index = data.ligand_bond_index  # Save original edges
        # print(N)
        # print(new_edge_index.size())
        data.ligand_edge_index, data.ligand_edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)  # modify data
        # print(new_edge_index.size())
        # print(edge_order.size())
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N)  # modify data
        data.ligand_is_bond = (data.ligand_edge_type < self.num_types)
        assert (data.ligand_edge_index == edge_index_1).all()

        return data


class AddEdgeLength(object):

    def __call__(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1)  # (num_edge, 1)
        data.edge_length = d
        return data

    # Add attribute placeholder for data object, so that we can use batch.to_data_list


class AddPlaceHolder(object):
    def __call__(self, data: Data):
        data.pos_gen = -1. * torch.ones_like(data.pos)
        data.d_gen = -1. * torch.ones_like(data.edge_length)
        data.d_recover = -1. * torch.ones_like(data.edge_length)
        return data


class AddEdgeName(object):

    def __init__(self, asymmetric=True):
        super().__init__()
        self.bonds = copy.deepcopy(BOND_NAMES)
        self.bonds[len(BOND_NAMES) + 1] = 'Angle'
        self.bonds[len(BOND_NAMES) + 2] = 'Dihedral'
        self.asymmetric = asymmetric

    def __call__(self, data: Data):
        data.edge_name = []
        for i in range(data.edge_index.size(1)):
            tail = data.edge_index[0, i]
            head = data.edge_index[1, i]
            if self.asymmetric and tail >= head:
                data.edge_name.append('')
                continue
            tail_name = get_atom_symbol(data.atom_type[tail].item())
            head_name = get_atom_symbol(data.atom_type[head].item())
            name = '%s_%s_%s_%d_%d' % (
                self.bonds[data.edge_type[i].item()] if data.edge_type[i].item() in self.bonds else 'E' + str(
                    data.edge_type[i].item()),
                tail_name,
                head_name,
                tail,
                head,
            )
            if hasattr(data, 'edge_length'):
                name += '_%.3f' % (data.edge_length[i].item())
            data.edge_name.append(name)
        return data


class AddAngleDihedral(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def iter_angle_triplet(bond_mat):
        n_atoms = bond_mat.size(0)
        for j in range(n_atoms):
            for k in range(n_atoms):
                for l in range(n_atoms):
                    if bond_mat[j, k].item() == 0 or bond_mat[k, l].item() == 0: continue
                    if (j == k) or (k == l) or (j >= l): continue
                    yield (j, k, l)

    @staticmethod
    def iter_dihedral_quartet(bond_mat):
        n_atoms = bond_mat.size(0)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i >= j: continue
                if bond_mat[i, j].item() == 0: continue
                for k in range(n_atoms):
                    for l in range(n_atoms):
                        if (k in (i, j)) or (l in (i, j)): continue
                        if bond_mat[k, i].item() == 0 or bond_mat[l, j].item() == 0: continue
                        yield (k, i, j, l)

    def __call__(self, data: Data):
        N = data.num_nodes
        if 'is_bond' in data:
            bond_mat = to_dense_adj(data.edge_index, edge_attr=data.is_bond).long().squeeze(0) > 0
        else:
            bond_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).long().squeeze(0) > 0

        # Note: if the name of attribute contains `index`, it will automatically
        #       increases during batching.
        data.angle_index = torch.LongTensor(list(self.iter_angle_triplet(bond_mat))).t()
        data.dihedral_index = torch.LongTensor(list(self.iter_dihedral_quartet(bond_mat))).t()

        return data


class CountNodesPerGraph(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.num_nodes_per_graph = torch.LongTensor([data.ligand_element.size(0)])
        data.ligand_nodes = torch.LongTensor([data.ligand_element.size(0)])
        data.protein_nodes = torch.LongTensor([data.protein_element.size(0)])
        # print(data.ligand_element.size(0))
        return data


class GetConformerPos(object):

    def __init__(self) -> None:
        super().__init__()
        self.path = './data/crossdocked_pocket10'

    def __call__(self, data):
        path = os.path.join(self.path, data.ligand_filename)
        rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=False)))
        rdmol = Chem.AddHs(rdmol)
        EmbedMolecule(rdmol, useRandomCoords=True)
        rdmol = Chem.RemoveHs(rdmol)
        conf_pos = torch.tensor(rdmol.GetConformer(0).GetPositions(), dtype=torch.float32)
        data.rdmol = rdmol
        data.conf_pos = conf_pos
        return data


class GetAdj(object):
    def __init__(self, g_cutoff=None, only_prot=False) -> None:
        super().__init__()
        self.cutoff = g_cutoff
        self.only_prot = only_prot

    def __call__(self, data):
        '''
        full connected edges or radius edges
        '''
        ligand_n_particles = data.ligand_nodes
        if not self.only_prot:
            if self.cutoff is None:
                ligand_adj = get_adj_matrix(ligand_n_particles)
            else:
                ligand_adj = radius_graph(data.ligand_pos, self.cutoff, batch=None, loop=False)
            data.ligand_bond_index = ligand_adj
            ligand_bond_type = torch.ones(ligand_adj.size(1), dtype=torch.long) * 2
            data.ligand_bond_type = ligand_bond_type
        
        data.ligand_edge_index = data.ligand_bond_index
        data.ligand_edge_type = data.ligand_bond_type

        protein_n_particles = data.protein_pos.size(0)
        # protein_adj = get_adj_matrix(protein_n_particles)
        protein_adj = radius_graph(data.protein_pos, 6, batch=None, loop=False)
        protein_bond_type = torch.ones(protein_adj.size(1), dtype=torch.long) * 4  # define the protien edge type as 2
        data.protein_bond_index = protein_adj
        data.protein_bond_type = protein_bond_type
        return data


class RandomEdge(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        '''
        random generate the edge_index and edge_type
        '''

        # adj = np.random.randint(2,size=(data.ligand_bond_index.size(0),data.ligand_bond_index.size(1)))
        # coo_A = coo_matrix(adj)
        # edge_index = [coo_A.row, coo_A.col]
        # r_edge_index  = torch.tensor(edge_index, dtype=torch.long)
        # data.ligand_bond_index = r_edge_index

        adj = to_dense_adj(data.ligand_bond_index).squeeze(0)
        adj, _ = dense_to_sparse(torch.randint_like(adj, 2))
        print(data.ligand_bond_index)

        adj = remove_self_loops(adj)
        # print(adj)
        # print(adj[0])
        adj = dropout_adj(adj[0], p=0.9, force_undirected=True)
        # print(adj)
        r_edge_index = to_undirected(adj[0])
        # print(r_edge_index)

        data.ligand_bond_index = r_edge_index
        # exit()

        r_edge_type = torch.randint(1, 5, (data.ligand_bond_index.size(1),))
        data.ligand_bond_type = r_edge_type

        print(data.ligand_bond_index)
        print(data.ligand_bond_index.size())
        print(data.ligand_bond_type)
        print(data.ligand_bond_type.size())

        return data


class RadiusEdge(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        '''
        generate radius edge_index
        '''
        # print(data.ligand_bond_index)
        # print(data.ligand_bond_index.size())
        # pos_init = torch.randn(data.ligand_element.size(0), 3)
        data.ligand_bond_index = radius_graph(data.ligand_pos, 2, batch=None, loop=False)
        # print(data.ligand_bond_index)
        # print(data.ligand_bond_index.size())
        data.ligand_bond_type = torch.ones(data.ligand_bond_index.size(1), dtype=torch.long)
        # print(data.ligand_bond_type.size())

        return data


class Merge_pl(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data, ligand_nodes=0):
        '''
        Merge protein data and ligand data
        '''
        data.pocket_atom_type = torch.cat([data.ligand_atom_feature, data.protein_atom_feature], dim=0)
        data.pocket_pos = torch.cat([data.ligand_pos, data.protein_pos], dim=0)
        if ligand_nodes == 0:
            protein_bond_index = data.protein_bond_index + data.ligand_nodes
        else:
            protein_bond_index = data.protein_bond_index + ligand_nodes
        # pocket_bond_index = torch.cat([data.ligand_bond_index,protein_bond_index], dim=1)
        # ligand = data.ligand_bond_index
        data.pocket_bond_index = torch.cat([data.ligand_bond_index, protein_bond_index], dim=1)
        data.pocket_bond_type = torch.cat([data.ligand_bond_type, data.protein_bond_type])
        data.pocket_element = torch.cat([data.ligand_element, data.protein_element])
        return data


# class Get_protein_emb(object):
#     def __init__(self) -> None:
#         super().__init__()

#     def __call__(self, data):
#         path = data.protein_filename
#         emb_path = os.path.join('./pretrained_emb/pth.files.pocket_embed', path.replace('.pdb', '.pth'))
#         protein_vectors = torch.load(emb_path, map_location='cpu')
#         emb = list(protein_vectors.values())
#         residule_emb = torch.stack(emb, 0)
#         protein_emb = torch.mean(residule_emb, dim=0)
#         data.protein_r_emb = residule_emb.float()
#         data.protein_emb = protein_emb.float()

#         return data

class Property_loss(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        '''
        generate radius edge_index
        '''
        ligand_file = data.ligand_filename
        mol_file = os.path.join('./data/crossdocked_pocket10', ligand_file)
        mol = Chem.SDMolSupplier(mol_file)[0]
        sa, _ = compute_sa_score(mol)
        QED = qed(mol)
        QED_loss = 1-QED
        p_score = sa*QED_loss
        data['p_score'] = p_score
        # data['p_score'] = sa

        return data
