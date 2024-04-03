import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch_geometric.data import Data
from tqdm import tqdm

from .data import ProteinLigandData
from .misc import get_adj_matrix

# n_nodes= {5: 3393, 6: 4848, 4: 9970, 2: 13832, 3: 9482,
#             8: 150, 1: 13364, 7: 53, 9: 48, 
#                 10: 26, 12: 25}
# n_nodes= {5: 193930, 6: 4848, 4: 39700, 2: 13832, 3: 9482,
#              1: 13364, 7: 53}
n_nodes= {3: 1, 4: 1,5:1,6:1}

class DistributionNodes:
    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


# n_nodes = {26: 4711, 31: 3365, 19: 3093, 22: 3344, 32: 3333, 25: 4533,
#            36: 1388, 23: 4375, 33: 2686, 29: 3242, 14: 2469, 28: 4838,
#            41: 630, 9: 1858, 18: 2621, 27: 5417, 10: 2865, 30: 3605,
#            42: 502, 13: 2164, 11: 3051, 21: 4493, 15: 2292, 12: 2900,
#            40: 691, 45: 184, 20: 4883, 24: 3716, 46: 213, 39: 752,
#            17: 2446, 16: 3094, 35: 1879, 38: 915, 44: 691, 43: 360,
#            50: 37, 8: 1041, 7: 655, 34: 2168, 47: 119, 49: 73, 6: 705,
#            37: 928, 51: 21, 4: 45, 48: 187, 5: 111, 52: 42, 54: 93,
#            56: 12, 57: 8, 55: 35, 71: 1, 61: 9, 58: 18, 59: 5, 67: 28,
#            3: 4, 65: 2, 63: 5, 62: 1, 86: 1, 66: 20, 106: 2, 53: 3, 77: 1, 68: 1, 98: 1}

nodes_dist = DistributionNodes(n_nodes)


def construct_dataset(num_sample, batch_size):
    data_list = []
    for n in tqdm(range(int(num_sample / batch_size))):
        datas = []
        nodesxsample = nodes_dist.sample(batch_size).tolist()
        for n_particles in nodesxsample:
            atom_type = torch.randn(n_particles, 8)
            pos = torch.randn(n_particles, 3)

            coors = pos
            rows, cols = [], []
            for i in range(n_particles):
                for j in range(i + 1, n_particles):
                    rows.append(i)
                    cols.append(j)
                    rows.append(j)
                    cols.append(i)
            # print(n_particles)
            rows = torch.LongTensor(rows).unsqueeze(0)
            cols = torch.LongTensor(cols).unsqueeze(0)
            # print(rows.size())
            adj = torch.cat([rows, cols], dim=0)

            data = Data(x=atom_type, edge_index=adj, pos=pos)
            datas.append(data)
        data_list.append(datas)
    return data_list


def construct_dataset_pocket(num_sample, batch_size, dataset_info, num_points=None, num_for_pdb=None, start_linker=None,ligand_data=None,
                             *protein_information):
    
    if start_linker is None:
        nodes_dist = DistributionNodes(dataset_info['n_nodes'])
    else:
        nodes_dist = DistributionNodes(n_nodes)
    data_list = []

    num_atom = len(dataset_info['atom_decoder'])  ## +1 if charge
    # print('num_atom',num_atom)
    # num_atom = 20
    nodesxsample_list = []
    protein_atom_feature, protein_atom_feature_full, protein_pos, protein_bond_index = protein_information
    num_node_frag = 0
    if start_linker is not None:
        print('linker atom number:', len(start_linker['element']))
        num_node_frag = len(start_linker['element'])
    if batch_size>num_sample:
        batch_size=num_sample
    for n in tqdm(range(int(num_sample // batch_size))):
        datas = []
        if ligand_data != None:
            ligand_atom_feature, ligand_atom_feature_full, ligand_pos, ligand_bond_index, ligand_bond_type, ligand_edge_index, ligand_edge_type = ligand_data
            num_node = torch.tensor([ligand_atom_feature.size(0)])
            # pos = torch.rand_like(ligand_pos)
            data = ProteinLigandData(ligand_atom_feature=ligand_atom_feature, 
                                     ligand_atom_feature_full=ligand_atom_feature_full,
                            ligand_num_node=num_node, ligand_pos=ligand_pos,
                            # ligand_bond_index=ligand_bond_index, ligand_bond_type = ligand_bond_type,
                            ligand_bond_index = ligand_edge_index, ligand_bond_type = ligand_edge_type,
                            protein_atom_feature=protein_atom_feature,
                            protein_atom_feature_full=protein_atom_feature_full, protein_pos=protein_pos,
                            protein_bond_index=protein_bond_index)
            datas.extend([data for i in range(batch_size)])
        else:
            if num_points is not None:
                if batch_size >= 100:
                    nodesxsample = nodes_dist.sample(batch_size - 1).tolist()
                    nodesxsample.append(num_points)
                else:
                    nodesxsample = nodes_dist.sample(batch_size).tolist()
            else:
                nodesxsample = nodes_dist.sample(batch_size).tolist()
            nodesxsample_list.append(nodesxsample)
            # atom_type_list = torch.randn(batch_size,max_nodes,6)
            # pos_list = torch.randn(batch_size,max_nodes,3)
            for i, n_particles in enumerate(nodesxsample):
                if num_for_pdb is not None:
                    n_particles = num_for_pdb
                atom_type = torch.randn(n_particles, num_atom)
                atom_feature = torch.randn(n_particles, 8)
                atom_feature_full = torch.cat([atom_type, atom_feature], dim=1)
                pos = torch.randn(n_particles, 3)
                if start_linker is not None:
                    atom_type_linker = torch.cat([start_linker['linker_atom_type'], atom_type])
                    atom_feature_linker = torch.cat([start_linker['atom_feature'], atom_feature])
                    atom_feature_full_linker = torch.cat([atom_type_linker, atom_feature_linker], dim=1)
                    frag_mask = torch.cat(
                        [torch.ones(num_node_frag, dtype=torch.long), torch.zeros(n_particles, dtype=torch.long)])
                    pos = torch.cat([start_linker['pos'], pos])
                # atom_type = torch.zeros(n_particles, num_atom)
                # pos = torch.zeros(n_particles, 3)
                rows, cols = [], []
                adj = get_adj_matrix(n_particles + num_node_frag)

                num_node = torch.tensor([n_particles + num_node_frag])
                ligand_bond_type = torch.ones(adj.size(1), dtype=torch.long) * 2
                if start_linker is not None:
                    data = ProteinLigandData(ligand_atom_feature=atom_type_linker,
                                            ligand_atom_feature_full=atom_feature_full_linker,
                                            ligand_num_node=num_node, ligand_bond_index=adj, ligand_pos=pos,
                                            frag_mask= frag_mask,ligand_bond_type = ligand_bond_type,
                                            protein_atom_feature_full=protein_atom_feature_full, 
                                            protein_atom_feature = protein_atom_feature,
                                            protein_pos=protein_pos,
                                            protein_bond_index=protein_bond_index)
                else:
                    data = ProteinLigandData(ligand_atom_feature=atom_type, ligand_atom_feature_full=atom_feature_full,
                                            ligand_num_node=num_node, ligand_bond_index=adj, ligand_pos=pos,
                                            ligand_bond_type = ligand_bond_type,
                                            protein_atom_feature=protein_atom_feature,
                                            protein_atom_feature_full=protein_atom_feature_full, protein_pos=protein_pos,
                                            protein_bond_index=protein_bond_index)
                datas.append(data)
        data_list.append(datas)
    return data_list, nodesxsample_list

def construct_dataset_pocket_mask(num_sample, batch_size, dataset_info, num_points=None, num_for_pdb=None, start_linker=None,
                             given_edge=None, bond_type = None, *protein_information):
    
    if start_linker is None:
        nodes_dist = DistributionNodes(dataset_info['n_nodes'])
    else:
        nodes_dist = DistributionNodes(n_nodes)
    data_list = []

    num_atom = len(dataset_info['atom_decoder'])  ## +1 if charge
    # print('num_atom',num_atom)
    # num_atom = 20
    nodesxsample_list = []
    protein_atom_feature, protein_atom_feature_full, protein_pos, protein_bond_index = protein_information
    num_node_frag = 0
    if start_linker is not None:
        print(len(start_linker['element']))
        num_node_frag = len(start_linker['element'])
    if batch_size>num_sample:
        batch_size=num_sample
    for n in tqdm(range(int(num_sample // batch_size))):
        datas = []
        if num_points is not None:
            if batch_size >= 100:
                nodesxsample = nodes_dist.sample(batch_size - 1).tolist()
                nodesxsample.append(num_points)
            else:
                nodesxsample = nodes_dist.sample(batch_size).tolist()
        else:
            nodesxsample = nodes_dist.sample(batch_size).tolist()
        nodesxsample_list.append(nodesxsample)
        # atom_type_list = torch.randn(batch_size,max_nodes,6)
        # pos_list = torch.randn(batch_size,max_nodes,3)
        for i, n_particles in enumerate(nodesxsample):
            if num_for_pdb is not None:
                n_particles = num_for_pdb
            atom_type = torch.randn(n_particles, num_atom)
            atom_feature = torch.randn(n_particles, 8)
            atom_feature_full = torch.cat([atom_type, atom_feature], dim=1)
            # pos = torch.randn(n_particles, 3) #comment if only atom
            if start_linker is not None:
                atom_type_frag = torch.cat([start_linker['linker_atom_type'], atom_type])
                atom_feature_frag = torch.cat([start_linker['atom_feature'], atom_feature])
                atom_feature_full_frag = torch.cat([atom_type_frag, atom_feature_frag], dim=1)
                frag_mask = torch.cat(
                    [torch.ones(num_node_frag, dtype=torch.long), torch.zeros(n_particles, dtype=torch.long)])
                # pos = torch.cat([start_linker['pos'], pos])
                pos = start_linker['pos'] #only atom
            # atom_type = torch.zeros(n_particles, num_atom)
            # pos = torch.zeros(n_particles, 3)
            rows, cols = [], []
            if given_edge is None:
                adj = get_adj_matrix(n_particles + num_node_frag)
            else: adj = given_edge

            num_node = torch.tensor([n_particles + num_node_frag])
            if start_linker is not None:
                data = ProteinLigandData(ligand_atom_feature=atom_type_frag,
                                         ligand_atom_feature_full=atom_feature_full_frag,
                                         ligand_num_node=num_node, ligand_bond_index=adj, ligand_bond_type = bond_type, ligand_pos=pos,
                                         frag_mask=frag_mask,
                                         protein_atom_feature_full=protein_atom_feature_full, protein_pos=protein_pos,
                                         protein_bond_index=protein_bond_index)
            else:
                data = ProteinLigandData(ligand_atom_feature=atom_type, ligand_atom_feature_full=atom_feature_full,
                                         ligand_num_node=num_node, ligand_bond_index=adj, ligand_bond_type = bond_type, ligand_pos=pos,
                                         protein_atom_feature=protein_atom_feature,
                                         protein_atom_feature_full=protein_atom_feature_full, protein_pos=protein_pos,
                                         protein_bond_index=protein_bond_index)
            datas.append(data)
        data_list.append(datas)
    return data_list, nodesxsample_list

def construct_dataset_pocket_mask_fix(num_sample, batch_size, dataset_info, num_points, num_for_pdb, start_linker, frag_index, linker_index,
                             given_edge=None, bond_type = None, *protein_information):
    

    nodes_dist = DistributionNodes(dataset_info['n_nodes'])
    data_list = []

    num_atom = len(dataset_info['atom_decoder'])  ## +1 if charge

    nodesxsample_list = []
    protein_atom_feature, protein_atom_feature_full, protein_pos, protein_bond_index = protein_information
    num_node_frag = 0
    
    print(len(start_linker['element']))
    num_node_frag = len(start_linker['element'])-num_for_pdb

    if batch_size>num_sample:
        batch_size=num_sample
    for n in tqdm(range(int(num_sample // batch_size))):
        datas = []
        if num_points is not None:
            if batch_size >= 100:
                nodesxsample = nodes_dist.sample(batch_size - 1).tolist()
                nodesxsample.append(num_points)
            else:
                nodesxsample = nodes_dist.sample(batch_size).tolist()
        else:
            nodesxsample = nodes_dist.sample(batch_size).tolist()
        nodesxsample_list.append(nodesxsample)
        # atom_type_list = torch.randn(batch_size,max_nodes,6)
        # pos_list = torch.randn(batch_size,max_nodes,3)
        for i, n_particles in enumerate(nodesxsample):
            if num_for_pdb is not None:
                n_particles = num_for_pdb
            atom_type = torch.randn(n_particles, num_atom)
            atom_feature = torch.randn(n_particles, 8)
            atom_feature_full = torch.cat([atom_type, atom_feature], dim=1)
            pos = torch.randn(n_particles, 3) #comment if only atom
            

            # atom_type_frag = torch.cat([start_linker['linker_atom_type'], atom_type])
            atom_type_frag = torch.cat([torch.index_select(start_linker['linker_atom_type'], 0, frag_index),torch.index_select(start_linker['linker_atom_type'], 0, linker_index)]).float()
            # atom_feature_frag = torch.cat([start_linker['atom_feature'], atom_feature])
            atom_feature_frag = torch.cat([torch.index_select(start_linker['atom_feature'], 0, frag_index),torch.index_select(start_linker['atom_feature'], 0, linker_index)]).float()
            atom_feature_full_frag = torch.cat([atom_type_frag, atom_feature_frag], dim=1)
            
            pos = torch.cat([torch.index_select(start_linker['pos'], 0, frag_index), pos])
            # pos = torch.cat([torch.index_select(start_linker['pos'], 0, frag_index),torch.index_select(start_linker['pos'], 0, linker_index)])
            frag_mask = torch.cat(
                [torch.ones(num_node_frag, dtype=torch.long), torch.zeros(n_particles, dtype=torch.long)])
            rows, cols = [], []


            if given_edge is None:
                adj = get_adj_matrix(n_particles + num_node_frag)
            else: adj = given_edge

            num_node = torch.tensor([n_particles + num_node_frag])
            
            data = ProteinLigandData(ligand_atom_feature=atom_type_frag,
                                         ligand_atom_feature_full=atom_feature_full_frag,
                                         ligand_num_node=num_node, ligand_bond_index=adj, ligand_bond_type = bond_type, ligand_pos=pos,
                                         frag_mask=frag_mask,
                                         protein_atom_feature_full=protein_atom_feature_full, protein_pos=protein_pos,
                                         protein_bond_index=protein_bond_index)
            datas.append(data)
        data_list.append(datas)
    return data_list, nodesxsample_list