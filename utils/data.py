import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

FOLLOW_BATCH_SBDD = ['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']
FOLLOW_BATCH_DPM = ['protein_element', 'ligand_element']
FOLLOW_BATCH = ['ligand_atom_feature', 'protein_atom_feature_full']


class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if
                                                  instance.ligand_bond_index[0, k].item() == i] for i in
                                       instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            if 'ligand_element' in self.keys():
                return self['ligand_element'].size(0)
            return self['ligand_atom_feature'].size(0)
        if key == 'protein_bond_index':
            if 'protein_element' in self.keys():
                return self['protein_element'].size(0)
            return self['protein_atom_feature_full'].size(0)
        if key == 'pocket_bond_index':
            if 'pocket_element' in self.keys():
                return self['pocket_element'].size(0)
            return self['pocket_atom_type'].size(0)
        elif key == 'ligand_context_bond_index':
            return self['ligand_context_element'].size(0)
        else:
            return super().__inc__(key, value)


class ProteinLigandDataLoader(DataLoader):

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            follow_batch=['ligand_element', 'protein_element'],
            **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)


def batch_from_data_list(data_list):
    return Batch.from_data_list(data_list, follow_batch=['ligand_element', 'protein_element'])


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output
