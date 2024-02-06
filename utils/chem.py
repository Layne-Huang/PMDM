from io import BytesIO

from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_scatter import scatter

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}


def rdmol_to_data(mol, smiles=None):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    if smiles is None:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    data.nx = to_networkx(data, to_undirected=True)

    return data


def generated_to_xyz(data):
    ptable = Chem.GetPeriodicTable()

    num_atoms = data.ligand_context_element.size(0)
    xyz = "%d\n\n" % (num_atoms,)
    for i in range(num_atoms):
        symb = ptable.GetElementSymbol(data.ligand_context_element[i].item())
        x, y, z = data.ligand_context_pos[i].clone().cpu().tolist()
        xyz += "%s %.8f %.8f %.8f\n" % (symb, x, y, z)

    return xyz


def generated_to_sdf(data):
    xyz = generated_to_xyz(data)
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "sdf")

    mol = openbabel.OBMol()
    obConversion.ReadString(mol, xyz)
    sdf = obConversion.WriteString(mol)
    return sdf


def sdf_to_rdmol(sdf):
    stream = BytesIO(sdf.encode())
    suppl = Chem.ForwardSDMolSupplier(stream)
    for mol in suppl:
        return mol
    return None


def generated_to_rdmol(data):
    sdf = generated_to_sdf(data)
    return sdf_to_rdmol(sdf)


def filter_rd_mol(rdmol):
    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]

    # 3-3 ring intersection
    for i, ring_a in enumerate(rings):
        if len(ring_a) != 3: continue
        for j, ring_b in enumerate(rings):
            if i <= j: continue
            inter = ring_a.intersection(ring_b)
            if (len(ring_b) == 3) and (len(inter) > 0):
                return False

    return True


import copy
import torch
from torchvision.transforms.functional import to_tensor

import rdkit
import rdkit.Chem.Draw
from rdkit import Chem
from rdkit.Chem import rdDepictor as DP
from rdkit.Chem import PeriodicTable as PT
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import Mol, GetPeriodicTable
from rdkit.Chem.Draw import rdMolDraw2D as MD2
from rdkit.Chem.rdmolops import RemoveHs
from typing import List, Tuple

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}


def set_conformer_positions(conf, pos):
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(i, pos[i].tolist())
    return conf


def draw_mol_image(rdkit_mol, tensor=False):
    rdkit_mol.UpdatePropertyCache()
    img = rdkit.Chem.Draw.MolToImage(rdkit_mol, kekulize=False)
    if tensor:
        return to_tensor(img)
    else:
        return img


def update_data_rdmol_positions(data):
    for i in range(data.pos.size(0)):
        data.rdmol.GetConformer(0).SetAtomPosition(i, data.pos[i].tolist())
    return data


def update_data_pos_from_rdmol(data):
    new_pos = torch.FloatTensor(data.rdmol.GetConformer(0).GetPositions()).to(data.pos)
    data.pos = new_pos
    return data


def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = copy.deepcopy(rdkit_mol)
    set_rdmol_positions_(mol, pos)
    return mol


def set_rdmol_positions_(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def get_atom_symbol(atomic_number):
    return PT.GetElementSymbol(GetPeriodicTable(), atomic_number)


def mol_to_smiles(mol: Mol) -> str:
    return Chem.MolToSmiles(mol, allHsExplicit=True)


def mol_to_smiles_without_Hs(mol: Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def remove_duplicate_mols(molecules: List[Mol]) -> List[Mol]:
    unique_tuples: List[Tuple[str, Mol]] = []

    for molecule in molecules:
        duplicate = False
        smiles = mol_to_smiles(molecule)
        for unique_smiles, _ in unique_tuples:
            if smiles == unique_smiles:
                duplicate = True
                break

        if not duplicate:
            unique_tuples.append((smiles, molecule))

    return [mol for smiles, mol in unique_tuples]


def get_atoms_in_ring(mol):
    atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        for a in ring:
            atoms.add(a)
    return atoms


def get_2D_mol(mol):
    mol = copy.deepcopy(mol)
    DP.Compute2DCoords(mol)
    return mol


def draw_mol_svg(mol, molSize=(450, 150), kekulize=False):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        DP.Compute2DCoords(mc)
    drawer = MD2.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    # return svg.replace('svg:','')
    return svg


def GetBestRMSD(probe, ref):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd
