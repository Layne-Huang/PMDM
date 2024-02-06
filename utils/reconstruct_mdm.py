import tempfile
import warnings

import numpy as np
import openbabel
import torch
from rdkit import Chem
from rdkit import Geometry
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, 'w') as f:
        f.write(out)


def write_sdf_file(sdf_path, molecules):
    # NOTE Changed to be compatitble with more versions of rdkit
    # with Chem.SDWriter(str(sdf_path)) as w:
    #    for mol in molecules:
    #        w.write(mol)

    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if m is not None:
            w.write(m)

    print(f'Wrote SDF file to {sdf_path}')


class MolReconsError(Exception):
    pass


atom_decoder = ['H', 'C', 'N', 'O', 'F']
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161, 'Se': 147},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214, 'Se': 197},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H': 119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243},
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152},
          'Se': {'H': 147, 'C': 197, 'Se': 232}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160, 'Se': 215},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186},
          'Se': {'Se': 215}}

bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}

# margin1, margin2, margin3 = 5, 2, 1
margin1, margin2, margin3 = 5, 3, 1


# margin1, margin2, margin3 = 10, 5, 3
# margin1, margin2, margin3 = 0, 0, 0
# margin1, margin2, margin3 = 10, 5, 2
def geom_predictor(p, l, margin1=5, limit_bonds_to_one=False):
    """ p: atom pair (couple of str)
        l: bond length (float)"""
    bond_order = get_bond_order(p[0], p[1], l, check_exists=True)

    # If limit_bonds_to_one is enabled, every bond type will return 1.
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    else:
        return bond_order


def num_confs(num: str):
    if num.endswith('x'):
        return lambda x: x * int(num[:-1])
    elif int(num) > 0:
        return lambda x: int(num)
    else:
        raise ValueError()


def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(positions, atom_types, dataset_info, raise_error=True):
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()
    n_atoms = positions.size(0)
    rd_conf = Chem.Conformer(n_atoms)
    xyz = positions.clone().tolist()
    for i, atom in enumerate(X):
        a = Chem.Atom(dataset_info['atom_decoder'][atom.item()])
        mol.AddAtom(a)
        rd_coords = Geometry.Point3D(*xyz[i])
        rd_conf.SetAtomPosition(i, rd_coords)
    mol.AddConformer(rd_conf)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
    # modify
    try:
        mol = modify_submol(mol)
    except:
        if raise_error:
            raise MolReconsError()
        else:
            print('MolReconsError')
    return mol


def build_xae_molecule(positions, atom_types, dataset_info):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    # atom_decoder = dataset_info['atom_decoder']
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    atom_decoder = dataset_info['atom_decoder']
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info[
                'name'] == 'qm9_first_half':
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'geom' or dataset_info['name'] == 'crossdock':
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j],
                                       limit_bonds_to_one=False)
                # order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])

            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E


def modify_submol(mol):  # modify mols containing C=N(C)O
    submol = Chem.MolFromSmiles('C=N(C)O', sanitize=False)
    sub_fragments = mol.GetSubstructMatches(submol)
    for fragment in sub_fragments:
        atomic_nums = np.array([mol.GetAtomWithIdx(atom).GetAtomicNum() for atom in fragment])
        idx_atom_N = fragment[np.where(atomic_nums == 7)[0][0]]
        idx_atom_O = fragment[np.where(atomic_nums == 8)[0][0]]
        mol.GetAtomWithIdx(idx_atom_N).SetFormalCharge(1)  # set N to N+
        mol.GetAtomWithIdx(idx_atom_O).SetFormalCharge(-1)  # set O to O-
    return mol


def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = copy.deepcopy(rdkit_mol)
    set_rdmol_positions_(mol, pos)
    return mol


def set_positions(rd_mol, pos):
    n_atoms = len(rd_mol.GetAtoms())
    rd_conf = Chem.Conformer(n_atoms)
    xyz = pos
    # add atoms and coordinates
    for i in range(n_atoms):
        rd_coords = Geometry.Point3D(*xyz[i])
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)
    return rd_mol


def uff_relax(mol, max_iter=200):
    """
    Uses RDKit's universal force field (UFF) implementation to optimize a
    molecule.
    """
    more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
    if more_iterations_required:
        warnings.warn(f'Maximum number of FF iterations reached. '
                      f'Returning molecule after {max_iter} relaxation steps.')
    return more_iterations_required


def process_molecule(rdmol, add_hydrogens=False, sanitize=False, relax_iter=0,
                     largest_frag=False):
    """
    Apply filters to an RDKit molecule. Makes a copy first.
    Args:
        rdmol: rdkit molecule
        add_hydrogens
        sanitize
        relax_iter: maximum number of UFF optimization iterations
        largest_frag: filter out the largest fragment in a set of disjoint
            molecules
    Returns:
        RDKit molecule or None if it does not pass the filters
    """

    # Create a copy
    mol = Chem.Mol(rdmol)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            warnings.warn('Sanitization failed. Returning None.')
            return None

    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

    if largest_frag:
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        if sanitize:
            # sanitize the updated molecule
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                return None

    if relax_iter > 0:
        # if not UFFHasAllMoleculeParams(mol):
        #     warnings.warn('UFF parameters not available for all atoms. '
        #                   'Returning None.')
        #     return None

        try:
            uff_relax(mol, relax_iter)
            if sanitize:
                # sanitize the updated molecule
                Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError) as e:
            return None

    return mol


def make_mol_openbabel(positions, atom_types, dataset_info):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_types: N
        atom_decoder: maps indices to atom types
    Returns:
        rdkit molecule
    """
    atom_decoder = dataset_info['atom_decoder']
    atom_types = [atom_decoder[x] for x in atom_types]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name

        # Write xyz file
        write_xyz_file(positions, atom_types, tmp_file)

        # Convert to sdf file with openbabel
        # openbabel will add bonds
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)

        obConversion.WriteFile(ob_mol, tmp_file)

        # Read sdf file with RDKit
        mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

    return mol
