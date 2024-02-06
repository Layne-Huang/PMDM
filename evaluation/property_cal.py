# import psi4
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


# from rdkit.Chem.Draw import IPythonConsole

def mol_output(mol_input):
    if isinstance(mol_input, str):
        return Chem.MolFromSmiles(mol_input)
    else:
        return mol_input


def mol2xyz(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    AllChem.UFFOptimizeMolecule(mol)
    atoms = mol.GetAtoms()
    string = "\n"
    for _, atom in enumerate(atoms):
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
    string += "units angstrom\n"
    return string, mol


def smile2homo_lumo(mol_input, outputfile):
    psi4.core.set_output_file(outputfile, True)
    mol = mol_output(mol_input)
    xyz, mol = mol2xyz(mol)
    psi4.set_memory('4 GB')
    psi4.set_num_threads(4)
    benz = psi4.geometry(xyz)
    scf_e, scf_wfn = psi4.energy("B3LYP/cc-pVDZ", return_wfn=True)

    HOMO = scf_wfn.epsilon_a_subset('AO', 'ALL').np[scf_wfn.nalpha() - 1]
    LUMO = scf_wfn.epsilon_a_subset('AO', 'ALL').np[scf_wfn.nalpha()]
    print(HOMO, LUMO, scf_e)
    return HOMO, LUMO, scf_e


def cal_Aromatic_ring(mol_input):
    mol = mol_output(mol_input)
    num_Aromatic_ring = rdMolDescriptors.CalcNumAromaticRings(mol)
    return num_Aromatic_ring


def cal_Aliphatic_ring(mol_input):
    mol = mol_output(mol_input)
    num_Aliphatic_ring = rdMolDescriptors.CalcNumAliphaticRings(mol)
    return num_Aliphatic_ring


def cal_num_ring(mol_input):
    mol = mol_output(mol_input)
    num_num_ring = rdMolDescriptors.CalcNumRings(mol)
    return num_num_ring


def get_atom_ring(mol_input):
    mol = mol_output(mol_input)
    rc = mol.GetRingInfo()
    atom_ring = rc.AtomRings()
    return atom_ring


def get_size_ring(mol_input):
    mol = mol_output(mol_input)
    ssr = Chem.GetSymmSSSR(mol)
    num_ring = len(ssr)
    ring_size_list = []
    atom_size_list = []
    for ring in ssr:
        ring_size_list.append(len(list(ring)))
        atom_size_list.append(list(ring))
    return num_ring, ring_size_list, atom_size_list


print(get_size_ring('c1ccccc1'))
print(cal_Aromatic_ring('c1ccccc1'))
print(cal_Aliphatic_ring('C1CC1'))
