import glob
import os
from rdkit import Chem
from rdkit.Chem import rdmolops
from oddt.toolkits.extras.rdkit import fixer
from openbabel import pybel

'''
obtain pocket and ligand from the complex file
'''

def split_pocket_ligand(path):
    root = os.path.dirname(path)
    pdb_code = os.path.basename(path)[:4]+'cut20'
    root = os.path.join(root,pdb_code)
    os.makedirs(root, exist_ok=True)
    complex_ = Chem.MolFromPDBFile(path, sanitize=False)
    try:
        pocket, ligand = fixer.ExtractPocketAndLigand(complex_, cutoff=20)
        #### write pocket flie
        inter_pdb_file = f"{root}/{pdb_code}_pocket_withH.pdb"
        Chem.MolToPDBFile(pocket, inter_pdb_file)
        # remove hydrogen and protonation
        os.system(f"obabel {inter_pdb_file} -d -O {root}/{pdb_code}_pocket.pdb")
        # if do protonation, the hydrogen is remained
        os.unlink(inter_pdb_file)
        #### write ligand no double bond
        # with Chem.SDWriter(f"{ligand_root}/{pdb_code}_ligand.sdf") as f:
        #     f.write(ligand)
        #### write ligand pdb flie
        Chem.MolToPDBFile(ligand, f"{root}/{pdb_code}_ligand.pdb")
        mol = next(pybel.readfile('pdb', f"{root}/{pdb_code}_ligand.pdb"))
        mol.write('sdf', f"{root}/{pdb_code}_ligand.sdf", overwrite=True)
        i += 1
        if i%100 == 0:
            print('===== writing %d-th samples =====' % i)

    except:
        print('error when loading data')

if __name__ == '__main__':
    path = './data/7l11.pdb'
    split_pocket_ligand(path)
    # ligand_path = '/./data/crossdocked_pocket10/DYR_STAAU_2_158_0/4xe6_X_rec_3fqc_55v_lig_tt_docked_4.sdf'
    # ligand_path =  './data/crossdocked_pocket10/LAT_MYCTU_1_449_0/2jjg_A_rec_2jjg_plp_lig_tt_min_0.sdf'
    # suppl = Chem.SDMolSupplier(ligand_path)
    # mols = [Chem.MolToSmiles(mol) for mol in suppl if mol]
    # print(mols)
    # atom_num = suppl[0].GetNumAtoms()
    # print(atom_num)

