import glob
import os
from rdkit import Chem
from rdkit.Chem import rdmolops
from oddt.toolkits.extras.rdkit import fixer
from openbabel import pybel
import argparse

'''
obtain pocket and ligand from the complex file
'''

def split_pocket_ligand(path,cutoff=20):
    root = os.path.dirname(path)
    pdb_code = os.path.basename(path)[:4]+f'cut{cutoff}'
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/3UG2.pdb', help='path to the complex pdb file')
    # path = 'data/3GU2.pdb'
    args = parser.parse_args()
    split_pocket_ligand(args.path,cutoff=10)


