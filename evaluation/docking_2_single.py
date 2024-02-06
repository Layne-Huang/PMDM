import os
import re
import torch
from pathlib import Path
import argparse

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from glob import glob
import numpy as np
from joblib import Parallel, delayed

def calculate_smina_score(pdb_file, sdf_file):
    # add '-o <name>_smina.sdf' if you want to see the output
    out = os.popen(f'smina.static -l {sdf_file} -r {pdb_file} '
                   f'--score_only').read()
    matches = re.findall(
        r"Affinity:[ ]+([+-]?[0-9]*[.]?[0-9]+)[ ]+\(kcal/mol\)", out)
    return [float(x) for x in matches]


# replace your own env path
def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(f'./miniconda/envs/mol/bin/obabel {sdf_file} -O {pdbqt_outfile} '
             f'-f {mol_id + 1} -l {mol_id + 1}').read()
    return pdbqt_outfile


def calculate_qvina2_score(receptor_file, sdf_file, out_dir, size=20,
                           exhaustiveness=16, return_rdmol=False):

    receptor_file = Path(receptor_file)
    sdf_file = Path(sdf_file)

    if receptor_file.suffix == '.pdb':
        # prepare receptor, requires Python 2.7
        receptor_pdbqt_file = Path(out_dir, receptor_file.stem + '.pdbqt')
        os.popen(f'prepare_receptor4.py -r {receptor_file} -O {receptor_pdbqt_file}')
    else:
        receptor_pdbqt_file = receptor_file

    scores = []
    rdmols = []  # for if return rdmols
    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
    pdb_name = receptor_file.stem
    for i, mol in enumerate(suppl):  # sdf file may contain several ligands
        ligand_name = f'{pdb_name}_{sdf_file.stem}_{i}'
        # prepare ligand
        ligand_pdbqt_file = Path(out_dir, ligand_name + '.pdbqt')
        out_sdf_file = Path(out_dir, ligand_name + '_out.sdf')

        if out_sdf_file.exists():
            with open(out_sdf_file, 'r') as f:
                scores.append(
                    min([float(x.split()[2]) for x in f.readlines()
                         if x.startswith(' VINA RESULT:')])
                )

        else:
            sdf_to_pdbqt(sdf_file, ligand_pdbqt_file, i)

            # center box at ligand's center of mass
            cx, cy, cz = mol.GetConformer().GetPositions().mean(0)

            # run QuickVina 2
            print(receptor_pdbqt_file, ligand_pdbqt_file)
            command = f'./qvina2.1 --receptor {receptor_pdbqt_file} --ligand {ligand_pdbqt_file} --center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} --size_x {size} --size_y {size} --size_z {size} --exhaustiveness {exhaustiveness}'
            print(command)
            out = os.popen(
                command
            ).read()

            # out = os.popen(
            #     f'./qvina2.1 --receptor {receptor_pdbqt_file} '
            #     f'--ligand {ligand_pdbqt_file} '
            #     f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
            #     f'--size_x {size} --size_y {size} --size_z {size} '
            #     f'--exhaustiveness {exhaustiveness}'
            # ).read()

            out_split = out.splitlines()
            best_idx = out_split.index('-----+------------+----------+----------') + 1
            best_line = out_split[best_idx].split()
            assert best_line[0] == '1'
            scores.append(float(best_line[1]))

            out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
            if out_pdbqt_file.exists():
                os.popen(f'obabel {out_pdbqt_file} -O {out_sdf_file}').read()

        if return_rdmol:
            rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
            rdmols.append(rdmol)

    if return_rdmol:
        return scores, rdmols
    else:
        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser('QuickVina evaluation')
    parser.add_argument('--receptor_file', type=Path,
                        help='Receptor files in pdbqt format')
    parser.add_argument('--sdf_file', type=Path, default=None,
                        help='Ligand files in sdf format')
    parser.add_argument('--out_dir', type=Path)
    args = parser.parse_args()

    # assert (args.sdf_dir is not None) ^ (args.sdf_files is not None)

    results = {'receptor': [], 'ligand': [], 'scores': []}
    results_dict = {}
    # sdf_files = list(args.sdf_dir.glob('[!.]*.sdf')) \
    #     if args.sdf_dir is not None else args.sdf_files
    # pbar = tqdm(sdf_files)
    # args.sdf_file = './data/8h6pcut6/select/-10.2_8h6p_16_revised.sdf'
    # args.out_dir = './data/8h6pcut6/docked/'
    # args.receptor_file = './data/8h6pcut6/8h6pcut6_pocket.pdbqt'
    data_list = []
    file_list = []



        # try:
    score = calculate_qvina2_score(
        args.receptor_file, args.sdf_file, args.out_dir, return_rdmol=False)
    
    print(score)

#./qvina2.1 --receptor ./data/8h6tcut6/8h6tcut6_pocket.pdbqt --ligand ./data/8h6tcut10/8h6tcut6_pocket_ligand_8h6t1_0.pdbqt  --center_x 35.5059 --center_y 22.1389 --center_z -7.6437 --size_x 20 --size_y 20 --size_z 20 --exhaustiveness 16

        


