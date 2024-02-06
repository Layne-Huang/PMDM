import os
import re
import torch
from pathlib import Path
import argparse

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import time


def calculate_smina_score(pdb_file, sdf_file):
    # add '-o <name>_smina.sdf' if you want to see the output
    out = os.popen(f'smina.static -l {sdf_file} -r {pdb_file} '
                   f'--score_only').read()
    matches = re.findall(
        r"Affinity:[ ]+([+-]?[0-9]*[.]?[0-9]+)[ ]+\(kcal/mol\)", out)
    return [float(x) for x in matches]


# def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
#     os.popen(f'obabel {sdf_file} -O {pdbqt_outfile} '
#              f'-f {mol_id + 1} -l {mol_id + 1}').read()
#     return pdbqt_outfile

def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(f'/home/leihuang/miniconda3/envs/mol/bin/obabel {sdf_file} -O {pdbqt_outfile} ').read()
    return pdbqt_outfile


def calculate_qvina2_score(receptor_file, sdf_file, out_dir, size=20,
                           exhaustiveness=16, return_rdmol=False, index=None):


    receptor_file = Path(receptor_file)
    receptor_name = os.path.basename(receptor_file)[:4]
    

    if receptor_file.suffix == '.pdb':
        # prepare receptor, requires Python 2.7
        receptor_pdbqt_file = Path('./data/test_data/test_pdbqt_1k', receptor_file.stem + '.pdbqt')
        os.popen(f'./miniconda/envs/adt/bin/prepare_receptor4.py -r {receptor_file} -O {receptor_pdbqt_file}')
    else:
        receptor_pdbqt_file = receptor_file

    scores = []
    rdmols = []  # for if return rdmols
    pdb_flag = False
    if type(sdf_file) == str:
        sdf_file = Path(sdf_file)
        suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
        pdb_flag = True
    else:
        suppl = [sdf_file]
        ligand_name = f'{receptor_name}_{index}'
        os.makedirs(os.path.join(out_dir,receptor_name), exist_ok=True)
        ligand_file = os.path.join(out_dir,receptor_name, ligand_name + '.sdf')
        if not Path(ligand_file).exists() or Path(ligand_file).stat().st_size== 0:
            sdf_writer = Chem.SDWriter(ligand_file)
            sdf_writer.write(sdf_file)
            sdf_writer.close()
        sdf_file = ligand_file

    for i, mol in enumerate(suppl):  # sdf file may contain several ligands
        if index is not None:
            i = index
        ligand_name = f'{receptor_name}_{i}'
        ligand_name = os.path.basename(sdf_file)
        ligand_name = os.path.basename(sdf_file).split('.sdf')[0]
        ligand_name = f'{ligand_name}_{i}'
        
        # prepare ligand
        if pdb_flag:
            ligand_pdbqt_file = Path(out_dir, ligand_name + '.pdbqt')
            out_sdf_file = Path(out_dir, ligand_name + '_out.sdf')
            out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
        else:
            ligand_pdbqt_file = Path(out_dir, receptor_name, ligand_name + '.pdbqt')
            out_sdf_file = Path(out_dir, receptor_name, ligand_name + '_out.sdf')
            out_pdbqt_file = Path(out_dir, receptor_name, ligand_name + '_out.pdbqt')

        # out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
        # you have to assdign your own mol envrionment
        if out_pdbqt_file.exists() and not out_sdf_file.exists():
            os.popen(f'./miniconda3/envs/mol/bin/obabel {out_pdbqt_file} -O {out_sdf_file}').read()
        if out_sdf_file.exists() and out_sdf_file.stat().st_size != 0:
            print(out_sdf_file)
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
            out = os.popen(
                f'./qvina2.1 --receptor {receptor_pdbqt_file} '
                f'--ligand {ligand_pdbqt_file} '
                f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
                f'--size_x {size} --size_y {size} --size_z {size} '
                f'--exhaustiveness {exhaustiveness}'
            ).read()

            out_split = out.splitlines()
            # print(out_split)
            best_idx = out_split.index('-----+------------+----------+----------') + 1
            best_line = out_split[best_idx].split()
            assert best_line[0] == '1'
            scores.append(float(best_line[1]))

            
            if out_pdbqt_file.exists():
                os.popen(f'/home/leihuang/miniconda3/envs/mol/bin/obabel {out_pdbqt_file} -O {out_sdf_file}').read()

        if return_rdmol:
            rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
            rdmols.append(rdmol)

    if return_rdmol:
        return scores, rdmols
    else:
        return scores


if __name__ == '__main__':
    start_t = time.time()
    parser = argparse.ArgumentParser('QuickVina evaluation')
    parser.add_argument('--pdbqt_dir', type=Path,
                        help='Receptor files in pdbqt format')
    parser.add_argument('--sdf_dir', type=Path, default=None,
                        help='Ligand files in sdf format')
    parser.add_argument('--sdf_files', type=Path, nargs='+', default=None)
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--write_csv', action='store_true')
    parser.add_argument('--write_dict', action='store_true')
    parser.add_argument('--dataset', type=str, default='crossdocked')
    args = parser.parse_args()
    # args.pdbqt_dir = './data/8h6pcut6/'
    # args.write_csv = True
    # args.out_dir = './data/8h6pcut6/select_docked'
    # args.sdf_dir = Path('./DiffDock/results/user_inference/8h6p/')
    assert (args.sdf_dir is not None) ^ (args.sdf_files is not None)

    results = {'receptor': [], 'ligand': [], 'scores': []}
    results_dict = {}
    # sdf_files = list(args.sdf_dir.glob('[!.]*.sdf')) \
    #     if args.sdf_dir is not None else args.sdf_files
    sdf_files = list(args.sdf_dir.glob('*/rank1.sdf')) \
        if args.sdf_dir is not None else args.sdf_files
    pbar = tqdm(sdf_files)
    for sdf_file in pbar:
        pbar.set_description(f'Processing {sdf_file.name}')

        if args.dataset == 'moad':
            """
            Ligand file names should be of the following form:
            <receptor-name>_<pocket-id>_<some-suffix>.sdf
            where <receptor-name> and <pocket-id> cannot contain any 
            underscores, e.g.: 1abc-bio1_pocket0_gen.sdf
            """
            ligand_name = sdf_file.stem
            receptor_name, pocket_id, *suffix = ligand_name.split('_')
            suffix = '_'.join(suffix)
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')
        elif args.dataset == 'crossdocked':
            ligand_name = sdf_file.stem
            # receptor_name = ligand_name[:-4]
            start = ligand_name.find('_') + 1 # 找到第一个_的位置并加一
            end = ligand_name.find('_gen', start) # 从start位置开始找到第一个_gen的位置
            receptor_name = ligand_name[start:end]
            receptor_name = '8h6pcut6_pocket'
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')
            receptor_name = str(sdf_file).split('/')[-2]

        try:
            sdf_file = str(sdf_file)
            scores, rdmols = calculate_qvina2_score(
                receptor_file, sdf_file, args.out_dir, return_rdmol=True, index=receptor_name)
        except (ValueError, AttributeError) as e:
            print(e)
            continue
        results['receptor'].append(str(receptor_file))
        results['ligand'].append(str(sdf_file))
        results['scores'].append(scores)

        if args.write_dict:
            results_dict[receptor_name] = [scores, rdmols]

    if args.write_csv:
        df = pd.DataFrame.from_dict(results)
        df.to_csv(Path(args.out_dir, 'qvina2_scores.csv'))

    if args.write_dict:
        torch.save(results_dict, Path(args.out_dir, 'qvina2_scores.pt'))
    
    end_t = time.time()
    print('Time:',end_t-start_t)