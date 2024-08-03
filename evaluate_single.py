import argparse
import pickle
from statistics import mean

from joblib import Parallel, delayed
from rdkit.Chem.Descriptors import MolLogP, qed  # , MolLogP
from rdkit.Chem import rdmolops

from configs.dataset_config import get_dataset_info
from evaluation import *
from evaluation.docking import *
from evaluation.sascorer import *
from evaluation.score_func import *
# from rdkit.Chem import Draw
from evaluation.similarity import calculate_diversity
from evaluation.similarity import *
from utils.reconstruct import *
from utils.transforms import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str, required=True)
    parser.add_argument('--mol_path', type=str, required=True)
    parser.add_argument('--pdb_path', type=str, required=True)
    args = parser.parse_args()


    protein_filename = ''

    protein_root = os.path.dirname(args.pdb_path)


    mol = Chem.SDMolSupplier(args.mol_path)[0]
    # mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol)
    # # smiles = 'CC(C)NC(=O)O[C@@H]1CC[C@H](C2=NN=C(Nc3cnccn3)C2)C1'
    print(smiles)
    # mol = Chem.MolFromSmiles(smiles)
    ref_mol = Chem.SDMolSupplier(args.ref_path)[0]
    smiles = Chem.MolToSmiles(ref_mol)
    print(smiles)
    # ref_mol = Chem.MolFromSmiles(smiles)

    simi = tanimoto_sim(ref_mol, mol)
    print(simi)

    g_sa = compute_sa_score(mol)
    print("Generate SA score:", g_sa)
    r_sa = compute_sa_score(ref_mol)
    print("Ref SA score:", r_sa)

    g_qed = qed(mol)
    print("Generate QED score:", g_qed)
    r_qed = qed(ref_mol)
    print("Ref QED score:", r_qed)

    g_logP = MolLogP(mol)
    print("Generate logP:", g_logP)
    r_logP = MolLogP(ref_mol)
    print("Ref logP:", r_logP)

    g_Lipinski = obey_lipinski(mol)
    print("Generate Lipinski:", g_Lipinski)
    r_Lipinski = obey_lipinski(ref_mol)
    print("Ref Lipinski:", r_Lipinski)


    # exit()
    vina_task = QVinaDockingTask.from_generated_data(protein_filename, mol, protein_root=protein_root)
    g_vina_results = vina_task.run_sync()
    g_vina_score = g_vina_results[0]['affinity']
    print("Generate vina score:", g_vina_score)

# vina_task = QVinaDockingTask.from_generated_data(protein_filename, ref_mol, protein_root=protein_root)
# r_vina_results = vina_task.run_sync()
# r_vina_score = r_vina_results[0]['affinity']
# print("Ref vina score:", r_vina_score)