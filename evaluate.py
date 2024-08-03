import argparse
import pickle
from statistics import mean
from pathlib import Path

from joblib import Parallel, delayed
from rdkit.Chem.Descriptors import MolLogP, qed  # , MolLogP

from configs.dataset_config import get_dataset_info
from evaluation import *
from evaluation.docking import *
from evaluation.docking_2 import *
from evaluation.sascorer import *
from evaluation.score_func import *
# from rdkit.Chem import Draw
from evaluation.similarity import calculate_diversity
from utils.reconstruct import *
from utils.transforms import *

def evaluate(m,n):
    smile = m['smile']
    protein_filename = m['protein_file']
    ligand_filename = m['ligand_file']

    # mol_id = hash(smile+protein_filename)

    # if mol_id in results:
    #     print(f'Skipping {smile}, already computed.')
    #     return results[mol_id]
    mol_id = n

    mol = m['mol']

    try:
        _, g_sa = compute_sa_score(mol)
        print("Generate SA score:", g_sa)

        g_qed = qed(mol)
        print("Generate QED score:", g_qed)

        g_logP = MolLogP(mol)
        print("Generate logP:", g_logP)

        g_Lipinski = obey_lipinski(mol)
        print("Generate Lipinski:", g_Lipinski)
    except:
        print('mol error')
        return None

    # try:
        # vina_task = QVinaDockingTask.from_generated_data(protein_filename, mol, protein_root=protein_root)
        # g_vina_results = vina_task.run_sync()
        # g_vina_score = g_vina_results[0]['affinity']

    receptor_file = os.path.basename(protein_filename).replace('.pdb','')+'.pdbqt'
    # receptor_file = protein_filename
    receptor_file = Path(os.path.join(protein_root,receptor_file))
    index = n%100
    g_vina_score = calculate_qvina2_score(
            receptor_file, mol, out_dir, return_rdmol=False, index=index)[0]
    print("Generate vina score:", g_vina_score)

    # except:
    #     print('Vina error: TypeError: NoneType object is not subscriptable')
    #     return None

    # rd_vina_score = test_vina_score_list[protein_filename]

    g_high_affinity = 0
    # if float(g_vina_score) < float(rd_vina_score):
    #     g_high_affinity = 1
    #     high_affinity.append(1)
    metrics = {'SA': g_sa, 'QED': g_qed, 'logP': g_logP, 'Lipinski': g_Lipinski, 'vina': g_vina_score,
               'high_affinity': g_high_affinity}
    result = {
        'smile': smile,
        'protein_file': protein_filename,
        'ligand_file': ligand_filename,
        'mol': mol,
        'metrics': metrics}
    # results_dict[mol_id] = result
    # with open(save_mol_result_path, 'wb') as f:
    #     pickle.dump(results, f)

    return result


def save_sdf(mol, sdf_dir, gen_file_name):
    writer = Chem.SDWriter(os.path.join(sdf_dir, gen_file_name))
    writer.write(mol, confId=0)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='crossdock')
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()

    dataset_info = get_dataset_info(args.dataset, False)
    path = args.path
    print(os.path.dirname(path))
    save_mol_result_path = os.path.join(os.path.dirname(path), 'mol_results.pkl')
    if os.path.exists(save_mol_result_path):
        with open(save_mol_result_path, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if args.dataset == 'crossdock':
        # protein_root = './data/test_data_1k/test_pdbqt'
        protein_root ='./data/crossdocked_pocket10'

    out_dir = os.path.join(os.path.dirname(path),'ligand')
    os.makedirs(out_dir,exist_ok=True)
    sdf_dir = os.path.dirname(path)
    results_mol = []
    high_affinity = []
    stable = 0
    valid = 0
    smile_list = []
    num_samples = 0
    position_list = []
    atom_type_list = []
    sa_list = []
    qed_list = []
    logP_list = []
    Lipinski_list = []
    vina_score_list = []
    diversity_list = []
    mol_dict = {}
    idx = 0
    t_vina_dict = {}

    with open('test_vina_{}_dict.pkl'.format(args.dataset), 'rb') as f:
        test_vina_score_list = pickle.load(f)

    for d in tqdm(data):
        mol = d['mol']
        protein_filename = d['protein_file']
        if protein_filename not in mol_dict.keys():
            mol_dict[protein_filename] = []
        mol_dict[protein_filename].append(mol)

    # for n, key in enumerate(tqdm(mol_dict)):
    #     if len(mol_dict[key]) != 100:
    #         print(key + ' generated mol: %d' % (len(mol_dict[key])))
    #     diversity_list.append(calculate_diversity(mol_dict[key]))

    # diversity_list = torch.tensor(diversity_list)
    # print(f"Diversity: {diversity_list.mean():.3f} \pm "
    #       f"{diversity_list.std(unbiased=False):.2f}")

    results = Parallel(n_jobs=-1)(delayed(evaluate)(m,n) for n, m in enumerate(tqdm(data)))
    # results = [evaluate(m,n) for n, m in enumerate(tqdm(data))]
    # results = []
    # for m in data:
    #     results.append(evaluate(m))
    for result in tqdm(results):
        if result is not None:
            results_mol.append(result)
            metrics = result['metrics']
            g_sa, g_qed, g_logP, g_Lipinski, g_vina,g_h_a = metrics['SA'], metrics['QED'], metrics['logP'], metrics[
                'Lipinski'], metrics['vina'], metrics['high_affinity']
            # if g_vina<-6.5:
            #     save_sdf(result['mol'],sdf_dir,str(g_vina)+'.sdf')
            sa_list.append(g_sa)
            qed_list.append(g_qed)
            logP_list.append(g_logP)
            Lipinski_list.append(g_Lipinski)
            high_affinity.append(g_h_a)
            valid+=1
            if g_vina < 0:
                vina_score_list.append(g_vina)
            

    num_samples = 2500
    # validity_dict = analyze_stability_for_molecules(position_list, atom_type_list, dataset_info)
    # print(validity_dict)

    print("Final validity:", valid / num_samples)
    print("Final stable:", stable / num_samples)
    # print(f"Time per pocket: {times_arr.mean():.3f} \pm "
    #         f"{times_arr.std(unbiased=False):.2f}")
    print('mean sa:%f' % mean(sa_list))
    print('mean qed:%f' % mean(qed_list))
    print('mean logP:%f' % mean(logP_list))
    print('mean Lipinski:%f' % np.mean(Lipinski_list))
    print('mean vina:%f' % mean(vina_score_list))
    print('high affinity:%d' % np.sum(high_affinity))

    # print(vina_score_list)

    sa_list = torch.tensor(sa_list)
    qed_list = torch.tensor(qed_list)
    logP_list = torch.tensor(logP_list)
    Lipinski_list = torch.tensor(Lipinski_list)
    vina_score_list = torch.tensor(vina_score_list)
    metrics_list = {
        'diversity': diversity_list,
        'sa': sa_list,
        'qed': qed_list,
        'logP': logP_list,
        'Lipinski': Lipinski_list,
        'vina': vina_score_list,
        'high_affinity': high_affinity}

    save_mol_result_path = os.path.join(os.path.dirname(path), 'mol_results.pkl')
    with open(save_mol_result_path, 'wb') as f:
        pickle.dump(results_mol, f)
        f.close()

    save_metric_result_path = os.path.join(os.path.dirname(path), 'metric_results.pkl')
    with open(save_metric_result_path, 'wb') as f:
        pickle.dump(metrics_list, f)
        f.close()
