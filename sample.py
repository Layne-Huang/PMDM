import argparse
import pickle
from statistics import mean

from rdkit.Chem.Descriptors import MolLogP, qed  # , MolLogP
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from configs.dataset_config import get_dataset_info
from evaluation.docking import *
from evaluation.docking_2 import *
from evaluation.sascorer import *
from evaluation.score_func import *
from evaluation.similarity import calculate_diversity
from models.epsnet import get_model
from utils.data import FOLLOW_BATCH_DPM
from utils.datasets import get_dataset
from utils.misc import *
from utils.reconstruct import *
from utils.reconstruct_mdm import (make_mol_openbabel)
from utils.sample import DistributionNodes
from utils.transforms import *
import traceback

STATUS_RUNNING = 'running'
STATUS_FINISHED = 'finished'
STATUS_FAILED = 'failed'
FOLLOW_BATCH = FOLLOW_BATCH_DPM

atomic_numbers_crossdock = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])
atomic_numbers_pocket = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17, 34])
atomic_numbers_pdbind = torch.LongTensor([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 23, 26, 27, 29, 33, 34, 35, 44, 51, 53, 78])
P_ligand_element_100 = torch.LongTensor([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 23, 26, 29, 33, 34, 35, 44, 51, 53, 78])
# P_ligand_element_filter = torch.LongTensor([1, 35, 5, 6, 7, 8, 9, 15, 16, 17, 53])
P_ligand_element_filter = torch.LongTensor([1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53])


def RMSD(probe, ref):
    rmsd = 0.0
    # print(amap)
    assert len(probe) == len(ref)
    atomNum = len(probe)
    for i in range(len(probe)):
        posp = probe[i]
        posf = ref[i]
        rmsd += dist_2(posp, posf)
    rmsd = math.sqrt(rmsd / atomNum)
    return rmsd


def dist_2(atoma_xyz, atomb_xyz):
    dis2 = 0.0
    for i, j in zip(atoma_xyz, atomb_xyz):
        dis2 += (i - j) ** 2
    return dis2

def num_confs(num: str):
    if num.endswith('x'):
        return lambda x: x * int(num[:-1])
    elif int(num) > 0:
        return lambda x: int(num)
    else:
        raise ValueError()


def save_sdf(mol, sdf_dir, gen_file_name):
    writer = Chem.SDWriter(os.path.join(sdf_dir, gen_file_name))
    writer.write(mol, confId=0)
    writer.close()


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-build_method', type=str, default='reconstruct', help='build or reconstruct')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--save_traj', action='store_true',
                        help='whether store the whole trajectory for visualization')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100)
    parser.add_argument('--save_results', type=bool, default=False)
    parser.add_argument('--save_sdf', type=bool, default=False)
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=0,
                        help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=float('inf'),
                        help='enable global gradients only when noise is low')
    parser.add_argument('--local_start_sigma', type=float, default=float('inf'),
                        help='enable local gradients only when noise is low')
    parser.add_argument('--w_global_pos', type=float, default=1.0,
                        help='weight for global gradients')
    parser.add_argument('--w_local_pos', type=float, default=1.0,
                        help='weight for local gradients')
    parser.add_argument('--w_global_node', type=float, default=1.0,
                        help='weight for global gradients')
    parser.add_argument('--w_local_node', type=float, default=1.0,
                        help='weight for local gradients')
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='generalized',
                        help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()
    
    model_id = args.ckpt.split('/')[-1].replace('.', '')
    # Load configs
   
    ckpt = torch.load(args.ckpt)
    config = ckpt['config']

    # if 'pocket' in args.ckpt:
    #     args.config = '/./configs/pocket.yml'
    #     config = load_config(args.config)

    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    # args.sampling_type = 'build'

    if args.n_steps == 0:
        args.n_steps = ckpt['config'].model.num_diffusion_timesteps

    # Logging

    # logger = get_logger('sample', log_dir)
    tag = 'result'
    output_dir = get_new_log_dir(log_dir, args.sampling_type + "_final_{}_{}_test".format(args.build_method, model_id) + tag,
                                 tag=args.tag)
    logger = get_logger('test', output_dir)

    logger.info(args)
    logger.info(config)

    # Data
    pocket = True
    if 'fintune' in args.ckpt:
        config.dataset.name = 'crossdock'
        config.dataset.path = './data/crossdocked_pocket10'
        config.dataset.split = './data/split_by_name.pt'

    logger.info('Loading {} data...'.format(config.dataset.name))
    if config.dataset.name == 'crossdock' or 'pdbind':
        if 'pocket' or 'fintune' in args.ckpt:
            atomic_numbers = atomic_numbers_pocket
            dataset_info = get_dataset_info('crossdock_pocket', False)
            pocket = True
        else:
            # atomic_numbers = atomic_numbers_pocket
            # pocket=True
            atomic_numbers = atomic_numbers_crossdock
            dataset_info = get_dataset_info('crossdock', False)
        # protein_root = './data/crossdocked_pocket10'
        protein_root = './data/test_data/test_pdbqt'
    else:
        if 'filter' in config.dataset.split:
            atomic_numbers = P_ligand_element_filter
        elif '100' in config.dataset.split:
            atomic_numbers = P_ligand_element_100
        else:
            atomic_numbers = atomic_numbers_pdbind
        protein_root = './data/protein_ligand/pdbind/v2020'
    atomic_numbers = atomic_numbers_crossdock

    histogram = dataset_info['n_nodes']
    nodes_dist = DistributionNodes(histogram)
    protein_featurizer = FeaturizeProteinAtom(config.dataset.name, pocket=pocket)
    ligand_featurizer = FeaturizeLigandAtom(config.dataset.name, pocket=pocket)

    masking = LigandMaskAll()

    transform = Compose([
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        FeaturizeLigandBond(),
        masking,
        CountNodesPerGraph(),
        GetAdj(),
    ])

    dataset, subsets = get_dataset(
        config=config.dataset,
        transform=transform,
    )
    testset = subsets['test']
    trainset = subsets['train']
    print(len(trainset))
    print(len(testset))

    test_set_selected = []
    for i, data in enumerate(testset):
        if not (args.start_idx <= i < args.end_idx): continue
        test_set_selected.append(data)
        # break

    print(len(test_set_selected))

    with open(os.path.join(log_dir, 'pocket_info.txt'), 'a') as f:
        f.write(data.protein_filename + '\n')

    logger.info('Building model...')
    # ckpt = torch.load(config.model.main.checkpoint, map_location=device)
    logger.info(config.model['network'])
    print(config.model)
    model = get_model(config.model).to(device)
    # model = torch.nn.DataParallel(model) # ddp mode
    model.load_state_dict(ckpt['model'])

    model.eval()

    clip_local = None
    print(device)
    sa_list = []
    r_sa_list = []
    rd_sa_list = []

    qed_list = []
    r_qed_list = []
    rd_qed_list = []

    num_samples = 1
    valid = 0
    stable = 0
    sum_rms = 0
    sum_rmsd = 0
    high_affinity = 0
    rmsd_list = []

    outliers = []

    smile_list = []
    results = []

    protein_files = []

    logP_list = []
    r_logP_list = []
    Lipinski_list = []
    r_Linpinski_list = []
    vina_score_list = []

    rd_vina_score_list = []

    mol_list = []
    try:
        num_atom_type = config.model['atom_type']  # 9 #config.model['atom_type']
    except:
        num_atom_type = config.model['num_atom']
    # if 'pocket' in args.ckpt:
    #     num_atom = 9

    save_results = args.save_results
    save_sdf_flag = args.save_sdf

    if save_sdf_flag:
        sdf_dir = os.path.join(output_dir, 'generated_ligand_all')
        print('sdf idr:', sdf_dir)
        if not os.path.exists(sdf_dir):
            os.mkdir(sdf_dir)
    if save_results:
        file_save_dir = './data/test_data/'
        file_dir = './data/crossdocked_pocket10'

        if not os.path.exists(file_save_dir):
            os.mkdir(file_save_dir)
    nodes_dist = DistributionNodes(dataset_info['n_nodes'])


    config.dataset.name = 'crossdock'
    with open('test_vina_{}.pkl'.format(config.dataset.name), 'rb') as f:
        test_vina_score_list = pickle.load(f)

    for n, data in enumerate(tqdm(test_set_selected)):
        try_num = 10
        FINISHED = False
        num_samples = 100

        element = data.ligand_element.tolist()
        protein_files.append(data.protein_filename)
        f_dir, f_name = os.path.split(data.protein_filename)

        gen_file_name = f_name.split('.')[0] + '_gen.sdf'
        print(gen_file_name)
        # sdf_dir =  os.path.join(file_save_dir, f_dir)

        pdb_name = f_name.split('_')[0]

        with torch.no_grad():
            num_points = data.ligand_element.size(0)
            batch = Batch.from_data_list([data] * 1, follow_batch=FOLLOW_BATCH).to(device)

            nodesxsample = nodes_dist.sample(1).tolist()

            pos_init = torch.randn(data.ligand_element.size(0), 3).to(device)
            atom_feature = torch.randn(data.ligand_element.size(0), num_atom_type).to(device)  # 8 for ligand, 9 for pocket
            bond_index = batch.ligand_bond_index
            ligand_batch = torch.zeros(data.ligand_element.size(0), dtype=torch.int64).to(device)
            ligand_bond_type = torch.ones(bond_index.size(1), dtype=torch.long) * 2
            ligand_bond_type = ligand_bond_type.to(device)


            # jointly noise
            context = None
            # protein_atom_type = batch.protein_atom_feature_full.float()
            # if 'pocket' in args.ckpt:
            protein_atom_type = batch.protein_atom_feature.float()
            while not FINISHED and try_num > 0:
                try:
                    try_num -= 1
                    pos_gen, pos_gen_traj, atom_type, atom_traj = model.langevin_dynamics_sample(
                        ligand_atom_type=atom_feature,
                        ligand_pos_init=pos_init,
                        ligand_bond_index=bond_index,
                        ligand_bond_type=ligand_bond_type,
                        ligand_num_node=torch.tensor([atom_feature.size(0)]).to(device),
                        ligand_batch=ligand_batch,
                        protein_atom_type=protein_atom_type,
                        protein_atom_feature_full=batch.protein_atom_feature_full.float(),
                        protein_pos=batch.protein_pos,
                        protein_bond_index=batch.protein_bond_index,
                        protein_bond_type=batch.protein_bond_type,
                        protein_backbone_mask=batch.protein_is_backbone,
                        protein_batch=batch.protein_element_batch,
                        num_graphs=batch.num_graphs,
                        extend_order=False,  # Done in transforms.
                        n_steps=args.n_steps,
                        step_lr=1e-6,  # 1e-6
                        w_global_pos=args.w_global_pos,
                        w_global_node=args.w_global_node,
                        w_local_pos=args.w_local_pos,
                        w_local_node=args.w_local_node,
                        global_start_sigma=args.global_start_sigma,
                        clip=args.clip,
                        clip_local=clip_local,
                        sampling_type=args.sampling_type,
                        eta=args.eta,
                        context=context
                    )

                    pos_list = pos_gen.reshape(1, -1, 3)
                    atom_list = atom_type.reshape(1, -1, atom_feature.size(1))
                    # atom_charge_list = atom_charge.reshape(num_samples, -1, 1)

                    pos = pos_list[0].detach().cpu()

                    atom_type = atom_list[0].detach().cpu()
                    num_atom_type = len(atomic_numbers)
                    if args.build_method == 'reconstruct':
                        new_element = torch.tensor(
                            [atomic_numbers_crossdock[m] for m in torch.argmax(atom_type[:, : ], dim=1)])
                        # print(new_element)
                        indicators_elements = torch.argmax(atom_type[:, num_atom_type:], dim=1)
                        indicators = torch.zeros([pos.size(0), len(ATOM_FAMILIES)])
                        for i, n in enumerate(indicators_elements):
                            indicators[i, n] = 1
                        gmol = reconstruct_from_generated(pos, new_element, indicators)
                    elif args.build_method == 'build':
                        new_element = torch.argmax(atom_type[:, :num_atom_type], dim=1)
                        # print(new_element)
                        gmol = make_mol_openbabel(pos, new_element, dataset_info)

                    a = 0

                    rmol = reconstruct_from_generated(data.ligand_pos, data.ligand_element, data.ligand_atom_feature)
                    r_smile = Chem.MolToSmiles(rmol)
                    print("reference smile:", r_smile)
                    g_smile = Chem.MolToSmiles(gmol)
                    print("generated smile:", g_smile)

                    if g_smile is not None:
                        valid += 1
                        num_samples -= 1
                        smile_list.append(g_smile)
                        logger.info('Successfully generate molecule for {}'.format(pdb_name))
                        if '.' not in g_smile:
                            stable += 1
                        if g_smile.count('.') > 0:
                            continue

                    else:
                        raise MolReconsError()
                    # valid = filter_rd_mol(rmol)
                    # exit()

                    idx = args.start_idx + n

                    _, r_sa = compute_sa_score(rmol)
                    print("Reference SA score:", r_sa)

                    r_qed = qed(rmol)
                    print("Reference QED score:", r_qed)

                    _, g_sa = compute_sa_score(gmol)
                    print("Generate SA score:", g_sa)

                    g_qed = qed(gmol)
                    print("Generate QED score:", g_qed)

                    g_logP = MolLogP(gmol)
                    r_logP = MolLogP(rmol)
                    # g_logP = Crippen.MolLogP(gmol)
                    print("Generate logP:", g_logP)

                    g_Lipinski = obey_lipinski(gmol)
                    r_Lipinski = obey_lipinski(rmol)
                    print("Generate Lipinski:", g_Lipinski)

                    # vina_task = QVinaDockingTask.from_generated_data(data.protein_filename, gmol,
                    #                                                  protein_root=protein_root)
                    receptor_file = os.path.basename(data.protein_filename).replace('.pdb','')+'.pdbqt'
                    receptor_file = Path(os.path.join(protein_root,receptor_file))
                    g_vina_score = calculate_qvina2_score(
                    receptor_file, gmol, sdf_dir, return_rdmol=False)[0]

                    # g_vina_results = vina_task.run_sync()
                    # g_vina_score = g_vina_results[0]['affinity']
                    if g_vina_score > -2:
                        raise MolReconsError()
                    print("Generate vina score:", g_vina_score)

                    rd_vina_score = test_vina_score_list[n]
                    print('Reference vina score:', rd_vina_score)
                    g_high_affinity = False
                    if g_vina_score < rd_vina_score:
                        high_affinity += 1.0
                        g_high_affinity = True
                    # if save_sdf_flag:
                    #     save_sdf(gmol, sdf_dir, str(g_vina_score) + "_" + gen_file_name)
                    if save_results:
                        metrics = {'SA': g_sa, 'QED': g_qed, 'logP': g_logP, 'Lipinski': g_Lipinski,
                                   'vina': g_vina_score, 'high_affinity': g_high_affinity}
                        result = {'atom_type': atom_type.detach().cpu(),
                                  'pos': pos.detach().cpu(),
                                  'smile': g_smile,
                                  'protein_file': data.protein_filename,
                                  'ligand_file': data.ligand_filename,
                                  'generated_ligand_sdf': gen_file_name,
                                  'mol': gmol,
                                  'metric_result': metrics}
                        results.append(result)
                    FINISHED = True
                    r_sa_list.append(r_sa)
                    r_qed_list.append(r_qed)
                    r_logP_list.append(r_logP)
                    r_Linpinski_list.append(r_Lipinski)
                    mol_list.append(gmol)
                    vina_score_list.append(g_vina_score)
                    sa_list.append(g_sa)
                    qed_list.append(g_qed)
                    logP_list.append(g_logP)
                    Lipinski_list.append(g_Lipinski)
                    break

                except FloatingPointError as e:  
                    print(e)
                    print(traceback.format_exc())
                    clip_local = 20
                    # nodesxsample = nodes_dist.sample(1).tolist()
                    nodesxsample[0] = data.ligand_element.size(0)
                    if try_num < 9:
                        nodesxsample = nodes_dist.sample(1).tolist()
                    pos_init = torch.randn(1, nodesxsample[0], 3).reshape(-1, 3).to(device)
                    atom_feature = torch.randn(1, nodesxsample[0], num_atom_type).reshape(-1, num_atom_type).to(device)
                    bond_index = get_adj_matrix(nodesxsample[0]).to(device)
                    ligand_batch = torch.zeros(nodesxsample[0], dtype=torch.int64).to(device)
                    ligand_bond_type = torch.ones(bond_index.size(1), dtype=torch.long) * 2
                    ligand_bond_type = ligand_bond_type.to(device)
                    logger.warning(
                        'Ignoring, because reconstruction error encountered or retrying with local clipping or vina error.')
                    print('Resample the number of the atoms and regenerate!')
                # print('mean_rmsd:', sum_rmsd/(n+1))

    logger.info(args.ckpt)
    logger.info('valid:%d' % valid)
    logger.info('stable:%d' % stable)
    # print('mean_rms:', sum_rms/100)
    # print('mean_rmsd:', sum_rmsd/100)

    logger.info('generate:%d' % len(sa_list))
    logger.info('reference mean sa:%f' % mean(r_sa_list))
    logger.info('reference mean qed:%f' % mean(r_qed_list))
    logger.info('reference mean logP:%f' % mean(r_logP_list))
    logger.info('reference mean Lipinski:{}'.format(np.mean(r_Linpinski_list)))
    logger.info('reference reference mean vina:%f' % mean(test_vina_score_list))

    # logger.info('original reference mean sa:%f'%mean(rd_sa_list))
    # logger.info('original reference mean qed:%f'%mean(rd_qed_list))
    # logger.info('original reference mean vina:%f'%mean(rd_vina_score_list))

    logger.info('mean sa:%f' % mean(sa_list))
    logger.info('mean qed:%f' % mean(qed_list))
    logger.info('mean logP:%f' % mean(logP_list))
    logger.info('mean Lipinski:{}'.format(np.mean(Lipinski_list)))
    print(np.mean(Lipinski_list))
    logger.info('mean vina:%f' % mean(vina_score_list))
    logger.info('high affinity:%d' % high_affinity)
    logger.info('diversity:%f' % calculate_diversity(mol_list))
    print(vina_score_list)
    print(Lipinski_list)
    # print(rmsd_list)
    # print(outliers)
    # print((sum_rmsd-sum(outliers))/(len(rmsd_list)-len(outliers)))
    # with open('test_vina_pdbind.pkl','wb') as f:
    #     pickle.dump(rd_vina_score_list, f)
    #     f.close()

    if save_results:
        save_path = os.path.join(output_dir, 'samples_all.pkl')
        logger.info('Saving samples to: %s' % save_path)

        save_smile_path = os.path.join(output_dir, 'samples_smile.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
            f.close()

        with open(save_smile_path, 'wb') as f:
            pickle.dump(smile_list, f)
            f.close()
