import argparse
import pickle



from configs.dataset_config import get_dataset_info
from evaluation import *
from evaluation.sascorer import *
from models.epsnet import get_model
from utils.datasets import get_dataset
from utils.misc import *
from utils.reconstruct import *
from utils.reconstruct_mdm import make_mol_openbabel
from utils.sample import DistributionNodes
from utils.sample import construct_dataset_pocket
from utils.transforms import *
from utils.protein_ligand import PDBProtein, parse_sdf_file
from utils.data import torchify_dict

from torch_geometric.data import Batch


STATUS_RUNNING = 'running'
STATUS_FINISHED = 'finished'
STATUS_FAILED = 'failed'
FOLLOW_BATCH = ['ligand_atom_feature', 'protein_atom_feature_full']

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



def get_adj_matrix(n_particles):
    rows, cols = [], []
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)
    # print(n_particles)
    rows = torch.LongTensor(rows).unsqueeze(0)
    cols = torch.LongTensor(cols).unsqueeze(0)
    # print(rows.size())
    adj = torch.cat([rows, cols], dim=0)
    return adj


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
    # sys.path.append('/.')
    # os.chdir('/.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--save_traj', action='store_true',
                        help='whether store the whole trajectory for sampling')
    parser.add_argument('--save_results', type=bool, default=False)
    parser.add_argument('--save_sdf', type=bool, default=False)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('-build_method', type=str, default='build',help='build or reconstruct')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=25)
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
    parser.add_argument('--sampling_type', type=str, default='ld',
                        help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    

    args = parser.parse_args()

    # Load configs

    ckpt = torch.load(args.ckpt)
    config = ckpt['config']

    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    num_samples = args.num_samples
    batch_size = args.batch_size

    seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    if args.n_steps == 0:
        args.n_steps = ckpt['config'].model.num_diffusion_timesteps

    # Logging

    # logger = get_logger('sample', log_dir)
    tag = 'result'
    output_dir = get_new_log_dir(log_dir, args.sampling_type + args.build_method+'_'+str(args.start_idx) +
                                 '_' + str(args.end_idx) + '_' + tag, tag=args.tag)
    logger = get_logger('test', output_dir)

    # for 1k sample
    config.dataset.split='./data/split_by_name_1k.pt'
    logger.info(args)
    logger.info(config)


    dataset_info = get_dataset_info('crossdock', False)
    histogram = dataset_info['n_nodes']
    nodes_dist = DistributionNodes(histogram)

    # Data

    logger.info('Loading {} data...'.format(config.dataset.name))
    if config.dataset.name == 'crossdock':
        if 'pocket' or'sa' in args.ckpt:
            atomic_numbers = atomic_numbers_pocket
            dataset_info = get_dataset_info('crossdock_pocket', False)
            pocket = True
        else:
            # atomic_numbers = atomic_numbers_pocket
            # pocket=True
            atomic_numbers = atomic_numbers_crossdock
            dataset_info = get_dataset_info('crossdock', False)
        protein_root = './data/crossdocked_pocket10'
    else:
        if 'filter' in config.dataset.split:
            atomic_numbers = P_ligand_element_filter
        elif '100' in config.dataset.split:
            atomic_numbers = P_ligand_element_100
        else:
            atomic_numbers = atomic_numbers_pdbind
        protein_root = './data/protein_ligand/pdbind/v2020'
    protein_featurizer = FeaturizeProteinAtom(config.dataset.name, pocket=pocket)
    ligand_featurizer = FeaturizeLigandAtom(config.dataset.name, pocket=pocket)

    transform = Compose([
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        FeaturizeLigandBond(),
        CountNodesPerGraph(),
        GetAdj()
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
    # FOLLOW_BATCH = ['ligand_atom_type','protein_atom_feature_full']
    for i, data in enumerate(testset):
        if not (args.start_idx <= i < args.end_idx): continue
        test_set_selected.append(data)
        # break

    print(len(test_set_selected))

    with open(os.path.join(log_dir, 'pocket_info.txt'), 'a') as f:
        f.write(data.protein_filename + '\n')

    logger.info('Building model...')
    logger.info(config.model['network'])
    print(config.model)
    model = get_model(config.model).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()

    clip_local = None
    print(device)
    time_list = []
    sa_list = []
    r_sa_list = []
    rd_sa_list = []

    qed_list = []
    r_qed_list = []
    rd_qed_list = []

    plogp_list = []
    r_plogo_list = []

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
    Lipinski_list = []
    vina_score_list = []

    rd_vina_score_list = []

    save_results = args.save_results
    save_sdf_flag = args.save_sdf
    if save_results:
        file_save_dir = './data/test_data/'
        if not os.path.exists(file_save_dir):
            os.mkdir(file_save_dir)
    if save_sdf_flag:
        sdf_dir = './results/crossdocked/MDM/protein_context_Schent_build/'
        if not os.path.exists(sdf_dir):
            os.mkdir(sdf_dir)

        
    nodes_dist = DistributionNodes(dataset_info['n_nodes'])

    # with open('test_vina_{}.pkl'.format(config.dataset.name), 'rb') as f:
    #     test_vina_score_list = pickle.load(f)

    for n, data in enumerate(tqdm(test_set_selected)):
        rmol = reconstruct_from_generated(data.ligand_pos, data.ligand_element,
                                        data.ligand_atom_feature)
        r_smile = Chem.MolToSmiles(rmol)
        print("reference smile:", r_smile)
        try_num = 20
        FINISHED = False

        

        element = data.ligand_element.tolist()
        protein_files.append(data.protein_filename)
        f_dir, f_name = os.path.split(data.protein_filename)
        # print(f_dir)

        gen_file_name = f_name.split('.')[0] + '_gen.sdf'
        print(gen_file_name)
        # sdf_dir =  os.path.join(file_save_dir, f_dir)

        pdb_name = f_name.split('_')[0]

        protein_atom_feature = data.protein_atom_feature.float()
        protein_atom_feature_full = data.protein_atom_feature_full.float()

        with torch.no_grad():
            num_points = data.ligand_element.size(0)
            num_points_fix = num_points
            context = None
            t_pocket_start = time.time()
            while num_samples > 0 and try_num > 0:
                largest_mol_flag = False

                if num_samples < 1:
                    print(num_samples)

                if try_num < 10:
                    num_points_fix = None
                num_points_fix = None  # only for no spatial
                data_list, _ = construct_dataset_pocket(num_samples, batch_size, dataset_info,
                                                        num_points, num_points_fix, None,None,
                                                        protein_atom_feature, protein_atom_feature_full,
                                                        data.protein_pos, data.protein_bond_index)

                batch = Batch.from_data_list(data_list[0], follow_batch=FOLLOW_BATCH).to(device)
                try:
                    try_num -= 1
                    pos_gen, pos_gen_traj, atom_type, atom_traj = model.langevin_dynamics_sample(
                        ligand_atom_type=batch.ligand_atom_feature,
                        ligand_pos_init=batch.ligand_pos,
                        ligand_bond_index=batch.ligand_bond_index,
                        ligand_bond_type=None,
                        ligand_num_node=batch.ligand_num_node,
                        ligand_batch=batch.ligand_atom_feature_batch,
                        protein_atom_type=batch.protein_atom_feature.float(),
                        protein_atom_feature_full=batch.protein_atom_feature_full.float(),
                        protein_pos=batch.protein_pos,
                        protein_bond_index=batch.protein_bond_index,
                        protein_backbone_mask=None,
                        protein_batch=batch.protein_atom_feature_full_batch,
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

                    pos_list = unbatch(pos_gen, batch.ligand_atom_feature_batch)
                    atom_list = unbatch(atom_type, batch.ligand_atom_feature_batch)
                    # atom_charge_list = atom_charge.reshape(num_samples, -1, 1)
                    for m in range(num_samples):
                        try:
                            pos = pos_list[m].detach().cpu()
                            # pos = pos+torch.mean(data.protein_pos,0)
                            atom_type = atom_list[m].detach().cpu()

                            

                            new_ligand = torch.zeros([pos.size(0), len(ATOM_FAMILIES)], dtype=np.long)

                            a = 0
                            num_atom_type = len(atomic_numbers)
                            if args.build_method == 'reconstruct':
                                new_element = torch.tensor([atomic_numbers_crossdock[m] for m in torch.argmax(atom_type[:,:8],dim=1)])
                                indicators_elements = torch.argmax(atom_type[:,8:],dim=1)
                                indicators = torch.zeros([pos.size(0), len(ATOM_FAMILIES)], dtype=np.long)
                                for i, n in enumerate(indicators_elements):
                                    indicators[i,n] = 1

                                gmol = reconstruct_from_generated(pos,new_element,indicators)
                            
                            elif args.build_method == 'build':
                                new_element = torch.argmax(atom_type[:,:num_atom_type], dim=1)
                                gmol = make_mol_openbabel(pos, new_element, dataset_info)

                            # gen_mol = set_rdmol_positions(rdmol, data.ligand_pos)
                            g_smile = Chem.MolToSmiles(gmol)
                            print("generated smile:", g_smile)

                            if g_smile is not None:
                                FINISHED = True
                                if '.' not in g_smile:
                                    stable += 1
                                if g_smile.count('.') > 1:
                                    raise MolReconsError()

                                if try_num < 10:
                                    largest_mol_flag = True
                                # if try_num<10:
                                #     args.sampling_type = 'ddpm_noisy'
                                if largest_mol_flag:
                                    mol_frags = Chem.rdmolops.GetMolFrags(gmol, asMols=True, sanitizeFrags=False)
                                    gmol = max(mol_frags, default=gmol, key=lambda m: m.GetNumAtoms())
                                    g_smile = Chem.MolToSmiles(gmol)
                                    print("largest generated smile part:", g_smile)
                                    if g_smile is None:
                                        raise MolReconsError()
                                if g_smile.count('.') > 0:
                                    raise MolReconsError()
                                if len(g_smile) < 4:
                                    raise MolReconsError()
                                if save_sdf_flag:
                                    save_sdf(gmol, sdf_dir, gen_file_name)
                                valid += 1
                                num_samples -= 1
                                smile_list.append(g_smile)
                                print('Successfully generate molecule for {}, remining {} samples generated'.format(
                                    pdb_name, num_samples))

                            else:
                                raise MolReconsError()

                            if save_results:
                                # metrics = {'SA':g_sa,'QED':g_qed,'logP':g_logP,'Lipinski':g_Lipinski,
                                #            'vina':g_vina_score,'high_affinity':g_high_affinity}
                                result = {'atom_type': atom_type.detach().cpu(),
                                          'pos': pos.detach().cpu(),
                                          'smile': g_smile,
                                          # 'l_smile':lg_smile,
                                          'protein_file': data.protein_filename,
                                          'ligand_file': data.ligand_filename,
                                          # 'generated_ligand_sdf': gen_file_name,
                                          'mol': gmol,
                                          # 'l_mol':largest_mol,
                                          # 'metric_result':metrics}
                                          }
                                results.append(result)
                            if num_samples == 0:
                                break

                        except(RuntimeError, MolReconsError, TypeError, IndexError,
                               OverflowError):  # MolReconsError,TypeError,IndexError,OverflowError
                            print('Invalid,continue')


                except (FloatingPointError):  # ,MolReconsError,TypeError,IndexError,OverflowError
                    clip_local = 20
                    logger.warning(
                        'Ignoring, because reconstruction error encountered or retrying with local clipping or vina error.')
                    print('Resample the number of the atoms and regenerate!')
            time_list.append(time.time() - t_pocket_start)
            logger.info(str(data.protein_filename) + 'takes {} seconds'.format(time.time() - t_pocket_start))
    times_arr = torch.tensor(time_list)
    try:
        logger.info(f"Time per pocket: {times_arr.mean():.3f} \pm "
                    f"{times_arr.std(unbiased=False):.2f}")
    except:
        logger.info(torch.mean(times_arr))

    if save_results:
        save_path = os.path.join(output_dir, 'samples_all.pkl')
        logger.info('Saving samples to: %s' % save_path)

        # save_smile_path = os.path.join(output_dir, 'samples_smile.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
            f.close()

        save_time_path = os.path.join(output_dir, 'time.pkl')
        logger.info('Saving time to: %s' % save_path)
        with open(save_time_path, 'wb') as f:
            pickle.dump(time_list, f)
            f.close()
