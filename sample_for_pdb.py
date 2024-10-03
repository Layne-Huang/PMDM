import os
os.system('module load openmind8/cuda/11.7')

import argparse
from statistics import mean

from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities
from easydict import EasyDict

from configs.dataset_config import get_dataset_info
from evaluation import *
# from rdkit.Chem import Draw
from evaluation.similarity import calculate_diversity
from models.epsnet import get_model
from utils.misc import *
from utils.protein_ligand import PDBProtein
from utils.reconstruct import *
from utils.reconstruct_mdm import (make_mol_openbabel,
                                   mol2smiles)
# from sample import *    # Import everything from `sample.py`
from utils.sample import *
from utils.sample import construct_dataset_pocket
from utils.transforms import *
from utils.data import torchify_dict
from utils.protein_ligand import PDBProtein, parse_sdf_file


from rdkit.Chem.rdchem import BondType
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures

from torch_geometric.data import Batch

FOLLOW_BATCH = ['ligand_atom_feature', 'protein_atom_feature_full']
atomic_numbers_crossdock = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])
atomic_numbers_pocket = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17, 34])
atomic_numbers_pdbind = torch.LongTensor([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 23, 26, 27, 29, 33, 34, 35, 44, 51, 53, 78])
P_ligand_element_100 = torch.LongTensor([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 23, 26, 29, 33, 34, 35, 44, 51, 53, 78])
# P_ligand_element_filter = torch.LongTensor([1, 35, 5, 6, 7, 8, 9, 15, 16, 17, 53])
P_ligand_element_filter = torch.LongTensor([1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53])


def save_sdf(mol, sdf_dir, gen_file_name):
    writer = Chem.SDWriter(os.path.join(sdf_dir, gen_file_name))
    writer.write(mol, confId=0)
    writer.close()

def parse_sdf_file(path):
    # Import a feature library, create a feature factory, and calculate chemical features using the feature factory
    # Calculate ring information
    
    
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=False, sanitize=False)))
    rdmol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(rdmol)
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)])

    '''
    Each feature found contains information about the feature family 
    (e.g., donor, acceptor), feature type, atoms associated with the feature, and the corresponding feature index.

    Feature family information: GetFamily()
    Feature type information: GetType()
    Atoms associated with the feature: GetAtomIds()
    Corresponding feature index: GetId()
    '''
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    with open(path, 'r') as f:
        sdf = f.read()

    sdf = sdf.splitlines()
    num_atoms, num_bonds = map(int, [sdf[3][0:3], sdf[3][3:6]])
    assert num_atoms == rd_num_atoms

    ptable = Chem.GetPeriodicTable()

    element, pos = [], []
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    for atom_line in map(lambda x: x.split(), sdf[4:4 + num_atoms]):
        x, y, z = map(float, atom_line[:3])
        symb = atom_line[3]
        atomic_number = ptable.GetAtomicNumber(symb.capitalize())
        # repalce Br as Cl
        if atomic_number == 35:
            atomic_number = 17
        element.append(atomic_number)
        pos.append([x, y, z])

        atomic_weight = ptable.GetAtomicWeight(atomic_number)
        accum_pos += np.array([x, y, z]) * atomic_weight
        accum_mass += atomic_weight

    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)

    element = np.array(element, dtype=np.int64)
    pos = np.array(pos, dtype=np.float32)

    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
    bond_type_map = {
        1: BOND_TYPES[BondType.SINGLE],
        2: BOND_TYPES[BondType.DOUBLE],
        3: BOND_TYPES[BondType.TRIPLE],
        4: BOND_TYPES[BondType.AROMATIC],
    }
    row, col, edge_type = [], [], []
    for bond_line in sdf[4 + num_atoms:4 + num_atoms + num_bonds]:
        start, end = int(bond_line[0:3]) - 1, int(bond_line[3:6]) - 1
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_type_map[int(bond_line[6:9])]]

    edge_index = np.array([row, col], dtype=np.int64)
    edge_type = np.array(edge_type, dtype=np.int64)

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
    }
    return data

def pdb_to_pocket_data(pdb_path, sdf_path, center=0, bbox_size=0):
    center = torch.FloatTensor(center)
    warnings.simplefilter('ignore', BiopythonWarning)
    ptable = Chem.GetPeriodicTable()
    parser = PDBParser()
    model = parser.get_structure(None, pdb_path)[0]

    protein_dict = EasyDict({
        'element': [],
        'pos': [],
        'is_backbone': [],
        'atom_to_aa_type': [],
    })
    for atom in unfold_entities(model, 'A'):
        res = atom.get_parent()
        resname = res.get_resname()
        if resname == 'MSE': resname = 'MET'
        if resname not in PDBProtein.AA_NAME_NUMBER: continue  # Ignore water, heteros, and non-standard residues.

        element_symb = atom.element.capitalize()
        if element_symb == 'H': continue
        x, y, z = atom.get_coord()
        pos = torch.FloatTensor([x, y, z])
        # if (pos - center).abs().max() > (bbox_size / 2): 
        #     continue

        protein_dict['element'].append(ptable.GetAtomicNumber(element_symb))
        protein_dict['pos'].append(pos)
        protein_dict['is_backbone'].append(atom.get_name() in ['N', 'CA', 'C', 'O'])
        protein_dict['atom_to_aa_type'].append(PDBProtein.AA_NAME_NUMBER[resname])

    # if len(protein_dict['element']) == 0:
    #     raise ValueError('No atoms found in the bounding box (center=%r, size=%f).' % (center, bbox_size))

    protein_dict['element'] = torch.LongTensor(protein_dict['element'])
    protein_dict['pos'] = torch.stack(protein_dict['pos'], dim=0)
    protein_dict['is_backbone'] = torch.BoolTensor(protein_dict['is_backbone'])
    protein_dict['atom_to_aa_type'] = torch.LongTensor(protein_dict['atom_to_aa_type'])

    if sdf_path == None:
        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=protein_dict,
            ligand_dict={
                'element': torch.empty([0, ], dtype=torch.long),
                'pos': torch.empty([0, 3], dtype=torch.float),
                'atom_feature': torch.empty([0, 8], dtype=torch.float),
                'bond_index': torch.empty([2, 0], dtype=torch.long),
                'bond_type': torch.empty([0, ], dtype=torch.long),
            }
        )
    else:
        ligand_data = torchify_dict(parse_sdf_file(sdf_path))
        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=protein_dict,
            ligand_dict=ligand_data)
    return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str,
                        default='./example/4yhj.pdb')
    parser.add_argument('--sdf_path', type=str,
                        default=None, help='path to the sdf file of reference ligand')
    parser.add_argument('--num_atom', type=int,
                        default=29)
    parser.add_argument('--build_method', type=str, default='reconstruct', help='build or reconstruct')
    parser.add_argument('--config', type=str)
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--save_sdf', type=bool, default=True)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=0,
                        help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=float('inf'),
                        help='enable global gradients only when noise is low')  # float('inf')
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

    protein_root = os.path.dirname(args.pdb_path)
    pdb_name = os.path.basename(args.pdb_path)[:4]
    protein_filename = os.path.basename(args.pdb_path)

    # Load configs
    ckpt = torch.load(args.ckpt)
    config = ckpt['config']
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    seed_all(config.train.seed)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), 'custom_pdb')


    if args.n_steps == 0:
        args.n_steps = ckpt['config'].model.num_diffusion_timesteps

    tag = 'result'
    output_dir = get_new_log_dir(log_dir, args.sampling_type + tag, tag=args.tag)
    logger = get_logger('test', output_dir)

    logger.info(args)
    logger.info(config)

    pocket = False
    logger.info('Loading {} data...'.format(config.dataset.name))
    if config.dataset.name == 'crossdock':
        pocket = True
        atomic_numbers = atomic_numbers_pocket
        dataset_info = get_dataset_info('crossdock_pocket', False)

    else:
        if 'filter' in config.dataset.split:
            atomic_numbers = P_ligand_element_filter
        elif '100' in config.dataset.split:
            atomic_numbers = P_ligand_element_100
        else:
            atomic_numbers = atomic_numbers_pdbind

    histogram = dataset_info['n_nodes']
    nodes_dist = DistributionNodes(histogram)
    # # Transform
    logger.info('Loading data...')
    protein_featurizer = FeaturizeProteinAtom(config.dataset.name, pocket=pocket)
    ligand_featurizer = FeaturizeLigandAtom(config.dataset.name, pocket=pocket)
    contrastive_sampler = ContrastiveSample(num_real=0, num_fake=0)
    masking = LigandMaskAll()
    transform = Compose([
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        CountNodesPerGraph(),
        GetAdj(only_prot=True),
        # AddHigherOrderEdges(order=config.model.edge_order)
    ])
    # # Data

    data = pdb_to_pocket_data(args.pdb_path, args.sdf_path)
    data = transform(data)
    if args.sdf_path is not None:
        ligand_data = data.ligand_atom_feature, data.ligand_atom_feature_full, data.ligand_pos, data.ligand_bond_index, data.ligand_bond_type,\
                    data.ligand_edge_index, data.ligand_edge_type
    else:
        ligand_data = None
    bond_index = data.ligand_bond_index
    bond_type = data.ligand_bond_type
    # Model
    logger.info('Building model...')
    logger.info(config.model['network'])
    print(config.model)
    model = get_model(config.model).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()

    # sample
    # gen_file_name = os.path.basename(args.pdb_path) + '_gen.sdf'
    # print(gen_file_name)
    save_sdf_flag = args.save_sdf
    if save_sdf_flag:
        sdf_dir = os.path.join(os.path.dirname(args.pdb_path), 'generate_ref')
        print('sdf idr:', sdf_dir)
        os.makedirs(sdf_dir, exist_ok=True)
    save_results = False
    valid = 0
    stable = 0
    high_affinity = 0.0
    num_samples = args.num_samples
    batch_size = args.batch_size
    num_points = args.num_atom  # random.randint(10,30)
    context = None
    smile_list = []
    results = []
    protein_files = []
    sa_list = []
    qed_list = []
    logP_list = []
    Lipinski_list = []
    vina_score_list = []
    rd_vina_score_list = []
    mol_list = []


    protein_atom_feature = data.protein_atom_feature.float()
    protein_atom_feature_full = data.protein_atom_feature_full.float()
    data_list, _ = construct_dataset_pocket(num_samples * 2, batch_size, dataset_info, num_points, num_points, None, ligand_data, 
                                            protein_atom_feature, protein_atom_feature_full, data.protein_pos, data.protein_bond_index)

    for n, datas in enumerate(tqdm(data_list)):
        batch = Batch.from_data_list(datas, follow_batch=FOLLOW_BATCH).to(device)

        if num_samples == 0:
            break
        with torch.no_grad():
            try:
                pos_gen, pos_gen_traj, atom_type, atom_traj = model.langevin_dynamics_sample(
                    ligand_atom_type=batch.ligand_atom_feature.float(),
                    ligand_pos_init=batch.ligand_pos,
                    ligand_bond_index=batch.ligand_bond_index,
                    ligand_bond_type=batch.ligand_bond_type,
                    ligand_num_node=batch.ligand_num_node,
                    ligand_batch=batch.ligand_atom_feature_batch,
                    protein_atom_type=batch.protein_atom_feature,
                    protein_atom_feature_full=batch.protein_atom_feature_full,
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
                    sampling_type=args.sampling_type,
                    eta=args.eta,
                    context=context
                )

                pos_list = unbatch(pos_gen, batch.ligand_atom_feature_batch)
                atom_list = unbatch(atom_type, batch.ligand_atom_feature_batch)
                if batch_size>num_samples:
                    batch_size=num_samples
                for m in range(batch_size):
                    try:
                        pos = pos_list[m].detach().cpu()
                        # pos = pos+torch.mean(data.protein_pos,0)
                        atom_type = atom_list[m].detach().cpu()
                        num_atom_type = len(atomic_numbers)
                        if args.build_method == 'reconstruct':
                            new_element = torch.tensor(
                                [atomic_numbers_crossdock[m] for m in torch.argmax(atom_type[:, :8], dim=1)])
                            indicators_elements = torch.argmax(atom_type[:, 8:], dim=1)
                            indicators = torch.zeros([pos.size(0), len(ATOM_FAMILIES)])
                            for i, n in enumerate(indicators_elements):
                                indicators[i, n] = 1
                            gmol = reconstruct_from_generated(pos, new_element, indicators)
                            # gmol = reconstruct_from_generated_with_edges(pos, new_element, bond_index, bond_type)
                        elif args.build_method == 'build':
                            new_element = torch.argmax(atom_type[:, :num_atom_type], dim=1)
                            # gmol = build_molecule(pos, new_element, dataset_info)
                            gmol = make_mol_openbabel(pos, new_element, dataset_info)

                        # gen_mol = set_rdmol_positions(rdmol, data.ligand_pos)
                        g_smile = mol2smiles(gmol)
                        print("generated smile:", g_smile)

                        if g_smile is not None:
                            FINISHED = True
                            valid += 1
                            if '.' not in g_smile:
                                stable += 1
                                num_samples -= 1
                                smile_list.append(g_smile)
                            else:
                                continue
                        else:
                            raise MolReconsError()

                        if save_sdf_flag:
                            print('save')
                            gen_file_name = '{}_{}.sdf'.format(pdb_name, str(num_samples))
                            print(gen_file_name)
                            save_sdf(gmol, sdf_dir, gen_file_name)
                        if save_results:
                            # metrics = {'SA':g_sa,'QED':g_qed,'logP':g_logP,'Lipinski':g_Lipinski,'vina':g_vina_score} 
                            result = {'atom_type': atom_type.detach().cpu(),
                                      'pos': pos.detach().cpu(),
                                      'smile': g_smile,
                                      'mol': gmol, }
                            # 'metric_result':metrics}
                            results.append(result)
                        logger.info(
                            'Successfully generate molecule for {}, remining {} samples generated'.format(pdb_name,
                                                                                                          num_samples))
                        mol_list.append(gmol)
                        if num_samples == 0:
                            break
                    except(MolReconsError):
                        print('Invalid,continue')


            except (FloatingPointError):  # ,MolReconsError,TypeError,IndexError,OverflowError
                clip_local = 20
                logger.warning(
                    'Ignoring, because reconstruction error encountered or retrying with local clipping or vina error.')
                print('Resample the number of the atoms and regenerate!')

    logger.info('valid:%d' % valid)
    logger.info('stable:%d' % stable)

