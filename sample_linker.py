import argparse
import os
import pickle
import warnings
from statistics import mean

import numpy as np
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities
from configs.dataset_config import get_dataset_info
from easydict import EasyDict
from evaluation import *
from evaluation.docking import *
# from rdkit.Chem import Draw
from evaluation.sascorer import *
from evaluation.score_func import *
from evaluation.similarity import calculate_diversity
from models.epsnet import get_model
from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP, qed
from utils.misc import *
from utils.protein_ligand import PDBProtein, parse_sdf_file
from utils.reconstruct import *
from utils.reconstruct_mdm import (build_molecule, make_mol_openbabel,
                                   mol2smiles)
# from sample import *    # Import everything from `sample.py`
from utils.sample import *
from utils.sample import construct_dataset_pocket
from utils.transforms import *
from utils.data import torchify_dict

from torch_geometric.data import Batch

FOLLOW_BATCH = ['ligand_atom_feature','protein_atom_feature_full']
atomic_numbers_crossdock = torch.LongTensor([1,6,7,8,9,15,16,17])
atomic_numbers_pocket = torch.LongTensor([1,6,7,8,9,15,16,17,34])
atomic_numbers_pdbind = torch.LongTensor([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 23, 26, 27, 29, 33, 34, 35, 44, 51, 53, 78])
P_ligand_element_100 = torch.LongTensor([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 23, 26, 29, 33, 34, 35, 44, 51, 53, 78])
# P_ligand_element_filter = torch.LongTensor([1, 35, 5, 6, 7, 8, 9, 15, 16, 17, 53])
P_ligand_element_filter = torch.LongTensor([1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53])

def save_sdf(mol,sdf_dir,gen_file_name):
    writer = Chem.SDWriter(os.path.join(sdf_dir, gen_file_name))
    writer.write(mol, confId=0)
    writer.close()

def pdb_to_pocket_data(pdb_path, center=0, bbox_size=0):
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
        if resname not in PDBProtein.AA_NAME_NUMBER: continue   # Ignore water, heteros, and non-standard residues.

        element_symb = atom.element.capitalize()
        if element_symb == 'H': continue
        x, y, z = atom.get_coord()
        pos = torch.FloatTensor([x, y, z])
        # if (pos - center).abs().max() > (bbox_size / 2): 
        #     continue

        protein_dict['element'].append( ptable.GetAtomicNumber(element_symb))
        protein_dict['pos'].append(pos)
        protein_dict['is_backbone'].append(atom.get_name() in ['N', 'CA', 'C', 'O'])
        protein_dict['atom_to_aa_type'].append(PDBProtein.AA_NAME_NUMBER[resname])
        
    # if len(protein_dict['element']) == 0:
    #     raise ValueError('No atoms found in the bounding box (center=%r, size=%f).' % (center, bbox_size))

    protein_dict['element'] = torch.LongTensor(protein_dict['element'])
    protein_dict['pos'] = torch.stack(protein_dict['pos'], dim=0)
    protein_dict['is_backbone'] = torch.BoolTensor(protein_dict['is_backbone'])
    protein_dict['atom_to_aa_type'] = torch.LongTensor(protein_dict['atom_to_aa_type'])

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict = protein_dict,
        ligand_dict = {
            'element': torch.empty([0,], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0,], dtype=torch.long),
        }
    )
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str,
                        default='./example/4yhj.pdb')
    parser.add_argument('--mol_file', type=str,
                        default='./example/4yhj_ligand.sdf')
    parser.add_argument('--num_atom', type=int,
                        default=29)
    parser.add_argument('--mask', type=int, nargs='+',)
    parser.add_argument('-build_method', type=str, default='reconstruct',help='build or reconstruct')
    parser.add_argument('--config', type=str)
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--savedir', type=str, default='linker_3-6atoms.pkl')
    parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--save_sdf', type=bool, default=True)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--n_steps', type=int, default=0,
                    help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=float('inf'),
                    help='enable global gradients only when noise is low') # float('inf')
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

    mol_file = args.mol_file

    protein_root = os.path.dirname(args.pdb_path)
    pdb_name = os.path.basename(args.pdb_path)[:4]
    protein_filename = os.path.basename(args.pdb_path)

    rmol = Chem.SDMolSupplier(mol_file)[0]

    # Load configs
    ckpt = torch.load(args.ckpt)
    config = ckpt['config']
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    seed_all(args.seed)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt)),'custom_pdb')

    if args.n_steps == 0:
        args.n_steps = ckpt['config'].model.num_diffusion_timesteps


    # Logging
    # logger = get_logger('sample', log_dir)
    tag = 'result'
    output_dir = get_new_log_dir(log_dir, args.sampling_type+"_linker_"+tag, tag=args.tag)
    logger = get_logger('test', output_dir)

    logger.info(args)
    logger.info(config)

    pocket = False
    logger.info('Loading {} data...'.format(config.dataset.name))
    if config.dataset.name=='crossdock':
        atomic_numbers = atomic_numbers_pocket
        dataset_info = get_dataset_info('crossdock_pocket', False)
        pocket=True
    else:
        if 'filter' in config.dataset.split:
            atomic_numbers = P_ligand_element_filter
        elif '100' in config.dataset.split:
            atomic_numbers = P_ligand_element_100
        else:
            atomic_numbers = atomic_numbers_pdbind

    # shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # dataset_info = get_dataset_info('qm9', False)
    # dataset_info = get_dataset_info('crossdock', False)
    histogram = dataset_info['n_nodes']
    nodes_dist = DistributionNodes(histogram)
    # # Transform
    logger.info('Loading data...')
    protein_featurizer = FeaturizeProteinAtom(config.dataset.name,pocket=pocket)
    ligand_featurizer = FeaturizeLigandAtom(config.dataset.name,pocket=pocket)
    transform = Compose([
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        CountNodesPerGraph(),
        GetAdj(),
    ])
    # # Data
    data = pdb_to_pocket_data(args.pdb_path)
    data = transform(data)

    # Model
    logger.info('Building model...')
    logger.info(config.model['network'])
    print(config.model)
    model = get_model(config.model).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()

    #sample
    # gen_file_name = os.path.basename(args.pdb_path) + '_gen.sdf'
    # print(gen_file_name)
    save_sdf_flag=args.save_sdf
    if save_sdf_flag:
        sdf_dir = os.path.join(os.path.dirname(args.pdb_path),'linker_gen')
        print('sdf idr:', sdf_dir)
        os.makedirs(sdf_dir, exist_ok=True)
    save_results=True
    valid = 0
    stable = 0
    high_affinity=0.0
    num_samples = args.num_samples
    batch_size = 50
    if batch_size>num_samples:
        batch_size=num_samples

    context=None
    smile_list = []
    results = []
    protein_files = []
    sa_list = []
    qed_list=[]
    logP_list = []
    Lipinski_list = []
    vina_score_list = []
    rd_vina_score_list = []
    mol_list = []
    
    start_linker = torchify_dict(parse_sdf_file(mol_file))
    atomic_numbers = torch.LongTensor([1,6,7,8,9,15,16,17,34,119]) # including pocket elements
    start_linker['linker_atom_type'] = start_linker['element'].view(-1, 1) == atomic_numbers.view(1, -1)


    # important: define your own mask
    # mask = [8,9,10,11,12] #8h6p
    # mask = [10,11,12,13,14] #8h6p 
    mask = args.mask
    # mask = [3,4,5,6,20,21]
    element = start_linker['element']
    b = list(range(0, start_linker['element'].size(0)))
    keep_index = torch.tensor([i for i in b if i not in mask])
    # keep_index = torch.tensor([29,10,11,8,9,12,2,3,5,14,15,16])
    mask_index = torch.tensor(mask)
    start_linker['element'] = torch.index_select(start_linker['element'], 0, keep_index)
    start_linker['atom_feature'] = torch.index_select(start_linker['atom_feature'], 0, keep_index)
    start_linker['linker_atom_type'] = torch.index_select(start_linker['linker_atom_type'], 0, keep_index)
    start_linker['pos'] = torch.index_select(start_linker['pos'], 0, keep_index)
    num_points = args.num_atom
  
    rmol = reconstruct_from_generated(start_linker['pos'], start_linker['element'],
                                                              start_linker['atom_feature'])
    r_smile = Chem.MolToSmiles(rmol)

    protein_atom_feature = data.protein_atom_feature.float()
    protein_atom_feature_full = data.protein_atom_feature_full.float()
    # if 'pocket' in args.ckpt:
    #     protein_atom_feature = data.protein_atom_feature.float()
    data_list,_ = construct_dataset_pocket(num_samples*1,batch_size,dataset_info,num_points,num_points,start_linker,None,
    protein_atom_feature,protein_atom_feature_full,data.protein_pos,data.protein_bond_index)

    for n, datas in enumerate(tqdm(data_list)):
        batch = Batch.from_data_list(datas, follow_batch=FOLLOW_BATCH).to(device)

        if num_samples==0:
            break
        with torch.no_grad():
            try:
                pos_gen, pos_gen_traj, atom_type, atom_traj = model.linker_sample(
                    ligand_atom_type=batch.ligand_atom_feature,
                    ligand_pos_init=batch.ligand_pos,
                    ligand_bond_index=batch.ligand_bond_index,
                    ligand_bond_type=None,
                    ligand_num_node=batch.ligand_num_node,
                    ligand_batch=batch.ligand_atom_feature_batch,
                    frag_mask = batch.frag_mask.type(torch.bool),
                    protein_atom_type = batch.protein_atom_feature_full,
                    protein_pos = batch.protein_pos,
                    protein_bond_index = batch.protein_bond_index,
                    protein_backbone_mask = None,
                    protein_batch = batch.protein_atom_feature_full_batch,
                    num_graphs=batch.num_graphs,
                    extend_order=False, # Done in transforms.
                    n_steps=args.n_steps,
                    step_lr=1e-6, #1e-6
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
                # atom_charge_list = atom_charge.reshape(num_samples, -1, 1)
                for m in range(batch_size):
                    try:
                        pos = pos_list[m].detach().cpu()
                        atom_type = atom_list[m].detach().cpu()
                        num_atom_type = len(atomic_numbers)-2 # use the number of atom types in the crossdocked dataset

                        if args.build_method == 'reconstruct':
                            new_element = torch.tensor([atomic_numbers_crossdock[m] for m in torch.argmax(atom_type[:,:num_atom_type],dim=1)])
                            indicators_elements = torch.argmax(atom_type[:,num_atom_type:],dim=1)
                            indicators = torch.zeros([pos.size(0), len(ATOM_FAMILIES)])
                            for i, n in enumerate(indicators_elements):
                                indicators[i,n] = 1
                            gmol = reconstruct_from_generated(pos,new_element,indicators)
                        
                        elif args.build_method == 'build':
                            new_element = torch.argmax(atom_type[:,:num_atom_type],dim=1)

                            gmol = make_mol_openbabel(pos, new_element, dataset_info)
                               
                        # gen_mol = set_rdmol_positions(rdmol, data.ligand_pos)
                        g_smile = mol2smiles(gmol)
                        print("generated smile:", g_smile)
                        
                        if g_smile is not None:
                            FINISHED = True
                            valid+=1
                            if '.' not in g_smile:
                                stable+=1

                            mol_frags = Chem.rdmolops.GetMolFrags(gmol, asMols=True)
                            largest_mol = max(mol_frags, default=gmol, key=lambda m: m.GetNumAtoms())
                            lg_smile = mol2smiles(largest_mol)
                            print("largest generated smile part:", lg_smile)
                            gmol = largest_mol
                            # if len(lg_smile)<4:
                            #     raise MolReconsError()
                            num_samples-=1
                            smile_list.append(g_smile)    
                        else:
                            raise MolReconsError()

                        _,g_sa = compute_sa_score(gmol)
                        print("Generate SA score:", g_sa)
                        sa_list.append(g_sa)

                        g_qed = qed(gmol)
                        print("Generate QED score:", g_qed)
                        qed_list.append(g_qed)

                        g_logP = MolLogP(gmol)
                        print("Generate logP:", g_logP)
                        logP_list.append(g_logP)

                        g_Lipinski = obey_lipinski(gmol)
                        print("Generate Lipinski:", g_Lipinski)
                        Lipinski_list.append(g_Lipinski)

                        vina_task = QVinaDockingTask.from_generated_data(protein_filename,gmol,protein_root)
                        g_vina_results = vina_task.run_sync()
                        g_vina_score = g_vina_results[0]['affinity']
                        print("Generate vina score:", g_vina_score)
                        vina_score_list.append(g_vina_score)
                        if save_sdf_flag:
                            print('save')
                            gen_file_name = '{}_{}_{}.sdf'.format(str(g_vina_score), pdb_name, str(num_samples))
                            # gen_file_name = '{}_{}.sdf'.format(pdb_name, str(num_samples))
                            print(gen_file_name) #str(g_vina_score)+"_"+
                            save_sdf(gmol, sdf_dir, gen_file_name)
                        if save_results:
                            # metrics = {'SA':g_sa,'QED':g_qed,'logP':g_logP,'Lipinski':g_Lipinski,'vina':g_vina_score} 
                            result = {'atom_type':atom_type.detach().cpu(), 
                            'pos':pos.detach().cpu(), 
                            'smile':g_smile, 
                            'mol':gmol,}
                            # 'metric_result':metrics}
                            results.append(result)
                        logger.info('Successfully generate molecule for {}, remaining {} samples generated'.format(pdb_name, num_samples))
                        mol_list.append(gmol)
                        if num_samples==0:
                            break
                    except Exception as e:
                        print(e)

    
            except (FloatingPointError): #,MolReconsError,TypeError,IndexError,OverflowError
                clip_local = 20
                logger.warning('Ignoring, because reconstruction error encountered or retrying with local clipping or vina error.')
                print('Resample the number of the atoms and regenerate!')
    
    logger.info('valid:%d'%valid)
    logger.info('stable:%d'%stable)
    logger.info('mean sa:%f'%mean(sa_list))
    logger.info('mean qed:%f'%mean(qed_list))
    logger.info('mean logP:%f'%mean(logP_list))
    logger.info('mean Lipinski:{}'.format(np.mean(Lipinski_list)))
    # logger.info('diversity:%f'%calculate_diversity(mol_list))
    # print(np.mean(Lipinski_list))
    # logger.info('mean vina:%f'%mean(vina_score_list))
    # logger.info('high affinity:%d'%high_affinity)
    # print(vina_score_list)
    print(Lipinski_list)

    # if save_results:

    #     save_path = os.path.join(os.path.dirname(args.pdb_path), args.savedir)
    #     logger.info('Saving samples to: %s' % save_path)

    #     # save_smile_path = os.path.join(output_dir, 'samples_smile.pkl')

    #     with open(save_path, 'wb') as f:
    #         pickle.dump(results, f)
    #         f.close()

        # save_time_path = os.path.join(output_dir, 'time.pkl')
        # logger.info('Saving time to: %s' % save_path)
        # with open(save_time_path, 'wb') as f:
        #     pickle.dump(time_list, f)
        #     f.close()
