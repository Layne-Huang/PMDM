import argparse
import shutil

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.tensorboard
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from models.epsnet import get_model
from utils.datasets import *
from utils.misc import *
from utils.train import *
# from utils.datasets import ConformationDataset
from utils.transforms import *
from utils.context import prepare_context, compute_mean_mad

# from apex import amp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='crossdock',
                    help='crossdock, pdbind')
parser.add_argument('--config', type=str, help='if it is a directory, resume training')
parser.add_argument('--config_name', type=str, default=None)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--use_mixed_precision', type=bool, default=False)
parser.add_argument('--dp', type=bool, default=True)
parser.add_argument('--resume_iter', type=int, default=500)
parser.add_argument('--logdir', type=str, default='./logs/EGNN/')
parser.add_argument("--context", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv')


def train(it):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(it)
    sum_loss, sum_n = 0, 0
    sum_loss_global, sum_node_global = 0, 0
    sum_loss_local, sum_node_local = 0, 0
    sum_loss_clash = 0
    if args.use_mixed_precision:
        with tqdm(total=len(train_loader), desc='Training') as pbar:
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()

                batch = batch.cuda()
                context = None
                loss_vae_KL = 0.00

                if 'full' in config.model.network:
                    with torch.cuda.amp.autocast():
                        loss = model(
                            batch,
                            context=context,
                            return_unreduced_loss=True
                        )

                    if config.model.vae_context:
                        loss, loss_global, loss_local, loss_node_global, loss_node_local, loss_vae_KL = loss
                        loss_vae_KL = loss_vae_KL.mean().item()
                    else:
                        loss, loss_global, loss_local, loss_node_global, loss_node_local = loss
                    loss = loss.mean()
                    scaler.scale(loss).backward()
                    # Make sure all async allreduces are done
                    optimizer.synchronize()

                    # In-place unscaling of all gradients before weights update
                    scaler.unscale_(optimizer)

                    # orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)

                    scaler.update()

                    sum_loss += loss.item()
                    sum_n += 1
                    sum_loss_global += loss_global.mean().item()
                    sum_loss_local += loss_local.mean().item()
                    sum_node_global += loss_node_global.mean().item()
                    sum_node_local += loss_node_local.mean().item()

                    pbar.set_postfix({'loss': '%.2f' % (loss.item())})
                    pbar.update(1)
                    # print('loss:%.2f'%(sum_loss))
    else:
        with tqdm(total=len(train_loader), desc='Training') as pbar:
            for i, batch in enumerate(train_loader):
                # optimizer_global.synchronize()
                optimizer.zero_grad()

                batch = batch.to(local_rank)

                
                loss_vae_KL = 0.00
                if len(args.context) > 0:
                    context = prepare_context(args.context, batch, property_norms)
                else:
                    context = None
                loss = model(
                    batch,
                    context=context,
                    return_unreduced_loss=True
                )

                if config.model.vae_context:
                    # loss, loss_global, loss_local, loss_node_global, loss_node_local, loss_clash, loss_vae_KL = loss
                    loss, loss_global, loss_local, loss_node_global, loss_node_local, loss_vae_KL = loss
                    loss_vae_KL = loss_vae_KL.mean().item()
                else:
                    # loss, loss_global, loss_local, loss_node_global, loss_node_local, loss_clash = loss
                    loss, loss_global, loss_local, loss_node_global, loss_node_local = loss
                loss = loss.mean()
                loss.backward()


                orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)


                # orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()

                sum_loss += loss.item()
                sum_n += 1
                sum_loss_global += loss_global.mean().item()
                sum_loss_local += loss_local.mean().item()
                sum_node_global += loss_node_global.mean().item()
                sum_node_local += loss_node_local.mean().item()
                # sum_loss_clash += loss_clash.mean().item()
                pbar.set_postfix({'loss': '%.2f' % (loss.item())})
                pbar.update(1)
                # print('loss:%.2f'%(sum_loss))

    train_loss = sum_loss / len(train_sampler)
    # train_loss = metric_average(train_loss, 'avg_loss')
    avg_loss = sum_loss / sum_n
    avg_loss_global = sum_loss_global / sum_n
    avg_loss_local = sum_loss_local / sum_n
    avg_loss_node_global = sum_node_global / sum_n
    avg_loss_node_local = sum_node_local / sum_n
    # avg_loss_clash = sum_loss_clash / sum_n

    if dist.get_rank() == 0:
        logger.info(
            '[Train] Epoch %05d | Loss %.2f | Loss(Global) %.2f | Loss(Local) %.2f | Loss(node_global) %.2f | Loss(node_local) %.2f | Loss(vae_KL) %.2f |Grad %.2f | LR %.6f' % (
                it, avg_loss, avg_loss_global, avg_loss_local, avg_loss_node_global, avg_loss_node_local, loss_vae_KL,
                orig_grad_norm, optimizer.param_groups[0]['lr'],
            ))
        # logger.info('[Train] Epoch %05d | Loss %.2f | horovod_Loss %.2f | Loss(Global) %.2f | Loss(Local) %.2f | Loss(node_global) %.2f | Loss(node_local) %.2f | Loss(clash) %.2f | Loss(vae_KL) %.2f |Grad %.2f | LR %.6f' % (
        #         it, avg_loss, train_loss, avg_loss_global, avg_loss_local, avg_loss_node_global, avg_loss_node_local, avg_loss_clash, loss_vae_KL, orig_grad_norm, optimizer_global.param_groups[0]['lr'],
        #     ))
        writer.add_scalar('train/loss', avg_loss, it)
        writer.add_scalar('train/loss_global', avg_loss_global, it)
        writer.add_scalar('train/loss_local', avg_loss_local, it)
        writer.add_scalar('train/loss_node_global', avg_loss_node_global, it)
        writer.add_scalar('train/loss_node_local', avg_loss_node_local, it)
        # writer.add_scalar('train/loss_clash', avg_loss_clash, it)
        writer.add_scalar('train/loss_vae_KL', loss_vae_KL, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        # writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad_norm', orig_grad_norm, it)
        writer.flush()


def validate(it):
    sum_loss, sum_n = 0, 0
    sum_loss_global, sum_n_global = 0, 0
    sum_loss_local, sum_n_local = 0, 0
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
            batch = batch.to(device)
            if len(args.context) > 0:
                context = prepare_context(args.context, batch, property_norms_val)
            else:
                context = None

            loss = model(
                batch,
                context=context,
                return_unreduced_loss=True
            )

            if config.model.vae_context:
                loss, loss_global, loss_local, loss_node_global, loss_node_local, loss_vae_KL = loss
                # loss, loss_global, loss_local, loss_node_global, loss_node_local, loss_clash, loss_vae_KL = loss
            else:
                loss, loss_global, loss_local, loss_node_global, loss_node_local = loss
                # loss, loss_global, loss_local, loss_node_global, loss_node_local, loss_clash = loss
            sum_loss += loss.sum().item()
            sum_n += loss.size(0)
            sum_loss_global += loss_global.sum().item()
            sum_n_global += loss_global.size(0)
            if 'global' not in config.model.network:
                sum_loss_local += loss_local.sum().item()
                sum_n_local += loss_local.size(0)
    avg_loss = sum_loss / sum_n
    avg_loss_global = sum_loss_global / sum_n_global
    if 'global' not in config.model.network:
        avg_loss_local = sum_loss_local / sum_n_local

    if config.train.scheduler.type == 'plateau':
        scheduler.step(avg_loss)

    else:
        scheduler.step()

    # scheduler.step(avg_loss)

    if dist.get_rank() == 0:
        if 'global' not in config.model.network:
            logger.info('[Validate] Iter %05d | Loss %.6f | Loss(Global) %.6f | Loss(Local) %.6f' % (
                it, avg_loss, avg_loss_global, avg_loss_local,
            ))
        else:
            logger.info('[Validate] Iter %05d | Loss %.6f | Loss(Global) %.6f' % (
                it, avg_loss, avg_loss_global))
        writer.add_scalar('val/loss', avg_loss, it)
        # writer.add_scalar('val/loss_global', avg_loss_global, it)
        # writer.add_scalar('val/loss_local', avg_loss_local, it)
        writer.flush()
    return avg_loss


if __name__ == '__main__':
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    local_rank = int(os.environ["LOCAL_RANK"])
    # set device
    torch.cuda.set_device(local_rank)
    # nccl backend (commonly used)
    dist.init_process_group(backend='nccl')
    # Transfer to the device
    device = torch.device("cuda", local_rank)

    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.

    resume = os.path.isdir(args.config)  # If you wanna resume training, enter the checkpoint dir
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    args.dataset = 'crossdock' if 'crossdock' in config.dataset['name'] else 'pdbind'
    # config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    if args.config_name == None:
        config_name = '{}_exp'.format(
        args.dataset)  
    seed_all(config.train.seed)

    fuse = True if 'fuse' in config.model['network'] else False
    pocket = True if 'pocket' in config.model['network'] else False
    print('pocket:', pocket)
    print('fuse:', fuse)

    # Logging
    if resume:
        config_name = args.config.split('/')[-1]
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume')
        # os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name)
        if not os.path.exists(os.path.join(log_dir, 'models')):
            shutil.copytree('./models', os.path.join(log_dir, 'models'), dirs_exist_ok=True)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train')

    if dist.get_rank() == 0:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        logger.info(args)
        logger.info(config)
        logger.info('Loading %s datasets...' % (args.dataset))

    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    # shutil.copyfile('train_geo.py', os.path.join(log_dir, 'train_geo.py'))

    # Datasets and loaders
    protein_featurizer = FeaturizeProteinAtom(config.dataset.name, pocket=(fuse or pocket))
    ligand_featurizer = FeaturizeLigandAtom(config.dataset.name, pocket=(fuse or pocket))
    masking = get_mask(config.train.transform.mask)
    if fuse:
        transform = Compose([
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
            FeaturizeLigandBond(),
            masking,
            CountNodesPerGraph(),
            # Pos2Distance(),
            GetAdj(),
            Merge_pl()
        ])
    else:
        transform = Compose([
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
            FeaturizeLigandBond(),
            masking,
            CountNodesPerGraph(),
            # Pos2Distance(),
            GetAdj(),
            # Property_loss()
            ])

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # Datasets and loaders
    # logger.info('Loading {} dataset...'.format(config.dataset.name))
    dataset, subsets = get_dataset(
        config=config.dataset,
        transform=transform,
    )
    train_set, val_set = subsets['train'], subsets['test']

    # context for future use   
    if len(args.context) > 0:
        print(f'Conditioning on {args.context}')
        property_norms = compute_mean_mad(train_set, args.context, args.dataset)
        property_norms_val = compute_mean_mad(val_set, args.context, args.dataset)
    else:
        property_norms = None
        context = None

    if fuse:
        follow_batch = ['protein_element', 'ligand_element', 'pocket_element']
    else:
        follow_batch = ['protein_element', 'ligand_element']  # 'pocket_element','protein_element', 'ligand_element',

    collate_exclude_keys = ['ligand_nbh_list']

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,  # config.train.batch_size
        # shuffle = True,
        sampler=train_sampler,
        follow_batch=follow_batch,
        exclude_keys=collate_exclude_keys,
        **kwargs
    )
    val_loader = DataLoader(
        val_set,
        config.train.batch_size,
        # shuffle=False,
        sampler=val_sampler,
        follow_batch=follow_batch,
        exclude_keys=collate_exclude_keys,
        **kwargs
    )

    # Model
    if dist.get_rank() == 0:
        logger.info('Building model...')
    # config.model.context = args.context
    # config.model.num_atom = len(dataset_info['atom_decoder'])+1
    model = get_model(config.model).to(local_rank)
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    # model, optimizer = amp.initialize(model, optimizer, 
    #                                   opt_level='O2')

    start_iter = 1

    if args.use_mixed_precision:
        # Initialize scaler in global scale
        scaler = torch.cuda.amp.GradScaler()

    # Resume from checkpoint
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
        logger.info('Resuming from: %s' % ckpt_path)
        logger.info('Iteration: %d' % start_iter)
        print(ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        config.train.max_iters = start_iter + 500
        start_iter += 1

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    best_val_loss = float('inf')
    for it in range(start_iter, config.train.max_iters + 1):
        start_time = time.time()
        train(it)
        end_time = (time.time() - start_time)
        if dist.get_rank() == 0:
            print('each iteration requires {} s'.format(end_time))
            avg_val_loss = validate(it)
            if it % config.train.val_freq == 0:
                if args.dataset == 'crossdock' or args.dataset == 'pdbind':
                    if avg_val_loss < best_val_loss:
                        ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                        torch.save({
                            'config': config,
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iteration': it,
                            'avg_val_loss': avg_val_loss,
                        }, ckpt_path)
                        print('Successfully saved the model!')
                        # best_val_loss = avg_val_loss
