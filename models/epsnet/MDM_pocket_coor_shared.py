import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from utils.chem import BOND_TYPES
from .diffusion import get_num_embedding
from ..common import (MultiLayerPerceptron, extend_graph_order_radius, get_edges)
from ..encoders import (EGNN_Sparse_Network, SchNetEncoder,
                        SchNetEncoder_protein,
                        get_edge_encoder)
from ..encoders.attention import BasicTransformerBlock
from ..geometry import eq_transform, get_distance


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class MDM_full_pocket_coor_shared(nn.Module):

    def __init__(self, config):
        super(MDM_full_pocket_coor_shared, self).__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)
        # self.hidden_dim = config.hidden_dim
        self.atom_type_input_dim = config.num_atom if 'num_atom' in config else 8  # contains simple tmb or charge or not qm9:5+1(charge) geom: 16+1(charge)
        self.atom_out_dim = config.num_atom if 'num_atom' in config else 8  # contains charge or not
        self.time_emb = config.time_emb if 'time_emb' in config else True
        self.atom_num_emb = config.atom_num_emb if 'atom_num_emb' in config else False
        self.vae_context = config.vae_context if 'vae_context' in config else False
        self.context = config.context if 'context' in config else []
        self.protein_input_dim = config.protein_feature_dim if 'protein_feature_dim' in config else 27
        self.hidden_dim = config.hidden_dim

        self.ligand_emblin = nn.Linear(self.atom_type_input_dim, self.hidden_dim)
        self.protein_emblin = nn.Linear(self.atom_type_input_dim, self.hidden_dim)
        self.atten_layer = BasicTransformerBlock(self.hidden_dim, 4, self.hidden_dim // 4, 0.1, self.hidden_dim)
        '''
        timestep embedding
        '''
        if self.time_emb:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.hidden_dim,
                                self.hidden_dim * 4),
                torch.nn.Linear(self.hidden_dim * 4,
                                self.hidden_dim * 4),
            ])
            # self.temb_proj = torch.nn.Linear(self.hidden_dim*4,
            #                                 self.hidden_dim//4)
            self.temb_proj = torch.nn.Linear(self.hidden_dim * 4,
                                             self.hidden_dim)  # -config.protein_hidden_dim

        '''
        atom numbers embedding
        '''
        if self.atom_num_emb:
            self.nemb = nn.Module()
            self.nemb.dense = nn.ModuleList([
                torch.nn.Linear(self.hidden_dim,
                                self.hidden_dim * 4),
                torch.nn.Linear(self.hidden_dim * 4,
                                self.hidden_dim * 4),
            ])
            # self.temb_proj = torch.nn.Linear(self.hidden_dim*4,
            #                                 self.hidden_dim//4)
            self.nemb_proj = torch.nn.Linear(self.hidden_dim * 4,
                                             self.hidden_dim)  # -config.protein_hidden_dim

        """
        The graph neural network that extracts node-wise features.
        """
        if self.vae_context:
            self.context_encoder = SchNetEncoder(
                hidden_channels=self.hidden_dim,
                num_filters=self.hidden_dim,
                num_interactions=config.num_convs,
                edge_channels=self.edge_encoder_global.out_channels,
                cutoff=10,  # config.cutoff
                smooth=config.smooth_conv,
                input_dim=self.atom_type_input_dim,
                time_emb=False,
                context=True
            )
            self.atom_type_input_dim = self.atom_type_input_dim * 2
        if self.context is not None and type(self.context) is not str:
            ctx_nf = len(self.context)
            self.atom_type_input_dim = self.atom_type_input_dim + ctx_nf

        # Protein encoder
        self.protein_encoder = SchNetEncoder_protein(
            hidden_channels=config.protein_hidden_dim,
            num_filters=config.protein_hidden_dim,
            num_interactions=config.protein_num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=config.encoder_cutoff,  # 10
            input_dim=self.protein_input_dim
        )

        # Ligand encoder
        self.ligand_encoder = SchNetEncoder_protein(
            hidden_channels=config.protein_hidden_dim,
            num_filters=config.protein_hidden_dim,
            num_interactions=config.protein_num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=config.encoder_cutoff,  # 10
            input_dim=self.atom_type_input_dim
        )

        # self.atom_type_input_dim += config.protein_hidden_dim
        # self.hidden_dim += config.protein_hidden_dim
        ### Global encoder (SchNet, EGNN, SphereNet)

        # # SchNet
        # self.encoder_global = SchNetEncoder_pocket(
        #     hidden_channels=self.hidden_dim,
        #     num_filters=self.hidden_dim,
        #     num_interactions=config.num_convs,
        #     edge_channels=self.edge_encoder_global.out_channels,
        #     cutoff=self.config.g_cutoff, #config.g_cutoff
        #     smooth=config.smooth_conv,
        #     input_dim = self.atom_type_input_dim,
        #     time_emb = self.time_emb
        # )

        # EGNN
        self.encoder_global = EGNN_Sparse_Network(
            n_layers=config.num_convs,
            feats_input_dim=self.atom_type_input_dim,
            feats_dim=config.hidden_dim,
            edge_attr_dim=config.hidden_dim,
            m_dim=config.hidden_dim,
            soft_edge=config.soft_edge,
            norm_coors=config.norm_coors
        )

        # # SphereNet
        # self.encoder_global = SphereNet(
        #     cutoff=10, #config.cutoff
        #     hidden_channels=config.hidden_dim,
        #     out_channels=config.hidden_dim
        # )

        ### Local encoder (GIN, SchNet, EGNN)

        # # GIN
        # self.encoder_local = GINEncoder(
        #     hidden_dim=config.hidden_dim,
        #     num_convs=config.num_convs_local,
        #     input_dim = self.atom_type_input_dim,
        #     time_emb = self.time_emb
        # )

        # # SchNet
        # self.encoder_local = SchNetEncoder_pocket(
        #     hidden_channels=self.hidden_dim,
        #     num_filters=self.hidden_dim,
        #     num_interactions=config.num_convs_local,
        #     edge_channels=self.edge_encoder_local.out_channels,
        #     cutoff=config.cutoff, #config.cutoff
        #     smooth=config.smooth_conv,
        #     input_dim = self.atom_type_input_dim,
        #     time_emb = self.time_emb
        # )

        # EGNN
        self.encoder_local = EGNN_Sparse_Network(
            n_layers=config.num_convs_local,
            feats_input_dim=self.atom_type_input_dim,
            feats_dim=config.hidden_dim,
            edge_attr_dim=config.hidden_dim,
            m_dim=config.hidden_dim,
            soft_edge=config.soft_edge,
            norm_coors=config.norm_coors
        )

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1) and node type.
        """

        # if edge attr, then 2*, else 1*
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            1 * 3,
            [self.hidden_dim // 2, self.hidden_dim // 4, 1],
            activation=config.mlp_act
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            1 * 3,
            [self.hidden_dim // 2, self.hidden_dim // 4, 1],
            activation=config.mlp_act
        )

        self.grad_global_node_mlp = MultiLayerPerceptron(
            1 * self.hidden_dim,
            [self.hidden_dim, self.hidden_dim // 2, self.atom_out_dim],
            activation=config.mlp_act
        )

        self.grad_local_node_mlp = MultiLayerPerceptron(
            1 * self.hidden_dim,
            [self.hidden_dim, self.hidden_dim // 2, self.atom_out_dim],
            activation=config.mlp_act
        )
        '''
        Incorporate parameters together
        '''
        self.model_global = nn.ModuleList(
            [self.edge_encoder_global, self.encoder_global, self.grad_global_node_mlp, self.grad_global_dist_mlp])
        self.model_local = nn.ModuleList(
            [self.edge_encoder_local, self.encoder_local, self.grad_local_node_mlp, self.grad_local_dist_mlp])
        # self.model_global = nn.ModuleList([self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp])
        # self.model_local = nn.ModuleList([self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp])

        self.model_type = config.type  # config.type  # 'diffusion'; 'dsm'

        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        ## variances
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

    def net(self, ligand_atom_type, ligand_pos, ligand_bond_index, ligand_bond_type, ligand_batch,
            protein_embeddings, protein_atom_feature, protein_pos, protein_backbone_mask, protein_batch, time_step,
            num_node_ctx=None,
            edge_index=None, edge_type=None, edge_length=None, return_edges=False,
            extend_order=True, extend_radius=True, is_sidechain=None, property_context=None, vae_noise=None,linker_mask=None):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            pos: atom coordinates
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        # print(num_node_ctx)
        N = ligand_atom_type.size(0)
        if not self.time_emb:
            time_step = time_step / self.num_timesteps
            time_emb = time_step.index_select(0, ligand_batch).unsqueeze(1)
            ligand_atom_type = torch.cat([ligand_atom_type, time_emb], dim=1)

        '''
        VAE noise
        '''
        if self.vae_context:
            if self.training:
                ligand_edge_length = get_distance(ligand_pos, ligand_bond_index).unsqueeze(-1)
                m, log_var = self.context_encoder(
                    z=ligand_atom_type,
                    edge_index=ligand_bond_index,
                    edge_length=ligand_edge_length,
                    edge_attr=None,
                    embed_node=False  # default is True
                )
                std = torch.exp(log_var * 0.5)
                z = torch.randn_like(log_var)
                ctx = m + std * z
                ligand_atom_type = torch.cat([ligand_atom_type, ctx], dim=1)
                kl_loss = 0.5 * torch.sum(torch.exp(log_var) + m ** 2 - 1. - log_var)

            else:
                ctx = torch.randn_like(ligand_atom_type)  # N(0,1)
                # ctx = torch.clamp(torch.randn_like(atom_type), min=-3, max=3) # clip N(0,1)
                # ctx = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device) # N(0,3)
                # ctx = torch.zeros_like(atom_type).uniform_(-1,+1) # U(-1,+1)
                # ctx = vae_noise
                ligand_atom_type = torch.cat([ligand_atom_type, ctx], dim=1)
                kl_loss = 0

        if len(self.context) > 0 and self.context is not None and type(self.context) is not str:
            print('Context:', self.context)
            print(type(self.context))
            ligand_atom_type = torch.cat([ligand_atom_type, context], dim=1)

        # ligand_atom_type = torch.cat([ligand_atom_type,protein_ctx],dim=1)
        protein_ctx = scatter_mean(protein_embeddings, protein_batch, dim=0)
        protein_ctx = protein_ctx.index_select(0, ligand_batch)
        context = protein_ctx

        '''
        Time embedding
        '''
        if self.time_emb:
            nonlinearity = nn.ReLU()
            temb = get_num_embedding(time_step, self.config.hidden_dim)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
            temb = self.temb_proj(nonlinearity(temb))  # (G, dim)
            # time_ctx = temb.index_select(0, pocket_batch)
            time_ctx = temb.index_select(0, ligand_batch)
            # context = time_ctx
            context = time_ctx + context

        '''
        Atom numbers embedding
        '''
        if self.atom_num_emb:
            context = context + num_node_ctx

        ligand_atom_feature = self.ligand_encoder(
            node_attr=ligand_atom_type,
            pos=ligand_pos,
            batch=ligand_batch,
        ) + context
        # ligand_atom_feature = self.ligand_emblin(ligand_atom_type)+context
        # protein_atom_feature = self.protein_emblin(protein_atom_feature)
        ligand_atom_feature, protein_embeddings = self.atten_layer(ligand_atom_feature, protein_embeddings)
        # ligand_atom_feature = self.atten_layer(ligand_atom_feature, protein_embeddings)
        # ligand_atom_feature  = self.atten_layer(ligand_atom_feature, protein_atom_feature)

        pocket_atom = torch.cat([ligand_atom_feature, protein_embeddings], dim=0)
        pocket_pos = torch.cat([ligand_pos, protein_pos], dim=0)
        pocket_batch = torch.cat([ligand_batch, protein_batch])
        pocket_mask = torch.cat([linker_mask,torch.zeros(protein_pos.size(0), dtype=torch.bool).to(linker_mask.device)]) if linker_mask is not None else None



        if edge_index is None or edge_type is None or edge_length is None:
            full_bond_type = torch.ones(ligand_bond_index.size(1), dtype=torch.long).to(ligand_bond_index.device)

            # Construct local and global edges
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=N,
                pos=ligand_pos,
                edge_index=ligand_bond_index,
                edge_type=full_bond_type,
                batch=ligand_batch,
                order=self.config.edge_order,
                cutoff=self.config.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=is_sidechain,
            )
            edge_length = get_distance(ligand_pos, edge_index).unsqueeze(-1)  # (E, 1)
            ligand_bond_index = None #comment if fix the edge

        local_pocket_edge = get_edges(pocket_pos, pocket_batch, ligand_batch, self.config.cutoff, self.config.cutoff,ligand_bond_index) #ligand_bond_index
        global_pocket_edge = get_edges(pocket_pos, pocket_batch, ligand_batch, self.config.g_cutoff,
                                       self.config.g_cutoff)  # self.config.g_cutoff
        local_pocket_edge_length = get_distance(pocket_pos, local_pocket_edge).unsqueeze(-1)
        global_pocket_edge_length = get_distance(pocket_pos, global_pocket_edge).unsqueeze(-1)

        if ligand_bond_type is not None:
            local_edge_mask = is_local_edge(ligand_bond_type)
        else:
            local_edge_mask = is_radius_edge(edge_type)
        
        

        # Emb time_step
        # with the parameterization of NCSNv2
        # DDPM loss implicit handle the noise variance scale conditioning

        # if self.time_emb:
        #     # node2graph = ligand_batch
        #     node2graph = pocket_batch
        #     # edge2graph = node2graph.index_select(0, edge_index[0])
        #     global_edge2graph = node2graph.index_select(0, global_pocket_edge[0])
        #     g_temb_edge = temb.index_select(0, global_edge2graph)
        #     g_protein_edge = protein_ctx.index_select(0, global_edge2graph)
        #     # ptemb_edge = torch.cat([protein_edge,temb_edge],dim=1)
        #     g_ptemb_edge = g_protein_edge+g_temb_edge
        #     local_edge2graph = node2graph.index_select(0, local_pocket_edge[0])
        #     l_temb_edge = temb.index_select(0, local_edge2graph)
        #     l_protein_edge = protein_ctx.index_select(0, local_edge2graph)
        #     # ptemb_edge = torch.cat([protein_edge,temb_edge],dim=1)
        #     l_ptemb_edge = l_protein_edge+l_temb_edge

        # Encoding global

        # edge_attr_global = self.edge_encoder_global(
        #     edge_length=edge_length,
        #     edge_type=edge_type
        # )   # Embed edges
        edge_attr_global = self.edge_encoder_global(
            edge_length=global_pocket_edge_length,
            edge_type=None
        )  # Embed edges
        # if self.time_emb:
        #     # edge_attr_global += temb_edge
        #     edge_attr_global += g_ptemb_edge

        # # SphereNet
        # _,node_attr_global,_ = self.encoder_global(
        #     z=atom_type,
        #     pos=pos,
        #     batch=batch
        # )

        # # SchNet
        # # print(self.atom_type_input_dim)
        # node_attr_global,pos_attr_global = self.encoder_global(
        #     z=pocket_atom,
        #     r=pocket_pos,
        #     edge_index=global_pocket_edge,
        #     edge_length=global_pocket_edge_length,
        #     edge_attr=edge_attr_global,
        #     ligand_batch=ligand_batch,
        #     ctx=context,
        #     embed_node = False # default is True
        # )

        # EGNN
        node_attr_global, pos_attr_global = self.encoder_global(
            z=pocket_atom,
            pos=pocket_pos,
            edge_index=global_pocket_edge,
            edge_attr=edge_attr_global,
            batch=pocket_batch,
            ligand_batch=ligand_batch,
            context=context,
            linker_mask=pocket_mask
        )

        # Encoding local
        # edge_attr_local = self.edge_encoder_local(
        #     edge_length=edge_length,
        #     edge_type=edge_type
        # )   # Embed edges
        edge_attr_local = self.edge_encoder_local(
            edge_length=local_pocket_edge_length,
            edge_type=None
        )  # Embed edges
        # if self.time_emb:
        #     # edge_attr_local += temb_edge
        #     edge_attr_local += l_ptemb_edge

        # # GIN
        # node_attr_local = self.encoder_local(
        #     z=ligand_atom_type,
        #     edge_index=edge_index[:, local_edge_mask],
        #     edge_attr=edge_attr_local[local_edge_mask],
        # )

        # EGNN
        node_attr_local, pos_attr_local = self.encoder_local(
            z=pocket_atom,
            pos=pocket_pos,
            edge_index=local_pocket_edge,
            edge_attr=edge_attr_local,
            batch=pocket_batch,
            ligand_batch=ligand_batch,
            context=context,
            linker_mask=pocket_mask
        )

        # # Schnet
        # node_attr_local, pos_attr_local = self.encoder_local(
        #     z=pocket_atom,
        #     r=pocket_pos,
        #     edge_index=local_pocket_edge,
        #     edge_length=local_pocket_edge_length,
        #     edge_attr=edge_attr_local,
        #     ligand_batch=ligand_batch,
        #     ctx=context,
        #     embed_node = False # default is True
        # )

        node_score_global = self.grad_global_node_mlp(node_attr_global)
        node_score_local = self.grad_local_node_mlp(node_attr_local)

        if self.vae_context:
            return pos_attr_global, pos_attr_local, node_score_global, node_score_local, edge_index, edge_type, edge_length, local_edge_mask, kl_loss
        else:
            return pos_attr_global, pos_attr_local, node_score_global, node_score_local, edge_index, edge_type, edge_length, local_edge_mask,

    def forward(self, batch, context=None, return_unreduced_loss=False, return_unreduced_edge_loss=False,
                extend_order=True, extend_radius=True, is_sidechain=None):

        ligand_atom_type = batch.ligand_atom_feature.float()  # full feature or not
        # print(ligand_atom_type)
        ligand_pos = batch.ligand_pos
        ligand_bond_index = batch.ligand_bond_index
        ligand_bond_type = batch.ligand_bond_type
        ligand_batch = batch.ligand_element_batch
        ligand_num_atom = batch.num_nodes_per_graph
        protein_atom_feature = batch.protein_atom_feature.float()  # full feature or not
        protein_atom_feature_full = batch.protein_atom_feature_full.float()  # full feature or not
        protein_pos = batch.protein_pos
        protein_batch = batch.protein_element_batch
        protein_backbone_mask = batch.protein_is_backbone

        N = ligand_atom_type.size(0)
        node2graph = ligand_batch
        num_graphs = node2graph[-1] + 1

        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        # Sample noise levels
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=ligand_pos.device)

        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]

        a = self.alphas.index_select(0, time_step)  # (G, )
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)

        # Independently noise
        pos_noise = torch.randn(size=ligand_pos.size(), device=ligand_pos.device)
        atom_noise = torch.randn(size=ligand_atom_type.size(), device=ligand_atom_type.device)

        # # Jointly noise
        # noise = torch.randn(size=(ligand_pos.size(0),ligand_pos.size(1)+ligand_atom_type.size(1)),device=ligand_pos.device)
        # pos_noise = noise[:,:ligand_pos.size(1)]
        # atom_noise = noise[:,ligand_pos.size(1):]

        # Move the ligand to COM, and move the protein to the ligand-COM
        ligand_pos, protein_pos = center_pos_pl(ligand_pos, protein_pos, ligand_batch, protein_batch)
        ## Perterb pos 
        ligand_pos_perturbed = ligand_pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        # Move to the COM again
        ligand_pos_perturbed, protein_pos = center_pos_pl(ligand_pos_perturbed, protein_pos, ligand_batch,
                                                          protein_batch)
        ## Perterb atom
        ligand_atom_perturbed = a_pos.sqrt() * ligand_atom_type + (1.0 - a_pos).sqrt() * atom_noise

        ## scaling features from EDM
        # atom_type = torch.cat([atom_type[:,:-1]/4,atom_type[:,-1:]/10], dim=1)
        # atom_perturbed = atom_type + atom_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()

        # vae_noise = torch.randn_like(ligand_atom_type) # N(0,1)
        # vae_noise = torch.clamp(torch.randn_like(atom_type), min=-3, max=3) # clip N(0,1)
        # vae_noise = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device) # N(0,3)
        # vae_noise = torch.zeros_like(atom_type).uniform_(-1,+1) # U(-1,1)

        # '''
        # Protein embedding
        # '''
        # protein_ctx = self.protein_encoder(
        #     node_attr = protein_atom_feature,
        #     pos = protein_pos,
        #     batch = protein_batch,        
        # )
        # protein_ctx = scatter_mean(protein_ctx, protein_batch, dim=0)
        # protein_ctx = protein_ctx.index_select(0, ligand_batch)

        '''
        Protein embedding
        '''
        # protein_pos = center_pos(protein_pos,protein_batch)
        protein_ctx = self.protein_encoder(
            node_attr=protein_atom_feature_full,
            pos=protein_pos,
            batch=protein_batch,
        )
        # protein_ctx = scatter_mean(protein_ctx, protein_batch, dim=0)

        '''
        Atom numbers embedding
        '''
        num_node_ctx = None
        if self.atom_num_emb:
            nonlinearity = nn.ReLU()
            nemb = get_num_embedding(ligand_num_atom, self.config.hidden_dim)
            nemb = self.nemb.dense[0](nemb)
            nemb = nonlinearity(nemb)
            nemb = self.nemb.dense[1](nemb)
            nemb = self.nemb_proj(nonlinearity(nemb))  # (G, dim)
            num_node_ctx = nemb.index_select(0, ligand_batch)

        net_out = self.net(
            ligand_atom_type=ligand_atom_perturbed,
            ligand_pos=ligand_pos_perturbed,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
            ligand_batch=ligand_batch,
            protein_embeddings=protein_ctx,
            protein_atom_feature=protein_atom_feature,
            protein_pos=protein_pos,
            protein_backbone_mask=protein_backbone_mask,
            protein_batch=protein_batch,
            time_step=time_step,
            num_node_ctx=num_node_ctx,
            return_edges=True,
            extend_order=extend_order,
            extend_radius=extend_radius,
            is_sidechain=is_sidechain,
            property_context=context,
            vae_noise=None
        )  # (E_global, 1), (E_local, 1)

        if self.vae_context:
            pos_eq_global, pos_eq_local, node_score_global, node_score_local, edge_index, edge_type, edge_length, local_edge_mask = net_out[
                                                                                                                                    :-1]
        else:
            pos_eq_global, pos_eq_local, node_score_global, node_score_local, edge_index, edge_type, edge_length, local_edge_mask = net_out
        edge2graph = node2graph.index_select(0, edge_index[0])
        # Compute sigmas_edge
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # Compute original and perturbed distances
        d_gt = get_distance(ligand_pos, edge_index).unsqueeze(-1)  # (E, 1)
        d_perturbed = edge_length

        train_edge_mask = is_train_edge(edge_index, is_sidechain)
        d_perturbed = torch.where(train_edge_mask.unsqueeze(-1), d_perturbed, d_gt)

        if self.config.edge_encoder == 'gaussian':
            # Distances must be greater than 0 
            d_sgn = torch.sign(d_perturbed)
            d_perturbed = torch.clamp(d_perturbed * d_sgn, min=0.01, max=float('inf'))

        d_target = (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()  # (E_global, 1), denoising direction
        # d_target = (d_perturbed - d_gt* a_edge.sqrt()) / (1.0 - a_edge).sqrt()   # (E_global, 1), denoising direction
        # d_target = -1*(d_perturbed - d_gt* a_edge.sqrt()) / (1.0 - a_edge)

        global_mask = torch.logical_and(
            torch.logical_or(torch.logical_and(d_perturbed > self.config.cutoff, d_perturbed <= self.config.g_cutoff),
                             local_edge_mask.unsqueeze(-1)),
            ~local_edge_mask.unsqueeze(-1)
        )

        # score matching
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        target_pos_global = eq_transform(target_d_global, ligand_pos_perturbed, edge_index, edge_length)

        # score matching
        target_pos_local = eq_transform(d_target[local_edge_mask], ligand_pos_perturbed, edge_index[:, local_edge_mask],
                                        edge_length[local_edge_mask])
        loss_pos = F.mse_loss(pos_eq_global + pos_eq_local, target_pos_global + target_pos_local, reduction='none')
        loss_pos = 1 * torch.sum(loss_pos, dim=-1, keepdim=True)

        loss_node = F.mse_loss(node_score_global + node_score_local, atom_noise, reduction='none')
        loss_node = 1 * torch.sum(loss_node, dim=-1, keepdim=True)
        # loss for atomic eps regression
        # loss = loss_global + loss_local + loss_node
        if self.vae_context:
            vae_KL_loss = net_out[-1]
            loss = loss_pos + loss_node + vae_KL_loss
        else:
            loss = loss_pos + loss_node
        # loss_pos = scatter_add(loss_pos.squeeze(), node2graph)  # (G, 1)

        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            if self.vae_context:
                return loss, loss_pos, loss_pos, loss_node, loss_node, vae_KL_loss
            return loss, loss_pos, loss_pos, loss_node, loss_node
        else:
            return loss

    def langevin_dynamics_sample(self, ligand_atom_type, ligand_pos_init, ligand_bond_index, ligand_bond_type,
                                 ligand_num_node, ligand_batch,
                                 protein_atom_type, protein_atom_feature_full, protein_pos, protein_backbone_mask,
                                 protein_batch,
                                 num_graphs, context, extend_order, extend_radius=True,
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0,
                                 is_sidechain=None,
                                 global_start_sigma=float('inf'), local_start_sigma=float('inf'), w_global_pos=0.2,
                                 w_global_node=0.2, w_local_pos=0.2, w_local_node=0.2, w_reg=1.0,white_noise=True, **kwargs):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        atom_traj = []

        with torch.no_grad():
            skip = self.num_timesteps // n_steps
            print(skip)
            seq = range(0, self.num_timesteps, skip)
            

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps
            # seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            protein_ori = protein_pos
            protein_com = scatter_mean(protein_pos, protein_batch, dim=0)
            # ligand_pos_init = center_pos(ligand_pos_init, ligand_batch)
            if white_noise:
                ligand_pos, protein_pos = center_pos_pl(ligand_pos_init + protein_com[ligand_batch], protein_pos,
                                                    ligand_batch, protein_batch)
            else:
                ligand_pos, protein_pos = center_pos_pl(ligand_pos_init, protein_pos, ligand_batch, protein_batch)

            # ligand_pos = center_pos(ligand_pos_init, ligand_batch)
            # pos = center_pos(pos_init* sigmas[-1], batch)
            # pos = center_pos(pos_init, batch)* sigmas[-1] 

            # VAE noise
            vae_noise = torch.zeros_like(ligand_atom_type).uniform_(-1, +1)
            # vae_noise = torch.randn_like(atom_type)
            # vae_noise = torch.clamp(torch.randn_like(atom_type), min=-3, max=3)
            # vae_noise = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device)

            '''
            Protein embedding
            '''
            protein_ctx = self.protein_encoder(
                node_attr=protein_atom_feature_full,
                pos=protein_pos,
                batch=protein_batch,
            )
            # protein_ctx = scatter_mean(protein_ctx, protein_batch, dim=0)
            # protein_ctx = protein_ctx.index_select(0, ligand_batch)

            '''
            Atom numbers embedding
            '''
            num_node_ctx = None
            if 'atom_num_emb' not in self.__dict__.keys():
                self.atom_num_emb = False
            if self.atom_num_emb:
                nonlinearity = nn.ReLU()
                nemb = get_num_embedding(ligand_num_node, self.config.hidden_dim)
                nemb = self.nemb.dense[0](nemb)
                nemb = nonlinearity(nemb)
                nemb = self.nemb.dense[1](nemb)
                nemb = self.nemb_proj(nonlinearity(nemb))  # (G, dim)
                num_node_ctx = nemb.index_select(0, ligand_batch)

            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)

                net_out = self.net(
                    ligand_atom_type=ligand_atom_type,
                    ligand_pos=ligand_pos,
                    ligand_bond_index=ligand_bond_index,
                    ligand_bond_type=ligand_bond_type,
                    ligand_batch=ligand_batch,
                    protein_embeddings=protein_ctx,
                    time_step=t,
                    num_node_ctx=num_node_ctx,
                    protein_atom_feature=protein_atom_type,
                    protein_pos=protein_pos,
                    protein_backbone_mask=protein_backbone_mask,
                    protein_batch=protein_batch,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain,
                    property_context=context,
                    vae_noise=None
                )  # (E_global, 1), (E_local, 1)
                if self.vae_context:
                    pos_eq_global, pos_eq_local, node_score_global, node_score_local, edge_index, edge_type, edge_length, local_edge_mask = net_out[
                                                                                                                                            :-1]
                else:
                    pos_eq_global, pos_eq_local, node_score_global, node_score_local, edge_index, edge_type, edge_length, local_edge_mask = net_out
                # Local float('inf')

                # local_start_sigma = random.uniform(0.1,1)
                if sigmas[i] < local_start_sigma:
                    node_eq_local = pos_eq_local
                    if clip_local is not None:
                        node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                else:
                    node_eq_local = 0
                    node_score_local = 0

                # Global
                if sigmas[i] < global_start_sigma:
                    node_eq_global = pos_eq_global
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                    node_score_global = 0
                # Sum
                eps_pos = w_local_pos * node_eq_local + w_global_pos * node_eq_global  # + eps_pos_reg * w_reg
                eps_node = w_local_node * node_score_local + w_global_node * node_score_global
                # eps_pos = 3 * node_eq_local + 1 * node_eq_global  # + eps_pos_reg * w_reg
                # eps_node = 1 * node_score_local + 1 * node_score_global
                # Update

                sampling_type = kwargs.get("sampling_type", 'ddpm_noisy')  # types: generalized, ddpm_noisy, ld

                noise = torch.randn_like(ligand_pos)
                noise_node = torch.randn_like(ligand_atom_type)  # center_pos(torch.randn_like(pos), batch)
                b = self.betas
                t = t[0]
                next_t = (torch.ones(1) * j).to(ligand_pos.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                if sampling_type == 'generalized' or sampling_type == 'ddpm_noisy':
                    if sampling_type == 'generalized':
                        eta = kwargs.get("eta", 1.)
                        et = -eps_pos

                        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        c2 = ((1 - at_next) - c1 ** 2).sqrt()

                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 3 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
                        step_size_pos = step_size_pos_ld if step_size_pos_ld < step_size_pos_generalized else step_size_pos_generalized

                        step_size_noise_ld = torch.sqrt((step_lr * (sigmas[i] / 0.01) ** 2) * 2)
                        step_size_noise_generalized = 5 * (c1 / at_next.sqrt())
                        step_size_noise = step_size_noise_ld if step_size_noise_ld < step_size_noise_generalized else step_size_noise_generalized

                        # w = 1+2 * i/self.num_timesteps
                        w = 1

                        eps_node = eps_node / (1 - at).sqrt()
                        pos_next = ligand_pos - et * step_size_pos + w * noise * step_size_noise
                        atom_next = ligand_atom_type - eps_node * step_size_pos + w * noise_node * step_size_noise
                    elif sampling_type == 'ddpm_noisy':
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        # pos0_from_e = (1.0 / at).sqrt() * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # pos0_from_e = 1 * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # mean_eps = (
                        #     (atm1.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_pos
                        # ) / (1.0 - at)
                        # mean = mean_eps
                        # mean = pos-beta_t/(1-at).sqrt()*e
                        mean = (ligand_pos - beta_t * e) / (1 - beta_t).sqrt()
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        pos_next = mean + mask * torch.exp(
                            0.5 * logvar) * noise  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ

                        e = eps_node
                        node0_from_e = (1.0 / at).sqrt() * ligand_atom_type - (1.0 / at - 1).sqrt() * e
                        mean_eps = (
                                           (atm1.sqrt() * beta_t) * node0_from_e + (
                                           (1 - beta_t).sqrt() * (1 - atm1)) * ligand_atom_type
                                   ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        atom_next = mean + mask * torch.exp(
                            0.5 * logvar) * noise_node  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ
                elif sampling_type == 'ld':
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = ligand_pos + step_size * eps_pos / sigmas[i] + noise * torch.sqrt(step_size * 2)
                    eps_node = eps_node / (1 - at).sqrt()
                    atom_next = ligand_atom_type - step_size * eps_node / sigmas[i] + noise_node * torch.sqrt(
                        step_size * 2)
                else:
                    raise ValueError('Unknown sampling type, it should be one of [generalized, ddpm_noisy, ld]')

                ligand_pos = pos_next
                ligand_atom_type = atom_next

                if torch.isnan(ligand_pos).any():
                    print('NaN detected. Please restart.')
                    print(node_eq_local)
                    print(node_eq_global)
                    raise FloatingPointError()
                # ligand_pos = center_pos(ligand_pos, ligand_batch)
                ligand_pos, protein_pos = center_pos_pl(ligand_pos, protein_pos, ligand_batch, protein_batch)
                if clip_pos is not None:
                    ligand_pos = torch.clamp(ligand_pos, min=-clip_pos, max=clip_pos)
                protein_t = scatter_mean(protein_pos, protein_batch, dim=0)
                move_dist = protein_com - protein_t
                ligand_pos_fix = ligand_pos + move_dist[ligand_batch]
                pos_traj.append(ligand_pos_fix.clone().cpu())
                atom_traj.append(ligand_atom_type.clone().cpu())
        protein_final = scatter_mean(protein_pos, protein_batch, dim=0)
        protein_pos = protein_pos + (protein_com - protein_final)[protein_batch]
        ligand_pos = ligand_pos + (protein_com - protein_final)[ligand_batch]
        # atom_type = torch.cat([atom_type[:,:-1]*4,atom_type[:,-1:]*10], dim=1)
        return ligand_pos, pos_traj, ligand_atom_type, atom_traj

    # for lead optimization
    def inpainting_sample(self, ligand_atom_type, ligand_pos_init, ligand_bond_index, ligand_bond_type, ligand_num_node,
                          ligand_batch, frag_mask, protein_atom_type, protein_pos,
                          protein_backbone_mask, protein_batch, num_graphs, context, extend_order, extend_radius=True,
                          n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, is_sidechain=None,
                          global_start_sigma=float('inf'), local_start_sigma=float('inf'), w_global_pos=0.2,
                          w_global_node=0.2, w_local_pos=0.2, w_local_node=0.2, w_reg=1.0, **kwargs):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        atom_traj = []

        with torch.no_grad():
            skip = self.num_timesteps // n_steps
            seq = range(0, self.num_timesteps, skip)

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps
            # seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            linker_mask = ~frag_mask #This is the part that we need to generate
            frag_pos = ligand_pos_init[frag_mask, :]
            ligand_pos = ligand_pos_init
            linker_pos = ligand_pos_init[linker_mask, :]

            frag_batch = ligand_batch[frag_mask]
            
            protein_pos_ori = protein_pos
            protein_com = scatter_mean(protein_pos, protein_batch, dim=0)
            # ligand_pos = ligand_pos_init + protein_com[ligand_batch] #important

            # subtract the center of mass of the pocket (COM)
            ligand_pos, protein_pos = center_pos_lp(ligand_pos, protein_pos, ligand_batch, protein_batch) #important
            # frag_pos = ligand_pos[frag_mask, :]
            # Recover the linker postion
            ligand_pos[linker_mask] = linker_pos #important
            original_atom_type = ligand_atom_type.clone()

            # VAE noise
            # vae_noise = torch.zeros_like(ligand_atom_type).uniform_(-1, +1)
            # vae_noise = torch.randn_like(atom_type)
            # vae_noise = torch.clamp(torch.randn_like(atom_type), min=-3, max=3)
            # vae_noise = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device)

            '''
            Protein embedding
            '''
            protein_ctx = self.protein_encoder(
                node_attr=protein_atom_type,
                pos=protein_pos,
                batch=protein_batch,
            )
            # protein_ctx = scatter_mean(protein_ctx, protein_batch, dim=0)
            # protein_ctx = protein_ctx.index_select(0, ligand_batch)

            '''
            Atom numbers embedding
            '''
            num_node_ctx = None
            if 'atom_num_emb' not in self.__dict__.keys():
                self.atom_num_emb = False
            if self.atom_num_emb:
                nonlinearity = nn.ReLU()
                nemb = get_num_embedding(ligand_num_node, self.config.hidden_dim)
                nemb = self.nemb.dense[0](nemb)
                nemb = nonlinearity(nemb)
                nemb = self.nemb.dense[1](nemb)
                nemb = self.nemb_proj(nonlinearity(nemb))  # (G, dim)
                num_node_ctx = nemb.index_select(0, ligand_batch)

            # linker_pos = ligand_pos_init[linker_mask,:]
            # linker_atom = ligand_atom_type[linker_mask,:]
            # original_atom_type = ligand_atom_type
            ligand_pos_ori = ligand_pos.clone()
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)
                b = self.betas
                at = compute_alpha(b, t[0].long())

                pos_noise = torch.randn(size=ligand_pos[frag_mask, :].size(), device=ligand_pos.device)
                atom_noise = torch.randn(size=ligand_atom_type[frag_mask, :].size(), device=ligand_atom_type.device)
                mask = 1 - (t[0] == 0).float()

                # linker_pos = ligand_pos
                frag_pos = ligand_pos[frag_mask, :] #important
                frag_atom_type = ligand_atom_type[frag_mask, :] #important

                # # fix the atom type
                # frag_pos_perturbed = frag_pos + pos_noise * (1.0 - at).sqrt() / at.sqrt() * mask #important
                # # ligand_pos = torch.cat([linker_pos_perturbed,ligand_pos[~linker_mask,:]])
                # ligand_pos[frag_mask] = frag_pos_perturbed #important

                # ligand_atom_type = torch.cat([linker_atom_perturbed,ligand_atom_type[~linker_mask,:]])
                frag_atom_perturbed = at.sqrt() * frag_atom_type + (1.0 - at).sqrt() * atom_noise * mask
                ligand_atom_type[frag_mask] = frag_atom_perturbed

                frag_pos_perturbed = frag_pos + pos_noise * (1.0 - at).sqrt() / at.sqrt() * mask
                ligand_pos[frag_mask] = frag_pos_perturbed

                # ligand_pos,protein_pos = center_pos_pl(ligand_pos, protein_pos, ligand_batch, protein_batch)
                net_out = self.net(
                    ligand_atom_type=ligand_atom_type,
                    ligand_pos=ligand_pos,
                    ligand_bond_index=ligand_bond_index,
                    ligand_bond_type=ligand_bond_type,
                    ligand_batch=ligand_batch,
                    protein_embeddings=protein_ctx,
                    time_step=t,
                    num_node_ctx=num_node_ctx,
                    protein_atom_feature=protein_atom_type[:, :10],
                    protein_pos=protein_pos,
                    protein_backbone_mask=protein_backbone_mask,
                    protein_batch=protein_batch,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain,
                    property_context=context,
                    vae_noise=None,
                )  # (E_global, 1), (E_local, 1)
                if self.vae_context:
                    pos_eq_global, pos_eq_local, node_score_global, node_score_local, edge_index, edge_type, edge_length, local_edge_mask = net_out[
                                                                                                                                            :-1]
                else:
                    pos_eq_global, pos_eq_local, node_score_global, node_score_local, edge_index, edge_type, edge_length, local_edge_mask = net_out
                # Local float('inf')

                # local_start_sigma = random.uniform(0.1,1)
                if sigmas[i] < local_start_sigma:
                    node_eq_local = pos_eq_local
                    if clip_local is not None:
                        node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                else:
                    node_eq_local = 0
                    node_score_local = 0

                # Global
                if sigmas[i] < global_start_sigma:
                    node_eq_global = pos_eq_global
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                    node_score_global = 0
                # Sum
                eps_pos = w_local_pos * node_eq_local + w_global_pos * node_eq_global  # + eps_pos_reg * w_reg
                eps_node = w_local_node * node_score_local + w_global_node * node_score_global
                # eps_pos = 3 * node_eq_local + 1 * node_eq_global  # + eps_pos_reg * w_reg
                # eps_node = 1 * node_score_local + 1 * node_score_global
                # Update

                sampling_type = kwargs.get("sampling_type", 'ddpm_noisy')  # types: generalized, ddpm_noisy, ld

                noise = torch.randn_like(ligand_pos)
                noise_node = torch.randn_like(ligand_atom_type)  # center_pos(torch.randn_like(pos), batch)

                t = t[0]
                next_t = (torch.ones(1) * j).to(ligand_pos.device)
                # at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                if sampling_type == 'generalized' or sampling_type == 'ddpm_noisy':
                    if sampling_type == 'generalized':
                        eta = kwargs.get("eta", 1.)
                        et = -eps_pos

                        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        c2 = ((1 - at_next) - c1 ** 2).sqrt()

                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 3 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
                        step_size_pos = step_size_pos_ld if step_size_pos_ld < step_size_pos_generalized else step_size_pos_generalized

                        step_size_noise_ld = torch.sqrt((step_lr * (sigmas[i] / 0.01) ** 2) * 2)
                        step_size_noise_generalized = 5 * (c1 / at_next.sqrt())
                        step_size_noise = step_size_noise_ld if step_size_noise_ld < step_size_noise_generalized else step_size_noise_generalized

                        # w = 1+2 * i/self.num_timesteps
                        w = 1

                        eps_node = eps_node / (1 - at).sqrt()
                        pos_next = ligand_pos - et * step_size_pos + w * noise * step_size_noise
                        atom_next = ligand_atom_type - eps_node * step_size_pos + w * noise_node * step_size_noise
                    elif sampling_type == 'ddpm_noisy':
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        # pos0_from_e = (1.0 / at).sqrt() * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # pos0_from_e = 1 * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # mean_eps = (
                        #     (atm1.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_pos
                        # ) / (1.0 - at)
                        # mean = mean_eps
                        # mean = pos-beta_t/(1-at).sqrt()*e
                        mean = (ligand_pos - beta_t * e) / (1 - beta_t).sqrt()
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        pos_next = mean + mask * torch.exp(
                            0.5 * logvar) * noise  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ

                        e = eps_node
                        node0_from_e = (1.0 / at).sqrt() * ligand_atom_type - (1.0 / at - 1).sqrt() * e
                        mean_eps = (
                                           (atm1.sqrt() * beta_t) * node0_from_e + (
                                           (1 - beta_t).sqrt() * (1 - atm1)) * ligand_atom_type
                                   ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        atom_next = mean + mask * torch.exp(
                            0.5 * logvar) * noise_node  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ
                elif sampling_type == 'ld':
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = ligand_pos + step_size * eps_pos / sigmas[i] + noise * torch.sqrt(step_size * 2)
                    eps_node = eps_node / (1 - at).sqrt()
                    atom_next = ligand_atom_type - step_size * eps_node / sigmas[i] + noise_node * torch.sqrt(
                        step_size * 2)
                else:
                    raise ValueError('Unknown sampling type, it should be one of [generalized, ddpm_noisy, ld]')

                ligand_pos = pos_next #important
                ligand_atom_type = atom_next

                if torch.isnan(ligand_pos).any():
                    print('NaN detected. Please restart.')
                    print(node_eq_local)
                    print(node_eq_global)
                    raise FloatingPointError()
                # ligand_pos = center_pos(ligand_pos, ligand_batch) 
                # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])
                ligand_pos[frag_mask] = frag_pos #important
                # ligand_pos = ligand_pos_ori
                # ligand_atom_type = torch.cat([linker_atom_type,ligand_atom_type[~linker_mask,:]])
                ligand_atom_type[frag_mask] = frag_atom_type #important
                # ligand_atom_type = original_atom_type.clone() #fix the atom type
                ligand_pos, protein_pos = center_pos_pl(ligand_pos, protein_pos, ligand_batch, protein_batch) #important
                # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])
                # ligand_atom_type = torch.cat([linker_atom,ligand_atom_type[~linker_mask,:]])
                if clip_pos is not None:
                    ligand_pos = torch.clamp(ligand_pos, min=-clip_pos, max=clip_pos)

                protein_t = scatter_mean(protein_pos, protein_batch, dim=0)
                move_dist = protein_com - protein_t
                ligand_pos_fix = ligand_pos + move_dist[ligand_batch]
                pos_traj.append(ligand_pos_fix.clone().cpu())
                atom_traj.append(ligand_atom_type.clone().cpu())
        protein_final = scatter_mean(protein_pos, protein_batch, dim=0)
        # protein_final = protein_pos
        protein_pos = protein_pos + (protein_com - protein_final)[protein_batch]
        ligand_pos = ligand_pos + (protein_com - protein_final)[ligand_batch] #important
        print(torch.equal(protein_pos_ori, protein_pos))
        # ligand_pos = ligand_pos_init
        # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])

        # atom_type = torch.cat([atom_type[:,:-1]*4,atom_type[:,-1:]*10], dim=1)
        return ligand_pos, pos_traj, ligand_atom_type, atom_traj

    # for linker sample    
    def linker_sample(self, ligand_atom_type, ligand_pos_init, ligand_bond_index, ligand_bond_type, ligand_num_node,
                          ligand_batch, frag_mask, protein_atom_type, protein_pos,
                          protein_backbone_mask, protein_batch, num_graphs, context, extend_order, extend_radius=True,
                          n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, is_sidechain=None,
                          global_start_sigma=float('inf'), local_start_sigma=float('inf'), w_global_pos=0.2,
                          w_global_node=0.2, w_local_pos=0.2, w_local_node=0.2, w_reg=1.0, **kwargs):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        atom_traj = []

        with torch.no_grad():
            skip = self.num_timesteps // n_steps
            seq = range(0, self.num_timesteps, skip)

            ## to test sampling with less intermediate diffusion steps
            # n_steps: the num of steps
            # seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            linker_mask = ~frag_mask
            frag_pos = ligand_pos_init[frag_mask, :]
            ligand_pos = ligand_pos_init
            linker_pos = ligand_pos_init[linker_mask, :]

            frag_batch = ligand_batch[frag_mask]
            
            protein_pos_ori = protein_pos
            protein_com = scatter_mean(protein_pos, protein_batch, dim=0)

            # subtract the center of mass of the pocket (COM)
            ligand_pos, protein_pos = center_pos_lp(ligand_pos, protein_pos, ligand_batch, protein_batch) #important
            frag_pos = ligand_pos[frag_mask, :]
            # Recover the linker postion
            ligand_pos[linker_mask] = linker_pos #important

            # VAE noise
            vae_noise = torch.zeros_like(ligand_atom_type).uniform_(-1, +1)
            # vae_noise = torch.randn_like(atom_type)
            # vae_noise = torch.clamp(torch.randn_like(atom_type), min=-3, max=3)
            # vae_noise = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device)

            '''
            Protein embedding
            '''
            protein_ctx = self.protein_encoder(
                node_attr=protein_atom_type,
                pos=protein_pos,
                batch=protein_batch,
            )
            # protein_ctx = scatter_mean(protein_ctx, protein_batch, dim=0)
            # protein_ctx = protein_ctx.index_select(0, ligand_batch)

            '''
            Atom numbers embedding
            '''
            num_node_ctx = None
            if 'atom_num_emb' not in self.__dict__.keys():
                self.atom_num_emb = False
            if self.atom_num_emb:
                nonlinearity = nn.ReLU()
                nemb = get_num_embedding(ligand_num_node, self.config.hidden_dim)
                nemb = self.nemb.dense[0](nemb)
                nemb = nonlinearity(nemb)
                nemb = self.nemb.dense[1](nemb)
                nemb = self.nemb_proj(nonlinearity(nemb))  # (G, dim)
                num_node_ctx = nemb.index_select(0, ligand_batch)

            # linker_pos = ligand_pos_init[linker_mask,:]
            # linker_atom = ligand_atom_type[linker_mask,:]
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)
                b = self.betas
                at = compute_alpha(b, t[0].long())

                pos_noise = torch.randn(size=ligand_pos[frag_mask, :].size(), device=ligand_pos.device)
                atom_noise = torch.randn(size=ligand_atom_type[frag_mask, :].size(), device=ligand_atom_type.device)
                mask = 1 - (t[0] == 0).float()

                frag_atom_type = ligand_atom_type[frag_mask,:]
                frag_pos = ligand_pos[frag_mask, :]
                net_out = self.net(
                    ligand_atom_type=ligand_atom_type,
                    ligand_pos=ligand_pos,
                    ligand_bond_index=ligand_bond_index,
                    ligand_bond_type=ligand_bond_type,
                    ligand_batch=ligand_batch,
                    protein_embeddings=protein_ctx,
                    time_step=t,
                    num_node_ctx=num_node_ctx,
                    protein_atom_feature=protein_atom_type[:, :10],
                    protein_pos=protein_pos,
                    protein_backbone_mask=protein_backbone_mask,
                    protein_batch=protein_batch,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain,
                    property_context=context,
                    vae_noise=None,
                    linker_mask=linker_mask
                )  # (E_global, 1), (E_local, 1)
                if self.vae_context:
                    pos_eq_global, pos_eq_local, node_score_global, node_score_local, edge_index, edge_type, edge_length, local_edge_mask = net_out[
                                                                                                                                            :-1]
                else:
                    pos_eq_global, pos_eq_local, node_score_global, node_score_local, edge_index, edge_type, edge_length, local_edge_mask = net_out
                # Local float('inf')

                # local_start_sigma = random.uniform(0.1,1)
                if sigmas[i] < local_start_sigma:
                    node_eq_local = pos_eq_local
                    if clip_local is not None:
                        node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                else:
                    node_eq_local = 0
                    node_score_local = 0

                # Global
                if sigmas[i] < global_start_sigma:
                    node_eq_global = pos_eq_global
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                    node_score_global = 0
                # Sum
                eps_pos = w_local_pos * node_eq_local + w_global_pos * node_eq_global  # + eps_pos_reg * w_reg
                eps_node = w_local_node * node_score_local + w_global_node * node_score_global
                # eps_pos = 3 * node_eq_local + 1 * node_eq_global  # + eps_pos_reg * w_reg
                # eps_node = 1 * node_score_local + 1 * node_score_global
                # Update

                sampling_type = kwargs.get("sampling_type", 'ddpm_noisy')  # types: generalized, ddpm_noisy, ld

                noise = torch.randn_like(ligand_pos)
                noise_node = torch.randn_like(ligand_atom_type)  # center_pos(torch.randn_like(pos), batch)

                t = t[0]
                next_t = (torch.ones(1) * j).to(ligand_pos.device)
                # at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                if sampling_type == 'generalized' or sampling_type == 'ddpm_noisy':
                    if sampling_type == 'generalized':
                        eta = kwargs.get("eta", 1.)
                        et = -eps_pos

                        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        c2 = ((1 - at_next) - c1 ** 2).sqrt()

                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 3 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
                        step_size_pos = step_size_pos_ld if step_size_pos_ld < step_size_pos_generalized else step_size_pos_generalized

                        step_size_noise_ld = torch.sqrt((step_lr * (sigmas[i] / 0.01) ** 2) * 2)
                        step_size_noise_generalized = 5 * (c1 / at_next.sqrt())
                        step_size_noise = step_size_noise_ld if step_size_noise_ld < step_size_noise_generalized else step_size_noise_generalized

                        # w = 1+2 * i/self.num_timesteps
                        w = 1

                        eps_node = eps_node / (1 - at).sqrt()
                        pos_next = ligand_pos - et * step_size_pos + w * noise * step_size_noise
                        atom_next = ligand_atom_type - eps_node * step_size_pos + w * noise_node * step_size_noise
                    elif sampling_type == 'ddpm_noisy':
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        # pos0_from_e = (1.0 / at).sqrt() * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # pos0_from_e = 1 * ligand_pos - (1.0 / at - 1).sqrt() * e
                        # mean_eps = (
                        #     (atm1.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * ligand_pos
                        # ) / (1.0 - at)
                        # mean = mean_eps
                        # mean = pos-beta_t/(1-at).sqrt()*e
                        mean = (ligand_pos - beta_t * e) / (1 - beta_t).sqrt()
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        pos_next = mean + mask * torch.exp(
                            0.5 * logvar) * noise  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ

                        e = eps_node
                        node0_from_e = (1.0 / at).sqrt() * ligand_atom_type - (1.0 / at - 1).sqrt() * e
                        mean_eps = (
                                           (atm1.sqrt() * beta_t) * node0_from_e + (
                                           (1 - beta_t).sqrt() * (1 - atm1)) * ligand_atom_type
                                   ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        atom_next = mean + mask * torch.exp(
                            0.5 * logvar) * noise_node  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ
                elif sampling_type == 'ld':
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = ligand_pos + step_size * eps_pos / sigmas[i] + noise * torch.sqrt(step_size * 2)
                    eps_node = eps_node / (1 - at).sqrt()
                    atom_next = ligand_atom_type - step_size * eps_node / sigmas[i] + noise_node * torch.sqrt(
                        step_size * 2)
                else:
                    raise ValueError('Unknown sampling type, it should be one of [generalized, ddpm_noisy, ld]')

                ligand_pos = pos_next #important
                ligand_atom_type = atom_next

                if torch.isnan(ligand_pos).any():
                    print('NaN detected. Please restart.')
                    print(node_eq_local)
                    print(node_eq_global)
                    raise FloatingPointError()
                # ligand_pos = center_pos(ligand_pos, ligand_batch) 
                # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])
                ligand_pos[frag_mask] = frag_pos #important
                # ligand_atom_type = torch.cat([linker_atom_type,ligand_atom_type[~linker_mask,:]])
                ligand_atom_type[frag_mask] = frag_atom_type

                ligand_pos, protein_pos = center_pos_pl(ligand_pos, protein_pos, ligand_batch, protein_batch) #important
                # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])
                # ligand_atom_type = torch.cat([linker_atom,ligand_atom_type[~linker_mask,:]])
                if clip_pos is not None:
                    ligand_pos = torch.clamp(ligand_pos, min=-clip_pos, max=clip_pos)

                protein_t = scatter_mean(protein_pos, protein_batch, dim=0)
                move_dist = protein_com - protein_t
                ligand_pos_fix = ligand_pos + move_dist[ligand_batch]
                pos_traj.append(ligand_pos_fix.clone().cpu())
                atom_traj.append(ligand_atom_type.clone().cpu())
        protein_final = scatter_mean(protein_pos, protein_batch, dim=0)
        # protein_final = protein_pos
        protein_pos = protein_pos + (protein_com - protein_final)[protein_batch]

        ligand_pos = ligand_pos + (protein_com - protein_final)[ligand_batch] #important
        ligand_pos = ligand_pos
        # ligand_pos = torch.cat([linker_pos,ligand_pos[~linker_mask,:]])

        # atom_type = torch.cat([atom_type[:,:-1]*4,atom_type[:,-1:]*10], dim=1)
        return ligand_pos, pos_traj, ligand_atom_type, atom_traj


def is_bond(edge_type):
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


# def is_radius_edge(edge_type):
#     return edge_type == 0

def is_local_edge(edge_type):
    return edge_type > 0
    # return edge_type == 0


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -torch.clamp(edge_length - rng, min=0.0, max=float('inf')) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def center_pos_pl(ligand_pos, pocket_pos, ligand_batch, pocket_batch):
    ligand_pos_center = ligand_pos - scatter_mean(ligand_pos, ligand_batch, dim=0)[ligand_batch]
    pocket_pos_center = pocket_pos - scatter_mean(ligand_pos, ligand_batch, dim=0)[pocket_batch]
    return ligand_pos_center, pocket_pos_center

def center_pos_lp(ligand_pos, pocket_pos, ligand_batch, pocket_batch):
    ligand_pos_center = ligand_pos - scatter_mean(pocket_pos, pocket_batch, dim=0)[ligand_batch]
    pocket_pos_center = pocket_pos - scatter_mean(pocket_pos, pocket_batch, dim=0)[pocket_batch]
    return ligand_pos_center, pocket_pos_center

def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom
