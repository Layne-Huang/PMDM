import torch.nn as nn
from torch_scatter import scatter_mean
import torch

# ------------------------------------------------------------------
# 1.  A thin wrapper that re-uses the *frozen* PMDM encoders
# ------------------------------------------------------------------
class _PMDMFeatureExtractor(nn.Module):
    """
    Takes the original (pre-trained) MDM_full_pocket_coor_shared model
    and returns a graph-level embedding for each ligand-protein pair.

    Everything here is frozen ⇒ NO gradients flow into PMDM.
    """
    def __init__(self, mdm):
        super().__init__()

        # Copy references to the already-loaded encoders
        self.protein_encoder  = mdm.protein_encoder
        self.ligand_encoder   = mdm.ligand_encoder
        self.atten_layer      = mdm.atten_layer          # cross-attention
        self.hidden_dim       = mdm.hidden_dim

        # Make sure they stay frozen
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, batch):
        """
        Args
        ----
        batch : torch_geometric.data.Batch constructed exactly the
                same way as for PMDM training / sampling
        Returns
        -------
        graph_emb : [G, hidden_dim] pooled representation (one row per graph)
        """

        # --- ligand + protein atom encodings --------------------------------
        lig_atom_feat = batch.ligand_atom_feature.float()
        lig_pos       = batch.ligand_pos
        lig_batch     = batch.ligand_atom_feature_batch     # [N_ligand]

        prot_atom_feat = batch.protein_atom_feature_full.float()
        prot_pos       = batch.protein_pos
        prot_batch     = batch.protein_atom_feature_full_batch  # [N_prot]

        # 1) Encode each side separately (SchNet encoders are already inside PMDM)
        prot_emb = self.protein_encoder(node_attr=prot_atom_feat,
                                        pos=prot_pos,
                                        batch=prot_batch)

        lig_emb  = self.ligand_encoder(node_attr=lig_atom_feat,
                                       pos=lig_pos,
                                       batch=lig_batch)

        # 2) One round of cross-attention (same layer PMDM uses)
        lig_emb, _ = self.atten_layer(lig_emb, prot_emb)   # cross-attend protein → ligand

        # 3) Pool ONLY ligand nodes to a graph-level vector
        graph_emb = scatter_mean(lig_emb, lig_batch, dim=0)    # shape [G, hidden_dim]
        return graph_emb


# ------------------------------------------------------------------
# 2.  Online reward head  (trainable from scratch every outer loop)
# ------------------------------------------------------------------
class OnlineRewardPMDM(nn.Module):
    """
    A small MLP on top of the frozen PMDM encoder.

    By default we predict the 6 scalar components used in your GT
    reward script:  [Tanimoto, SA, QED, logP, Lipinski, vina_score]
    but `out_dim` can be adjusted easily.
    """
    def __init__(self, mdm_pretrained, out_dim: int = 6):
        super().__init__()

        # Frozen backbone
        self.backbone = _PMDMFeatureExtractor(mdm_pretrained)

        # Train-from-scratch MLP head (≈ 50 k params for hidden_dim=256)
        h = self.backbone.hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, h // 2),
            nn.ReLU(inplace=True),
            nn.Linear(h // 2, out_dim)
        )

    # ------------------------------------------------------------------
    def forward(self, batch):
        g = self.backbone(batch)          # [G, hidden_dim]  (no grad)
        return self.mlp(g)                # [G, out_dim]     (trainable)


# ------------------------------------------------------------------
# 3.  Convenience helper – factory function
# ------------------------------------------------------------------
def build_online_reward_net(ckpt_path: str,
                            out_dim: int = 6,
                            device: torch.device = "cuda"):
    """
    ckpt_path : path to the *diffusion* model checkpoint (.pt or .pth)
    Returns   : OnlineRewardPMDM model with frozen encoders loaded
    """
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    # Re-instantiate the full PMDM model and load weights
    from models.epsnet import get_model      # same helper your sampling script uses
    mdm = get_model(config.model).to(device)
    mdm.load_state_dict(ckpt["model"], strict=True)
    mdm.eval()                               # we do *not* train it

    # Build reward network
    reward_net = OnlineRewardPMDM(mdm, out_dim=out_dim).to(device)
    return reward_net
