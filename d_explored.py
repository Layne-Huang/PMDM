# d_explored.py ---------------------------------------------------------------
import os, pickle, random, math, warnings
from pathlib import Path
from typing import List, Sequence, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_scatter import scatter_mean
from torch_geometric.data import Batch

from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP
from rdkit.Chem import qed

# ---------------------------------------------------------------------------
# 0)  tiny wrappers around your existing metric functions
# ---------------------------------------------------------------------------
from evaluation.sascorer import compute_sa_score          # SA
from evaluation.similarity import tanimoto_sim            # similarity
from evaluation.score_func import obey_lipinski           # Lipinski
from evaluation.docking import QVinaDockingTask           # docking

_METRIC_NAMES = ["sim", "SA", "QED", "logP", "Lipinski", "Vina"]

def _score_one_mol(
        mol: Chem.Mol,
        ref_mol: Chem.Mol,
        protein_filename: str,
        protein_root: str
) -> Sequence[float]:
    """Returns the 6-dim metric vector for a single ligand."""
    sim    = tanimoto_sim(ref_mol, mol)                          # ↑ better
    sa     = compute_sa_score(mol)                               # ↓ better
    qed_v  = qed(mol)                                            # ↑ better
    logp   = MolLogP(mol)                                        # depends
    lip_ok = obey_lipinski(mol)                                  # 0 / 1
    vina   = QVinaDockingTask.from_generated_data(
                protein_filename, mol, protein_root).run_sync()[0]["affinity"]

    return [sim, sa, qed_v, logp, lip_ok, vina]


# ---------------------------------------------------------------------------
# 1)  Re-use the feature-extractor / reward head you already wrote
# ---------------------------------------------------------------------------
from online_reward_model import (
    _PMDMFeatureExtractor, OnlineRewardPMDM, build_online_reward_net
)

# For building ProteinLigandData objects
from utils.protein_ligand import ProteinLigandData, PDBProtein, parse_sdf_file
from utils.transforms import (
    FeaturizeProteinAtom, FeaturizeLigandAtom,
    LigandCountNeighbors, CountNodesPerGraph, GetAdj, Compose
)
from utils.data import torchify_dict

# pointed to by your earlier scripts
FOLLOW_BATCH = ['ligand_atom_feature', 'protein_atom_feature_full']


# ---------------------------------------------------------------------------
# 2)  DExplored  (single-pocket version)
# ---------------------------------------------------------------------------
class DExplored(nn.Module):
    """
    Maintains the growing dataset {(x_i, y_i)} for ONE protein pocket
    and re-trains the Online-Reward head when asked.
    """
    def __init__(
        self,
        pdb_path:       str,
        ref_lig_path:   str,
        mdm_ckpt:       str,
        device:         torch.device = torch.device("cuda"),
        results_dir:    str = "./dexplored_cache",
    ):
        """
        pdb_path      : target pocket (same for all ligands)
        ref_lig_path  : reference ligand (for similarity metric)
        mdm_ckpt      : path to PMDM checkpoint (.pth / .pt)
        """
        super().__init__()
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        #   (1) frozen PMDM → backbone  ,   (2) fresh MLP head
        # ------------------------------------------------------------------
        self.reward_net = build_online_reward_net(
            mdm_ckpt, out_dim=len(_METRIC_NAMES), device=device
        )
        self.reward_net.mlp.apply(self._init_weights)  # fresh every run

        # frozen extractor for convenience
        self.extractor = self.reward_net.backbone

        # ------------------------------------------------------------------
        #   (3) pocket data / transforms prepared once
        # ------------------------------------------------------------------
        self.pdb_path   = Path(pdb_path)
        self.ref_mol    = Chem.SDMolSupplier(ref_lig_path)[0]

        self.protein_filename = self.pdb_path.name
        self.protein_root     = str(self.pdb_path.parent)

        # build Protein dict once  (ligand part is filled later)
        self._protein_data, self._transform = self._prepare_protein_part()

        # (4) experience buffers
        self.x = torch.empty(0, self.extractor.hidden_dim, device=device)
        self.y = torch.empty(0, len(_METRIC_NAMES),      device=device)

    # --------------------------------------------------------------
    #   utility: prepare static protein part & transforms
    # --------------------------------------------------------------
    def _prepare_protein_part(self):
        """Returns ProteinLigandData (*empty ligand*) and transform pipeline."""
        from easydict import EasyDict        # only needed here
        import numpy as np
        from Bio.PDB import PDBParser, Selection   # same as your script

        # --- parse PDB once ---------------------------------------
        ptable = Chem.GetPeriodicTable()
        parser = PDBParser(QUIET=True)
        model  = parser.get_structure(None, str(self.pdb_path))[0]

        p_dict = EasyDict({"element": [], "pos": [], "is_backbone": [], "atom_to_aa_type": []})
        for atom in Selection.unfold_entities(model, "A"):
            res = atom.get_parent()
            resname = res.get_resname()
            if resname == "MSE": resname = "MET"
            if resname not in PDBProtein.AA_NAME_NUMBER: continue
            if atom.element.upper() == "H":              continue
            p_dict["element"].append(ptable.GetAtomicNumber(atom.element.capitalize()))
            p_dict["pos"].append(torch.tensor(atom.get_coord(), dtype=torch.float32))
            p_dict["is_backbone"].append(atom.get_name() in ["N", "CA", "C", "O"])
            p_dict["atom_to_aa_type"].append(PDBProtein.AA_NAME_NUMBER[resname])

        p_dict["element"]       = torch.LongTensor(p_dict["element"])
        p_dict["pos"]           = torch.stack(p_dict["pos"])
        p_dict["is_backbone"]   = torch.BoolTensor(p_dict["is_backbone"])
        p_dict["atom_to_aa_type"] = torch.LongTensor(p_dict["atom_to_aa_type"])

        empty_lig = {
            "element": torch.empty(0, dtype=torch.long),
            "pos":     torch.empty(0, 3),
            "atom_feature": torch.empty(0, 8),
            "bond_index": torch.empty(2, 0, dtype=torch.long),
            "bond_type":  torch.empty(0, dtype=torch.long),
        }
        protein_data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=p_dict, ligand_dict=empty_lig)

        # ---------- build *exactly* the same transform pipeline ----
        protein_featurizer = FeaturizeProteinAtom("pdbbind", pocket=False)
        ligand_featurizer  = FeaturizeLigandAtom ("pdbbind", pocket=False)
        transform = Compose([
            LigandCountNeighbors(),
            protein_featurizer,
            ligand_featurizer,
            CountNodesPerGraph(),
            GetAdj(only_prot=True),
        ])
        protein_data = transform(protein_data)   # pre-compute prot features
        return protein_data, transform

    # --------------------------------------------------------------
    #   ingest freshly generated ligands (.sdf files in a folder)
    # --------------------------------------------------------------
    @torch.no_grad()
    def update_from_sdf_folder(self,
                               sdf_dir: str,
                               batch_size: int = 32) -> None:
        """
        * Reads every *.sdf* in `sdf_dir`
        * builds ligand+protein Batch
        * gets embedding **x**
        * computes GT metrics **y**
        * stores them in self.x / self.y  (+ .pkl snapshot)
        """
        sdf_paths = list(Path(sdf_dir).glob("*.sdf"))
        if len(sdf_paths) == 0:
            print(f"[DExplored] No SDF files found in {sdf_dir}")
            return

        new_x, new_y = [], []

        # we evaluate metrics molecule-by-molecule (docking is expensive)
        for sdf_path in sdf_paths:
            mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
            if mol is None: continue

            # ---------------- build PyG Data ----------------------
            lig_dict = torchify_dict(parse_sdf_file(str(sdf_path)))
            data = ProteinLigandData.from_protein_ligand_dicts(
                protein_dict=self._protein_data.to_dict()["protein"],
                ligand_dict=lig_dict
            )
            data = self._transform(data)
            batch = Batch.from_data_list([data], follow_batch=FOLLOW_BATCH).to(self.device)

            # embedding
            emb = self.extractor(batch).squeeze(0)     # [hidden_dim]
            new_x.append(emb)

            # ground-truth metrics
            y = torch.tensor(
                _score_one_mol(mol, self.ref_mol, self.protein_filename,
                               self.protein_root),
                dtype=torch.float32, device=self.device)
            new_y.append(y)

        if len(new_x) == 0: return
        new_x = torch.stack(new_x)
        new_y = torch.stack(new_y)

        # ------- append to buffers & persist ---------------------
        self.x = torch.cat([self.x, new_x], dim=0)
        self.y = torch.cat([self.y, new_y], dim=0)

        with open(self.results_dir / "dataset.pkl", "wb") as f:
            pickle.dump({"x": self.x.cpu(), "y": self.y.cpu()}, f)

        print(f"[DExplored] added {len(new_x)} samples → total {len(self.x)}")

    # --------------------------------------------------------------
    #   train reward net  (simple MSE, 5 % val split)
    # --------------------------------------------------------------
    def train_reward(self,
                     epochs: int = 300,
                     lr: float = 1e-3,
                     batch_size: int = 512,
                     val_split: float = 0.05):
        if len(self.x) < 10:
            print("[DExplored] not enough data to train")
            return

        # freeze backbone, train only mlp
        for p in self.extractor.parameters(): p.requires_grad_(False)
        for p in self.reward_net.mlp.parameters(): p.requires_grad_(True)
        self.reward_net.train()

        # split
        N = len(self.x)
        idx = list(range(N))
        random.shuffle(idx)
        split = int(N * (1 - val_split))
        train_idx, val_idx = idx[:split], idx[split:]

        train_ds = TensorDataset(self.x[train_idx], self.y[train_idx])
        val_ds   = TensorDataset(self.x[val_idx],   self.y[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        opt = torch.optim.Adam(self.reward_net.mlp.parameters(), lr=lr)
        best_val = math.inf
        best_state = None

        mse = nn.MSELoss()
        for ep in range(epochs):
            # --- train -----------------------------------------------------
            loss_list = []
            for xb, yb in train_loader:
                opt.zero_grad()
                pred = self.reward_net.mlp(xb)
                loss = mse(pred, yb)
                loss.backward()
                opt.step()
                loss_list.append(loss.item())
            train_loss = sum(loss_list) / len(loss_list)

            # --- val -------------------------------------------------------
            with torch.no_grad():
                vloss = []
                for xb, yb in val_loader:
                    vloss.append(mse(self.reward_net.mlp(xb), yb).item())
                val_loss = sum(vloss) / len(vloss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.reward_net.mlp.state_dict().items()}

            if (ep+1) % 20 == 0 or ep == 0:
                print(f"[DExplored] epoch {ep:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        # load best
        self.reward_net.mlp.load_state_dict(best_state)
        self.reward_net.eval()
        torch.save(self.reward_net.state_dict(), self.results_dir / "reward_head.pt")
        print(f"[DExplored] training done – best val {best_val:.4f}")

    # --------------------------------------------------------------
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
