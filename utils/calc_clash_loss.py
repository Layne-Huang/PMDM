"""Calculate the steric clash between protein pocket & small molecule."""
import torch

from .misc import unbatch

# Van der Waals radius (reference: https://en.wikipedia.org/wiki/Van_der_Waals_radius)
VDW_RADIUS_DICT = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Se': 1.90}
VDW_RADIUS_LIST = [1.20, 1.70, 1.55, 1.52, 1.47, 1.80, 1.80, 1.75, 1.90]


def cal_clash_loss(coor_prot, coor_mole, atom_type_prot, atom_type_mole, protein_batch, ligand_batch):
    coor_prot_list = unbatch(coor_prot, protein_batch)
    coor_mole_list = unbatch(coor_mole, ligand_batch)
    atom_prot_list = unbatch(atom_type_prot, protein_batch)
    atom_mole_list = unbatch(atom_type_mole, ligand_batch)
    loss_list = []
    for i in range(len(coor_prot_list)):
        atom_prot = torch.argmax(atom_prot_list[i], dim=1)
        atom_mole = torch.argmax(atom_mole_list[i], dim=1)
        coor_mole = coor_mole_list[i] + torch.mean(coor_prot_list[i])
        loss = clash_loss(coor_prot_list[i], coor_mole, atom_prot, atom_mole)
        loss_list.append(loss)
    loss = torch.mean(torch.tensor(loss_list))
    return loss


def clash_loss(cord_mat_prot, cord_mat_mole, atom_types_prot, atom_types_mole, debug=False):
    """Calculate the steric clash between protein pocket & small molecule.

    Args:
    * cord_mat_prot: per-atom 3D coordinates in the protein pocket of size N x 3 (torch.Tensor)
    * cord_mat_mole: per-atom 3D coordinates in the small molecule of size M x 3 (torch.Tensor)
    * atom_types_prot: list of atom types in the protein pocket of length N (list)
    * atom_types_mole: list of atom types in the small molecule of length M (list)
    * debug: (optional) whether debug-only information should be displayed

    Returns:
    * loss: steric clash loss (torch.Tensor / scalar)

    Notes:
    * Valid atom types include: 'C', 'N', 'O', 'F', 'S', and 'Cl'? (to be discussed)
    * If the distance between two atoms (one from the protein pocket and the other from the small
        molecule) is lower than the sum of their Van der Waals radius minus the tolerance (1.5A),
        then a steric clash loss will be calculated.
    * Tow normalization schemes are available:
      - normalize by the number of atom pairs
      - normalize by the number of atom pairs violating the distance lower bound (default)
    """

    # configurations
    eps = 1e-6
    dist_tol = 1.5

    # initialization
    dtype = cord_mat_prot.dtype
    device = cord_mat_prot.device

    # check for unexpected atom types
    # atom_types_vald = set(VDW_RADIUS_DICT.keys())
    # atom_types_uexp = set(atom_types_prot) - atom_types_vald
    atom_types_vald = len(VDW_RADIUS_LIST)
    atom_types_uexp = max(atom_types_prot) - atom_types_vald
    if atom_types_uexp > 0:
        raise ValueError(f'unexpected atom types in the protein pocket: {atom_types_uexp}')
    atom_types_uexp = max(atom_types_mole) - atom_types_vald
    if atom_types_uexp > 0:
        raise ValueError(f'unexpected atom types in the small molecule: {atom_types_uexp}')

    # build the Van der Waals radius vector for protein pocket and small molecule
    vdw_vec_prot = torch.tensor(
        [VDW_RADIUS_LIST[x] for x in atom_types_prot], dtype=dtype, device=device)
    vdw_vec_mole = torch.tensor(
        [VDW_RADIUS_LIST[x] for x in atom_types_mole], dtype=dtype, device=device)

    # calculate the minimal allowed distance between atoms in the protein pocket and small molecule
    dist_mat_lbnd = vdw_vec_prot.view(-1, 1) + vdw_vec_mole.view(1, -1) - dist_tol

    # calculate the actual distance between atoms in the protein pocket and small molecule
    dist_mat = torch.cdist(
        cord_mat_prot, cord_mat_mole, compute_mode='donot_use_mm_for_euclid_dist')

    # calculate the steric clash loss
    mask_mat = torch.lt(dist_mat, dist_mat_lbnd)
    derr_mat = torch.clamp(dist_mat_lbnd - dist_mat, min=0.0)
    # loss = torch.mean(derr_mat)  # normalization scheme #1
    loss = torch.sum(derr_mat) / (torch.sum(mask_mat) + eps)  # normalization scheme #2

    # [debug-only] display all the atom pairs violating the distance lower bound
    if debug:
        idxs_nnz = torch.nonzero(mask_mat)
        print(f'# of atoms pairs violating the distance lower bound: {idxs_nnz.shape[0]}')
        for ir in range(idxs_nnz.shape[0]):
            idx_atom_prot, idx_atom_mole = idxs_nnz[ir].tolist()
            dist_val = dist_mat[idx_atom_prot][idx_atom_mole].item()
            dist_val_lbnd = dist_mat_lbnd[idx_atom_prot][idx_atom_mole].item()
            print(f'({idx_atom_prot}, {idx_atom_mole}) {dist_val:.4f} / {dist_val_lbnd:.4f}')

    return loss
