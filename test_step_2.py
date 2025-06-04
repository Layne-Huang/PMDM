from dexplored import DExplored
device = torch.device("cuda:0")

dexp = DExplored(
    pdb_path      = "./example/4yhj.pdb",
    ref_lig_path  = "./example/4yhj_ref.sdf",
    mdm_ckpt      = "./checkpoints/pmdm.ckpt",
    device        = device,
    results_dir   = "./dexplored_4yhj"
)

# 1) After each sampling round (SDFs appear in ./generated_ligs)
dexp.update_from_sdf_folder("./generated_ligs", batch_size=16)

# 2) Once you have ≥ a few hundred points
dexp.train_reward(epochs=200, lr=1e-3)
