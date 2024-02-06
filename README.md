# PMDM: A dual diffusion model enables 3D binding bioactive molecule generation and lead optimization given target pockets

Official implementation of **PMDM**, an equivariant model for A dual diffusion model enables 3D binding bioactive molecule generation and lead optimization given target pockets, by Lei Huang.

## 📢 News

- Our paper is accepted by **Nature Communications** !!

[![biorxiv](https://img.shields.io/badge/biorxiv-526011-AE353A.svg)](https://www.biorxiv.org/content/10.1101/2023.01.28.526011v1.abstract)

<div align="center">  
<img src="img/model.png" width="600">
</div>
<div align="center"> 
<img src="img/traj.gif" alt="GIF" width="200">
</div>
1. [Dependencies](#dependencies)
   1. [Conda environment](#conda-environment)
   2. [QuickVina 2](#quickvina-2)
   3. [Pre-trained models](#pre-trained-models)
2. [Benchmarks](#benchmarks)
   1. [CrossDocked Benchmark](#crossdocked)
   2. [Binding MOAD](#binding-moad)
4. [Training](#training)
5. [Inference](#inference)
   1. [Sample molecules for a given pocket](#sample-molecules-for-a-given-pocket)
   2. [Test set sampling](#sample-molecules-for-all-pockets-in-the-test-set)
   3. [Fix substructures](#fix-substructures)
   4. [Metrics](#metrics)
   5. [QuickVina2](#quickvina2)
6. [Citation](#citation)

## Dependencies

### Conda environment
Please use our environment file to install the environment.
```bash
# Clone the environment
conda env create -f mol.yml
# Activate the environment
conda activate mol
```
### QuickVina 2
For docking, install QuickVina 2:

```bash
wget https://github.com/QVina/qvina/raw/master/bin/qvina2.1
chmod +x qvina2.1
```

Preparing the receptor for docking (pdb -> pdbqt) requires a new environment which is based on python 2x, so we should create a new environment:
```bash
# Clone the environment
conda env create -f evaluation/env_adt.yml
# Activate the environment
conda activate adt
```
### Pre-trained models
The pre-trained models could be downloaded from [Zenodo](https://zenodo.org/record/8183747).

## Benchmarks
### CrossDocked

#### Data preparation
Download and extract the dataset as described by the authors of Pocket2Mol: https://github.com/pengxingang/Pocket2Mol/tree/main/data

### Binding MOAD
#### Data preparation
Download the dataset
```bash
wget http://www.bindingmoad.org/files/biou/every_part_a.zip
wget http://www.bindingmoad.org/files/biou/every_part_b.zip
wget http://www.bindingmoad.org/files/csv/every.csv

unzip every_part_a.zip
unzip every_part_b.zip
```

## Training
We provide two training scripts **train.py** and **train_ddp_op.py** for single-GPU training and multi-GPU training.

Starting a new training run:
```bash
python -u train.py --config <config>.yml
```
The example configure file is in `configs/crossdock_epoch.yml`

Resuming a previous run:
```bash
python -u train.py --config <configure file path>
```
The config argument should be the upper path of the configure file.

## Inference
### Sample molecules for all pockets in the test set
```bash
python -u sample_batch.py --ckpt <checkpoint> --num_samples <number of samples> --sampling_type generalized
```

### Sample molecules for given customized pockets
```bash
python -u sample_for_pdb.py --ckpt <checkpoint> --pdb_path <pdb path> --num_atom <num atom> --num_samples <number of samples> --sampling_type generalized
```
`num_atom` is the number of atoms of generated molecules.

### Sample novel molecules given seed fragments
```bash
python -u sample_frag.py --ckpt <checkpoint> --pdb_path <pdb path> --mol_file <mole file> --keep_index <seed fragments index> --num_atom <num atom> --num_samples <number of samples> --sampling_type generalized
```
`num_atom` is the number of atoms of generated fragments. `keep_index` is the index of the atoms of the seed fragments.

### Sample novel molecules for linker 
```bash
python -u sample_linker.py --ckpt <checkpoint> --pdb_path <pdb path> --mol_file <mole file> --keep_index <seed fragments index> --num_atom <num atom> --num_samples <number of samples> --sampling_type generalized
```
`num_atom` is the number of atoms of generated fragments. `mask` is the index of the linker that you would like to replace in the original molecule.

### Metrics
Evaluate the batch of generated molecules (You need to turn on the `save_results` arguments in sample* scripts)
```bash
python -u evaluate --path <molecule_path>
```

If you want to evaluate a single molecule, use `evaluate_single.py`.

### QuickVina2
First, convert all protein PDB files to PDBQT files using adt envrionment
```bash
conda activate adt
prepare_receptor4.py -r {} -o {}
cd evaluation
```
Then, compute QuickVina scores:
```bash
conda deactivate
conda activate mol
python docking_2_single.py --receptor_file <prepapre_receptor4_outdir> --sdf_file <sdf file> --out_dir <qvina_outdir>
```

### Citation
```
@article {Huang2023.01.28.526011,
	author = {Lei Huang and Tingyang Xu and Yang Yu and Peilin Zhao and Ka-Chun Wong and Hengtong Zhang},
	title = {A dual diffusion model enables 3D binding bioactive molecule generation and lead optimization given target pockets},
	elocation-id = {2023.01.28.526011},
	year = {2023},
	doi = {10.1101/2023.01.28.526011},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/01/30/2023.01.28.526011},
	eprint = {https://www.biorxiv.org/content/early/2023/01/30/2023.01.28.526011.full.pdf},
	journal = {bioRxiv}
}
```




