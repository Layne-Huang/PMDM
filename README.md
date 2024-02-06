# PMDM: A dual diffusion model enables 3D binding bioactive molecule generation and lead optimization given target pockets

Official implementation of **PMDM**, an equivariant model for A dual diffusion model enables 3D binding bioactive molecule generation and lead optimization given target pockets, by Lei Huang.

## 📢 News

- Our paper is accepted by **Nature Communications** !!

[![biorxiv](https://img.shields.io/badge/biorxiv-526011-AE353A.svg)](https://www.biorxiv.org/content/10.1101/2023.01.28.526011v1.abstract)

<div  align="center">  
<img src="img/model.png" width="600">
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
