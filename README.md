# PMDM: A dual diffusion model enables 3D binding bioactive molecule generation and lead optimization given target pockets

Official implementation of **DiffSBDD**, an equivariant model for structure-based drug design, by Arne Schneuing*, Yuanqi Du*, Charles Harris, Arian Jamasb, Ilia Igashov, Weitao Du, Tom Blundell, Pietro Lió, Carla Gomes, Max Welling, Michael Bronstein & Bruno Correia.

[![arXiv](https://img.shields.io/badge/arXiv-2210.13695-B31B1B.svg)](http://arxiv.org/abs/2210.13695)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arneschneuing/DiffSBDD/blob/main/colab/DiffSBDD.ipynb)

![](img/overview.png)

1. [Dependencies](#dependencies)
   1. [Conda environment](#conda-environment)
   2. [QuickVina 2](#quickvina-2)
   3. [Pre-trained models](#pre-trained-models)
2. [Benchmarks](#benchmarks)
   1. [CrossDocked Benchmark](#crossdocked)
   2. [Binding MOAD](#binding-moad)
   3. [Sampled molecules](#sampled-molecules)
4. [Training](#training)
5. [Inference](#inference)
   1. [Sample molecules for a given pocket](#sample-molecules-for-a-given-pocket)
   2. [Test set sampling](#sample-molecules-for-all-pockets-in-the-test-set)
   3. [Fix substructures](#fix-substructures)
   4. [Metrics](#metrics)
   5. [QuickVina2](#quickvina2)
6. [Citation](#citation)
