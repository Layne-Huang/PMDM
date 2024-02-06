# PMDM: A dual diffusion model enables 3D binding bioactive molecule generation and lead optimization given target pockets

Official implementation of **PMDM**, an equivariant model for A dual diffusion model enables 3D binding bioactive molecule generation and lead optimization given target pockets, by Lei Huang

[![biorxiv](https://img.shields.io/badge/arXiv-2210.13695-B31B1B.svg)](https://www.biorxiv.org/content/10.1101/2023.01.28.526011v1.abstract)

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
