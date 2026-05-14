# HPC Course Project

This repository contains the code for my HPC course project on **MPI/METIS distributed GNN inference for finite element graphs**.

The complete project code is available in this GitHub repository:

**Repository:** https://github.com/am6823/gnn-fea-surrogate.git

The part specifically related to the HPC project is located in the `hpc/` folder.

This folder contains the scripts used for METIS graph partitioning, owned/ghost node construction, MPI halo exchange, distributed GNN inference, comparison with serial full-graph inference, and runtime/scaling analysis.

More detailed explanations and reproducible commands are provided in `hpc/README.md`.

The main objective of the HPC implementation is to verify that the MPI reconstructed prediction exactly matches the serial full-graph GNN prediction.