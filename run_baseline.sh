#!/bin/bash

#SBATCH --job-name=fairmlc-baseline
#SBATCH --gres=gpu:k80:1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END

#SBATCH --time=8:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM

module load Conda/3
conda activate fairmlc


python baseline.py -dataset adult -latent_dim 8 -epoch 20 -cuda 6


python fairreg.py -dataset adult -latent_dim 8 -labels_cluster_distance_threshold 0.01 -epoch 20 -labels_embed_method mpvae -labels_cluster_method kmeans
