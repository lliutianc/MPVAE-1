#!/bin/bash

#SBATCH --job-name=fairmlc-regularized-kmeans
#SBATCH --gres=gpu:k80:1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END

#SBATCH --time=10:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM

module load Conda/3
conda activate fairmlc

# tune cluster numbers
python fairkmeans.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 8
python fairkmeans.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 16
python fairkmeans.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 32
python fairkmeans.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 64
