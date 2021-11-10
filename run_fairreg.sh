#!/bin/bash

#SBATCH --job-name=fairmlc-regularized
#SBATCH --gres=gpu:k80:1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END

#SBATCH --time=10:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM

module load Conda/3
conda activate fairmlc

python fairreg.py -dataset adult -latent_dim 8 -labels_cluster_distance_threshold 0.01 -epoch 20 -labels_embed_method none -labels_cluster_method kmeans
python fairreg.py -dataset adult -latent_dim 8 -labels_cluster_distance_threshold 0.01 -epoch 20 -labels_embed_method cbow -labels_cluster_method kmeans
python fairreg.py -dataset adult -latent_dim 8 -labels_cluster_distance_threshold 0.01 -epoch 20 -labels_embed_method mpvae -labels_cluster_method kmeans


# python fairreg.py -dataset adult -latent_dim 8 -labels_cluster_distance_threshold 0.01 -epoch 20 -labels_embed_method none -labels_cluster_method kmodes
