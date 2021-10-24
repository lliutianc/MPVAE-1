#!/bin/bash

#SBATCH --job-name=fairmlc-regularized
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END

#SBATCH --time=10:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM

module load Conda/3
conda activate fairmlc

python fairreg.py -dataset adult -latent_dim 8 -resume -labels_cluster_distance_threshold 0.01 -epoch 50 -labels_embed_method mpvae
python fairreg.py -dataset adult -latent_dim 8 -resume -labels_cluster_distance_threshold 0.01 -epoch 50 -labels_embed_method cbow

