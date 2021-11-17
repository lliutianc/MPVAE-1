#!/bin/bash

#SBATCH --job-name=fairmlc-regularized
#SBATCH --gres=gpu:k80:1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END

#SBATCH --time=10:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM

module load Conda/3
conda activate fairmlc

# tune min_support
python fairarule.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 8 -min_support 0.001
python fairarule.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 8 -min_support 0.005
python fairarule.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 8 -min_support 0.01
python fairarule.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 8 -min_support 0.05


# tune min_confidence
python fairarule.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 8 -min_confidence 0.01
python fairarule.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 8 -min_confidence 0.05
python fairarule.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 8 -min_confidence 0.1
python fairarule.py -dataset adult -latent_dim 8 -epoch 20 -labels_embed_method none -labels_cluster_num 8 -min_confidence 0.25