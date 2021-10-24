#!/bin/bash

#SBATCH --job-name=fairmlc-regularized
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END

#SBATCH --time=5:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM

module load Conda/3
conda activate fairmlc


python baseline.py -dataset adult -latent_dim 8 -resume -epoch 50

