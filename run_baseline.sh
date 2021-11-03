#!/bin/bash

#SBATCH --job-name=fairmlc-baseline
#SBATCH --gres=gpu:k80:1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END

#SBATCH --time=8:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM

module load Conda/3
conda activate fairmlc


python baseline.py -dataset adult -latent_dim 8 -epoch 20 -bs 64

