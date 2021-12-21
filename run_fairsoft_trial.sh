#!/bin/bash

#SBATCH --job-name=fairmlc-soft-trial
#SBATCH --gres=gpu:k80:1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END

#SBATCH --time=8:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM

module load Conda/3
conda activate fairmlc

python fairsoft_trial.py -dataset adult -latent_dim 8 