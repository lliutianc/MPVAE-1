#!/bin/bash

#SBATCH --job-name=run_fairsoft_trial
#SBATCH --gres=gpu:k80:1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END

#SBATCH --time=10:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM
#SBATCH --array=0-4

module load Conda/3
conda activate fairmlc

python fairsoft_trial.py -dataset donor -latent_dim 8 -target_label_idx 0 -mask_target_label 0