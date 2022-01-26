#!/bin/bash

#SBATCH --job-name=run_fairsoft_trial
#SBATCH --gres=gpu:k80:1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END

#SBATCH --time=20:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM
#SBATCH --array=1-10

module load Conda/3
conda activate fairmlc


# python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID
# python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 1 -seed $SLURM_ARRAY_TASK_ID

python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 200 -bs 32 -fair_coeff 0.1
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 200 -bs 32 -fair_coeff 1
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 200 -bs 32 -fair_coeff 10
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 200 -bs 32 -fair_coeff 100
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 200 -bs 32 -fair_coeff 500
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 200 -bs 32 -fair_coeff 1000
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 200 -bs 32 -fair_coeff 5000

python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 20 -bs 128 -fair_coeff 0.1
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 20 -bs 128 -fair_coeff 1
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 20 -bs 128 -fair_coeff 10
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 20 -bs 128 -fair_coeff 100
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 20 -bs 128 -fair_coeff 500
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 20 -bs 128 -fair_coeff 1000
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 20 -bs 128 -fair_coeff 5000



# python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed 1 -epoch 200 -bs 32 -fair_coeff 0.1

# 


