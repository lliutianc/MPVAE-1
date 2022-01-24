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


python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -label_z_fair_coeff 500.
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label 1 -seed $SLURM_ARRAY_TASK_ID -label_z_fair_coeff 500.

python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -epoch 200 -bs 32 -label_z_fair_coeff 500.
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 1 -seed $SLURM_ARRAY_TASK_ID -epoch 200 -bs 32 -label_z_fair_coeff 500.


# python fairsoft_trial.py -dataset donor -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed $SLURM_ARRAY_TASK_ID -label_z_fair_coeff 5. -feat_z_fair_coeff 5.
# python fairsoft_trial.py -dataset donor -latent_dim 8 -target_label_idx 0 -mask_target_label 1 -seed $SLURM_ARRAY_TASK_ID -label_z_fair_coeff 5. -feat_z_fair_coeff 5.





# python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label 0 -seed 1 -epoch 50 -bs 16