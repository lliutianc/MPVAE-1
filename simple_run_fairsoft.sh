#!/bin/bash

conda activate fairmlc

seed=$1
mask_target_label=$2

python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -epoch 500 -bs 32 -fair_coeff 0.1
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 500 -bs 32 -fair_coeff 1
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 500 -bs 32 -fair_coeff 10
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 500 -bs 32 -fair_coeff 100
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 500 -bs 32 -fair_coeff 500
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 500 -bs 32 -fair_coeff 1000
python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 500 -bs 32 -fair_coeff 5000

python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 20 -bs 128 -fair_coeff 0.1
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 20 -bs 128 -fair_coeff 1
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 20 -bs 128 -fair_coeff 10
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 20 -bs 128 -fair_coeff 100
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 20 -bs 128 -fair_coeff 500
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 20 -bs 128 -fair_coeff 1000
python fairsoft_trial.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed $seed -epoch 20 -bs 128 -fair_coeff 5000



# python fairsoft_trial.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label 0 -seed 1 -epoch 500 -bs 32 -fair_coeff 0.1

# 



