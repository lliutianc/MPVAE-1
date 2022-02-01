#!/bin/bash

seed=$1
mask_target_label=$2
cuda=$3

python fairsoft_trial_postprocess.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -epoch 500 -bs 32 -learn_logit 0 -lr 0.0001 -fair_coeff 0.01  -cuda $cuda
python fairsoft_trial_postprocess.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -epoch 500 -bs 32 -learn_logit 0 -lr 0.0001 -fair_coeff 0.1 -cuda $cuda
python fairsoft_trial_postprocess.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -epoch 500 -bs 32 -learn_logit 0 -lr 0.0001 -fair_coeff 1 -cuda $cuda
python fairsoft_trial_postprocess.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -epoch 500 -bs 32 -learn_logit 0 -lr 0.0001 -fair_coeff 2 -cuda $cuda
python fairsoft_trial_postprocess.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -epoch 500 -bs 32 -learn_logit 0 -lr 0.0001 -fair_coeff 5 -cuda $cuda

python fairsoft_trial_postprocess.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -epoch 20 -bs 128 -learn_logit 0 -lr 0.0001 -fair_coeff 0.01 -cuda $cuda
python fairsoft_trial_postprocess.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -epoch 20 -bs 128 -learn_logit 0 -lr 0.0001 -fair_coeff 0.1 -cuda $cuda
python fairsoft_trial_postprocess.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -epoch 20 -bs 128 -learn_logit 0 -lr 0.0001 -fair_coeff 1 -cuda $cuda
python fairsoft_trial_postprocess.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -epoch 20 -bs 128 -learn_logit 0 -lr 0.0001 -fair_coeff 2 -cuda $cuda
python fairsoft_trial_postprocess.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -epoch 20 -bs 128 -learn_logit 0 -lr 0.0001 -fair_coeff 5 -cuda $cuda

# ./simple_run_fairsoft.sh 1 1 0