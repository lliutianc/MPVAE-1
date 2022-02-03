#!/bin/bash

seed=$1
mask_target_label=$2
cuda=$3

python fairsoft_representation_extract.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed  -bs 32 -fair_coeff 0.1 -cuda $cuda
python fairsoft_representation_extract.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed  -bs 32 -fair_coeff 1 -cuda $cuda
python fairsoft_representation_extract.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed  -bs 32 -fair_coeff 10 -cuda $cuda
python fairsoft_representation_extract.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed  -bs 32 -fair_coeff 100 -cuda $cuda
python fairsoft_representation_extract.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed  -bs 32 -fair_coeff 500 -cuda $cuda
python fairsoft_representation_extract.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed  -bs 32 -fair_coeff 1000 -cuda $cuda
python fairsoft_representation_extract.py -dataset credit -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed  -bs 32 -fair_coeff 5000 -cuda $cuda

python fairsoft_representation_extract.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -bs 128 -fair_coeff 0.1 -cuda $cuda
python fairsoft_representation_extract.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -bs 128 -fair_coeff 1 -cuda $cuda
python fairsoft_representation_extract.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -bs 128 -fair_coeff 10 -cuda $cuda
python fairsoft_representation_extract.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -bs 128 -fair_coeff 100 -cuda $cuda
python fairsoft_representation_extract.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -bs 128 -fair_coeff 500 -cuda $cuda
python fairsoft_representation_extract.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -bs 128 -fair_coeff 1000 -cuda $cuda
python fairsoft_representation_extract.py -dataset adult -latent_dim 8 -target_label_idx 0 -mask_target_label $mask_target_label -seed $seed -bs 128 -fair_coeff 5000 -cuda $cuda