#!/bin/bash

# FILENAME: run_fairreg.sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -A partner
#SBATCH -J run_fairreg

module load cuda
module load cudnn
module load ml-toolkit-gpu/pytorch/1.7.1
module load learning/conda-2020.11-py38-gpu

module list

echo $CUDA_VISIBLE_DEVICES

python -X faulthandler fairreg.py -dataset adult
