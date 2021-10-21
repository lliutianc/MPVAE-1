#!/bin/bash

# FILENAME: run_fairreg.sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -A partner
#SBATCH -J run_fairreg


module load anaconda
conda activate fairmlc

module purge
module load cuda
module load cudnn
module load ml-toolkit-gpu/pytorch/1.7.1
module load learning/conda-2020.11-py38-gpu

module list

echo $CUDA_VISIBLE_DEVICES

python fairreg.py -dataset adult


#cd /home/liu3351/NC/

#python -X faulthandler mbert_fed_kmeans_all_layer.py