#!/bin/bash

# FILENAME: run_fairreg.sh
#SBATCH -A partner

#SBATCH --nodes=1 --time=01:00:00

module load anaconda
conda activate fairmlc

module load cuda
module load cudnn
module load ml-toolkit-gpu/pytorch/1.7.1

echo $CUDA_VISIBLE_DEVICES


python fairreg.py -dataset adult


#cd /home/liu3351/NC/

#module load learning/conda-2020.11-py38-gpu
#python -X faulthandler mbert_fed_kmeans_all_layer.py