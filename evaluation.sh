#!/bin/bash
#SBATCH --partition=TrixieMain
# Request GPUs for the job. In this case 4 GPUs
#SBATCH --gres=gpu:1
# Print out the hostname that the jobs is running on
hostname
# Run nvidia-smi to ensure that the job sees the GPUs
/usr/bin/nvidia-smi

module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu
# Activate the conda pytorch environment created in step 1
source activate prots
# Load the miniconda module on the compute node
# Activate the conda pytorch environment created in step 1
# Launch our test pytorch python file
python evaluation.py