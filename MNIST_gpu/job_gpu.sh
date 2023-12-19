#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

module load scipy-stack/2022a symengine/0.9.0 cuda/11.4
source ../env/bin/activate
python MNIST_gpu.py
