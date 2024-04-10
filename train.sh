#!/bin/bash

#SBATCH --output="train.out"
#SBATCH --error="train.out"
#SBATCH --ntasks=1
#SBATCj --gpus-per-task=1
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodelist=c[12]

module purge
module load anaconda3
source activate image_generation

export CUDA_HOME=$CONDA_PREFIX/lib:$CUDA_HOME
export PATH=$CUDA_HOME:$CONDA_PREFIX/lib:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python3 code/train.py


