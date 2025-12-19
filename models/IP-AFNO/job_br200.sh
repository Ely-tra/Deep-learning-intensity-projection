#!/bin/bash -l

#SBATCH -N 1
#SBATCH -t 7:59:00
#SBATCH -J TCNN-ctl
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=128G

module load python/3.10.10
set -x

python IP-build_model.py