#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
manipulation=$2
max_depth=$3

python3 scripts/experiments/attack.py \
  --dataset $dataset \
  --manipulation $manipulation \
  --max_depth $max_depth
