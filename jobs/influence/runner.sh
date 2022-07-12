#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
model=$2
method=$3
k=$4

python3 scripts/experiments/influence.py \
  --dataset $dataset \
  --model $model \
  --method $method \
  --k $k
