#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

num_attributes=$1
num_copies=$2
max_depth=$3

python3 scripts/experiments/counterexample.py \
  --p $num_attributes \
  --num_copies $num_copies \
  --max_depth $max_depth
