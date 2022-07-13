dataset=$1
model=$2
method=$3
k=$4
seed=$5
ncpu=$6
time=$7
partition=$8

job_name=I_${dataset}_${model}_${method}_${k}

sbatch --cpus-per-task=$ncpu \
    --time=$time \
    --partition=$partition \
    --job-name=$job_name \
    --output=jobs/logs/influence/$job_name \
    --error=jobs/errors/influence/$job_name \
    jobs/influence/runner.sh $dataset $model $method $k $seed
