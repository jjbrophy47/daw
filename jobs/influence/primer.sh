dataset=$1
model=$2
method=$3
k=$4
ncpu=$5
time=$6
partition=$7

job_name=A_${dataset}_${model}_${method}_${k}

sbatch --cpus-per-task=$ncpu \
    --time=$time \
    --partition=$partition \
    --job-name=$job_name \
    --output=jobs/logs/influence/$job_name \
    --error=jobs/errors/influence/$job_name \
    jobs/influence/runner.sh $dataset $model $method $k
