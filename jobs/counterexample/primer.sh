num_attributes=$1
num_copies=$2
mem=$3
time=$4
partition=$5

max_depth_list=(1 2 3 4 5)

for max_depth in ${max_depth_list[@]}; do
    job_name=CE_${num_attributes}_${num_copies}_${max_depth}

    sbatch --mem=${mem}G \
        --time=$time \
        --partition=$partition \
        --job-name=$job_name \
        --output=jobs/logs/counterexample/$job_name \
        --error=jobs/errors/counterexample/$job_name \
        jobs/counterexample/runner.sh $num_attributes $num_copies $max_depth
done
