dataset=$1
manipulation=$2
mem=$3
time=$4
partition=$5

max_depth_list=(1 2 3 4 5)

for max_depth in ${max_depth_list[@]}; do
    job_name=A_${dataset}_${manipulation}_${max_depth}

    sbatch --mem=${mem}G \
        --time=$time \
        --partition=$partition \
        --job-name=$job_name \
        --output=jobs/logs/attack/$job_name \
        --error=jobs/errors/attack/$job_name \
        jobs/attack/runner.sh $dataset $manipulation $max_depth
done
