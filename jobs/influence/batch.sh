
dataset_list=('adult' 'bank_marketing' 'surgical' 'vaccine' 'no_show')
model_list=('lr' 'dt' 'lgb' 'rf')
method_list=('random' 'loo' 'aki' 'subsample')
seed_list=(1 2 3 4 5)

ncpu=5  # number of CPUs to use
time=1440  # minutes
partition='short'

for dataset in ${dataset_list[@]}; do
    for model in ${model_list[@]}; do
        for method in ${method_list[@]}; do
            if [ $method != 'aki' ]; then
                k_list=(1)
            else
                k_list=(1 10 100)
            fi
            for k in ${k_list[@]}; do
                for seed in ${seed_list[@]}; do
                    ./jobs/influence/primer.sh $dataset $model $method $k $ncpu $time $partition
                done
            done
        done
    done
done
