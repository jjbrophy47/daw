dataset_list=(adult bank_marketing census credit_card diabetes
              flight_delays gas_sensor higgs no_show olympics
              surgical synthetic twitter vaccine)
dataset_list=(adult bank_marketing census credit_card diabetes
              flight_delays gas_sensor no_show olympics
              surgical twitter vaccine)
dataset_list=(synthetic higgs)
max_depth_list=(1 2 3 4 5)
manipulation_list=('deletion' 'addition')

for dataset in ${dataset_list[@]}; do
    for manipulation in ${manipulation_list[@]}; do
        for max_depth in ${max_depth_list[@]}; do
            python3 scripts/experiments/robustness.py \
                --dataset=${dataset} \
                --max_depth=${max_depth} \
                --topd=0 \
                --random_state=1 \
                --k=10000 \
                --manipulation=${manipulation}
        done
    done
done
