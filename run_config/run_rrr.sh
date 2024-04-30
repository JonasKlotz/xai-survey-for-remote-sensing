#!/bin/bash

# Activate the environment
source /home/jonas/miniconda3/bin/activate /home/jonas/micromamba/envs/xai_for_rs

# Ensure python executable is from the correct environment
which python

# Input parameters
gpu=$1
methods_list=$2
config_path=$3
IFS=' ' read -r -a methods <<< "$methods_list"

echo "Running on GPU $gpu with config path $config_path"

# Iterate over each method and a set of predefined seeds
for method in "${methods[@]}"; do
  for seed in 42 43 44; do
    echo "Running with method $method on gpu $gpu with seed $seed"

        # Run training with MSE distance and different lambda values
        python3 src/main.py run-training $config_path --explanation-method $method --gpu $gpu --mode rrr --random-seed $seed --rrr-distance mse --rrr-lambda 1
        python3 src/main.py run-training $config_path --explanation-method $method --gpu $gpu --mode rrr --random-seed $seed --rrr-distance mse --rrr-lambda 10

        # Run training with elementwise distance and different lambda values
        python3 src/main.py run-training $config_path --explanation-method $method --gpu $gpu --mode rrr --random-seed $seed --rrr-distance elementwise --rrr-lambda 1
        python3 src/main.py run-training $config_path --explanation-method $method --gpu $gpu --mode rrr --random-seed $seed --rrr-distance elementwise --rrr-lambda 10
  done
done

#  USAGE: ./run_rrr.sh 0 "method1 method2 method3"  config/deepglobe_vgg_config.yml