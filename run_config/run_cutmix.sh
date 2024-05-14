  GNU nano 6.2                                                                                            git/xai-survey-for-remote-sensing/run_cutmix.sh
#!/bin/bash

# Activate environment
source /home/jonas/miniconda3/bin/activate /home/jonas/micromamba/envs/xai_for_rs

# Check python executable
which python

# Read GPU, methods list, seed, and config path from command line arguments
gpu=$1
methods_list=$2
seed=$3
config_path=$4

# Convert space-separated methods list into an array
IFS=' ' read -r -a methods <<< "$methods_list"

# Iterate over each method in the array
for method in "${methods[@]}"; do
    echo "Running with method $method on gpu $gpu with config $config_path"

    # Run the training for each configuration
    python3 src/main.py run-training $config_path --explanation-method $method --gpu $gpu --mode cutmix --random-seed $seed --min-aug-area 0.1 --max-aug-area 0.5
    python3 src/main.py run-training $config_path --explanation-method $method --gpu $gpu --mode cutmix --random-seed $seed --min-aug-area 0.3 --max-aug-area 0.7

   done




#  USAGE: ./run_cutmix.sh 0 "method1 method2 method3" 42 config/deepglobe_vgg_config.yml




