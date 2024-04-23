#!/bin/bash

source /home/jonas/miniconda3/bin/activate /home/jonas/micromamba/envs/xai_for_rs

which python

gpu=$1
method=$2

echo "Running with method $method on gpu $gpu"

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 42 --rrr-distance mse --rrr-lambda 1

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 43 --rrr-distance mse --rrr-lambda 1

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 44 --rrr-distance mse --rrr-lambda 1

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 42 --rrr-distance mse --rrr-lambda 10

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 43 --rrr-distance mse --rrr-lambda 10

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 44 --rrr-distance mse --rrr-lambda 10

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 42 --rrr-distance elementwise --rrr-lambda 1

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 43 --rrr-distance elementwise --rrr-lambda 1

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 44 --rrr-distance elementwise --rrr-lambda 1

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 42 --rrr-distance elementwise --rrr-lambda 10

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 43 --rrr-distance elementwise --rrr-lambda 10

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method $method --gpu $gpu  --mode rrr  --random-seed 44 --rrr-distance elementwise --rrr-lambda 10