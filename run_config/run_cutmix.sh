#!/bin/bash

source /home/jonas/miniconda3/bin/activate /home/jonas/micromamba/envs/xai_for_rs

which python

gpu=$1
method=$2

echo "Running with method $method on gpu $gpu"

python3 src/main.py run-training config/deepglobe_vgg_config.yml --explanation-method $method --gpu $gpu  --mode cutmix  --random-seed 42 --min-aug-area 0.1 --max-aug-area 0.5

python3 src/main.py run-training config/deepglobe_vgg_config.yml --explanation-method $method --gpu $gpu  --mode cutmix  --random-seed 42 --min-aug-area 0.3 --max-aug-area 0.7

python3 src/main.py run-training config/deepglobe_vgg_config.yml --explanation-method $method --gpu $gpu  --mode cutmix  --random-seed 42 --min-aug-area 0.1 --max-aug-area 0.5 --model-name resnet

python3 src/main.py run-training config/deepglobe_vgg_config.yml --explanation-method $method --gpu $gpu  --mode cutmix  --random-seed 42 --min-aug-area 0.3 --max-aug-area 0.7 --model-name resnet
