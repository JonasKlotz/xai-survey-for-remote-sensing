#!/bin/bash

source /home/jonas/miniconda3/bin/activate /home/jonas/micromamba/envs/xai_for_rs

which python

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method gradcam --mode rrr --gpu 2 --random-seed 43 --rrr-distance mse --rrr-lambda 1

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method gradcam --mode rrr --gpu 2 --random-seed 44 --rrr-distance mse --rrr-lambda 1

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method gradcam --mode rrr --gpu 2 --random-seed 42 --rrr-distance elementwise --rrr-lambda 10

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method gradcam --mode rrr --gpu 2 --random-seed 43 --rrr-distance elementwise --rrr-lambda 10

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method gradcam --mode rrr --gpu 2 --random-seed 44 --rrr-distance elementwise --rrr-lambda 10

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method gradcam --mode rrr --gpu 2 --random-seed 42 --rrr-distance mse --rrr-lambda 10

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method gradcam --mode rrr --gpu 2 --random-seed 43 --rrr-distance mse --rrr-lambda 10

python3 src/main.py run-training config/caltech_vgg_config.yml --explanation-method gradcam --mode rrr --gpu 2 --random-seed 44 --rrr-distance mse --rrr-lambda 10
