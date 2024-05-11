#!/bin/bash

# Activate environment
source /home/jonas/miniconda3/bin/activate /home/jonas/micromamba/envs/xai_for_rs

python3 -W ignore src/main.py run-generate-explanations config/ben_vgg_config.yml --mode cutmix --gpu 1 --random-seed 42 
python3 -W ignore src/main.py run-generate-explanations config/ben_vgg_config.yml --mode cutmix --gpu 1 --random-seed 43
python3 -W ignore src/main.py run-generate-explanations config/ben_vgg_config.yml --mode cutmix --gpu 1 --random-seed 44 
