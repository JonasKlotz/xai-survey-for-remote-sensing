import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"Added {project_root} to path.")
CONFIGPATH = os.path.join(project_root, "config")

import yaml
import pytorch_lightning as pl

from src.training.train import train
from src.xai.generate_explanations import generate_explanations
from xai.metrics.evaluate_explanation_methods import evaluate_explanation_methods
from utility.cluster_logging import logger

# Fix all seeds with lightning
pl.seed_everything(42)
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')  # handle too many open files error


def parse_config(config_path):
    # read all files in config_path
    file_names = os.listdir(config_path)
    configs = {}
    # load all yaml files
    for file_name in file_names:
        if file_name.endswith(".yaml") or file_name.endswith(".yml"):
            with open(os.path.join(config_path, file_name)) as file:
                loaded_cfg = yaml.load(file, Loader=yaml.FullLoader)
                configs[file_name[:-11]] = loaded_cfg

    configs = add_important_paths_to_cfg(configs, project_root)
    return configs


def add_important_paths_to_cfg(configs: dict, project_root: str):
    data_path = os.path.join(project_root, 'data')
    models_path = os.path.join(project_root, 'models')
    log_path = os.path.join(project_root, 'logs')
    results_path = os.path.join(project_root, 'results')

    for cfg in configs.values():
        cfg['data_path'] = data_path
        cfg['models_path'] = models_path
        cfg['log_path'] = log_path
        cfg['results_path'] = results_path

    return configs


def main():
    logger.debug("In main")
    configs = parse_config(CONFIGPATH)
    general_config = configs["general"]

    logger.debug(f"General config: {general_config}")
    if general_config["training"]:
        train(general_config)

    if general_config["explanations"]:
        generate_explanations(general_config)
        # for expl in general_config['explanations

    if general_config["evaluations"]:
        evaluate_explanation_methods(general_config)


if __name__ == "__main__":
    main()
