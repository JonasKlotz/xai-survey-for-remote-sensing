import os
from datetime import datetime

import yaml


def parse_config(config_path, project_root):
    """
    Parse the config file and add important paths to the config dictionary.
    Parameters
    ----------
    config_path
    project_root

    Returns
    -------

    """
    with open(config_path, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    configs = add_important_paths_to_cfg(configs, project_root)
    return configs


def add_important_paths_to_cfg(config: dict, project_root: str):
    for key in ["data", "models", "logs", "results"]:
        config[f"{key}_path"] = os.path.join(project_root, key)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config["timestamp"] = timestamp
    config[
        "experiment_name"
    ] = f'{config["dataset_name"]}_{config["model_name"]}_{timestamp}'

    return config