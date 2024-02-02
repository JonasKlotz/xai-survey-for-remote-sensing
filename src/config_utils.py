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
    configs = load_yaml(config_path)
    configs = add_important_paths_to_cfg(configs, project_root)
    return configs


def load_yaml(config_path):
    with open(config_path, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs


def save_config_to_yaml(config: dict, path: str):
    with open(path, "w") as f:
        yaml.dump(config, f)


def add_important_paths_to_cfg(config: dict, project_root: str):
    for key in ["data", "logs", "results", "models", "visualization"]:
        config[f"{key}_path"] = os.path.join(project_root, key)

    if "model_path" in config:
        config["model_path"] = os.path.join(project_root, config["model_path"])
    if "zarr_path" in config:
        config["zarr_path"] = os.path.join(project_root, config["zarr_path"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config["timestamp"] = timestamp
    config[
        "experiment_name"
    ] = f'{config["dataset_name"]}_{config["model_name"]}_{timestamp}'
    config["training_root_path"] = os.path.join(
        config["models_path"], config["experiment_name"]
    )
    config = _add_task(config)
    return config


def _add_task(config):
    if config["dataset_name"] == "deepglobe":
        config["task"] = "multilabel"
    else:
        config["task"] = "multiclass"
    return config
