import os
from datetime import datetime

import yaml


def parse_config(config_path, project_root):
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
    data_path = os.path.join(project_root, "data")
    models_path = os.path.join(project_root, "models")
    log_path = os.path.join(project_root, "logs")
    results_path = os.path.join(project_root, "results")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for cfg in configs.values():
        cfg["data_path"] = data_path
        cfg["models_path"] = models_path
        cfg["log_path"] = log_path
        cfg["results_path"] = results_path

        cfg["timestamp"] = timestamp

    return configs
