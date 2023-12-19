import os
import sys

import pytorch_lightning as pl
import torch.multiprocessing

from config_utils import parse_config

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"Added {project_root} to path.")
CONFIGPATH = os.path.join(project_root, "config")

# Trick linter
from src.training.train import train  # noqa: E402
from src.xai.generate_explanations import generate_explanations  # noqa: E402
from xai.metrics.evaluate_explanation_methods import evaluate_explanation_methods  # noqa: E402
from utility.cluster_logging import logger  # noqa: E402

# Fix all seeds with lightning
pl.seed_everything(42)

torch.multiprocessing.set_sharing_strategy(
    "file_system"
)  # handle too many open files error


def main():
    logger.debug("In main")
    configs = parse_config(CONFIGPATH, project_root)
    general_config = configs["general"]

    # plot_dataset_distribution_zarr(general_config)
    # plot_pixel_distribution_zarr(general_config)

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
