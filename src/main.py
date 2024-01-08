import argparse
import os
import sys

import pytorch_lightning as pl
import torch.multiprocessing

from config_utils import parse_config

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"Added {project_root} to path.")

# Trick linter
from src.training.train import train  # noqa: E402
from src.xai.explanations.generate_explanations import generate_explanations  # noqa: E402
from xai.metrics.evaluate_explanation_methods import evaluate_explanation_methods  # noqa: E402
from utility.cluster_logging import logger  # noqa: E402
from visualization.visualize import visualize  # noqa: E402

# Fix all seeds with lightning
pl.seed_everything(42)

torch.multiprocessing.set_sharing_strategy(
    "file_system"
)  # handle too many open files error


def main(
    config_path,
    training=False,
    explanations=False,
    evaluations=False,
    visualizations=False,
    debug=False,
):
    logger.debug("In main")
    general_config = parse_config(config_path, project_root)
    general_config["debug"] = debug

    # plot_dataset_distribution_zarr(general_config)
    # plot_pixel_distribution_zarr(general_config)

    logger.debug(f"General config: {general_config}")
    if training:
        train(general_config)

    if visualizations:
        visualize(general_config)

    if explanations:
        generate_explanations(general_config)

    if evaluations:
        evaluate_explanation_methods(general_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training", action="store_true", help="Enable training")
    parser.add_argument(
        "--explanations", action="store_true", help="Generate explanations"
    )
    parser.add_argument(
        "--evaluations", action="store_true", help="Evaluate explanation methods"
    )

    # add visualization flag
    parser.add_argument(
        "--visualizations", action="store_true", help="Visualize explanation methods"
    )

    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )

    default_config_path = os.path.join(project_root, "config", "general_config.yml")

    parser.add_argument(
        "--config_path",
        type=str,
        default=default_config_path,
        help="Path to the config folder",
    )
    args = parser.parse_args()

    main(
        args.config_path,
        training=args.training,
        explanations=args.explanations,
        evaluations=args.evaluations,
        visualizations=args.visualizations,
        debug=args.debug,
    )
