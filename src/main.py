import argparse
import os
import sys


import pytorch_lightning as pl  # noqa: E402
import torch.multiprocessing  # noqa: E402

from config_utils import parse_config, load_yaml  # noqa: E402

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(project_root)
print(f"Added {project_root} to path.")
quantus_path = os.path.join(project_root, "src/xai/metrics/Quantus")
sys.path.append(quantus_path)
print(f"Added {quantus_path} to path.")

from src.training.train import train  # noqa: E402
from src.xai.explanations.generate_explanations import generate_explanations  # noqa: E402
from xai.metrics.evaluate_explanation_methods import evaluate_explanation_methods  # noqa: E402
from utility.cluster_logging import logger  # noqa: E402
from visualization.visualize import visualize  # noqa: E402
from xai.explanations.debug_explanations import debug_explanations  # noqa: E402

# Fix all seeds with lightning
pl.seed_everything(42)


torch.multiprocessing.set_sharing_strategy(
    "file_system"
)  # handle too many open files error


def main(
    config_path,
    metrics_config_path,
    training=False,
    explanations=False,
    evaluations=False,
    visualizations=False,
    debug=False,
    debug_explanations_bool=False,
    explanation_methods=None,
    gpu=3,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    logger.debug("In main")
    config_path = os.path.join(project_root, config_path)
    metrics_config_path = os.path.join(project_root, metrics_config_path)

    general_config = parse_config(config_path, project_root)
    general_config["debug"] = debug

    # print all cuda devices
    logger.debug(f"Available cuda devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.debug(f"Device {i}: {torch.cuda.get_device_name(i)}")
        logger.debug(
            f"Device {i} memory: {torch.cuda.get_device_properties(i)}"
            f"total_memory: {torch.cuda.get_device_properties(i).total_memory}"
        )

    general_config["device"] = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    general_config["load_from_zarr"] = (
        debug and general_config["dataset_name"] == "deepglobe"
    )

    # plot_dataset_distribution_zarr(general_config)
    # plot_pixel_distribution_zarr(general_config)
    if explanation_methods:
        logger.debug(f"Explanation methods: {explanation_methods}")
        general_config["explanation_methods"] = explanation_methods
        general_config["rrr_explanation"] = explanation_methods[0]
        general_config[
            "experiment_name"
        ] = f'{general_config["experiment_name"]}_{explanation_methods[0]}'
        general_config["training_root_path"] = os.path.join(
            general_config["models_path"], general_config["experiment_name"]
        )

    logger.debug(f"General config: {general_config}")
    if training:
        train(general_config)

    if visualizations:
        visualize(general_config)

    if explanations:
        generate_explanations(general_config)

    if evaluations:
        metrics_config = load_yaml(metrics_config_path)
        evaluate_explanation_methods(general_config, metrics_config)

    if debug_explanations_bool:
        debug_explanations(general_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training", action="store_true", help="Enable training")
    parser.add_argument(
        "--explanations", action="store_true", help="Generate explanations"
    )

    parser.add_argument(
        "--debug-explanations", action="store_true", help="Debug explanations"
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

    parser.add_argument(
        "--metrics_config_path",
        type=str,
        default=os.path.join(
            project_root, "config/metrics_configs/all_metrics_config.yml"
        ),
        help="Path to the metrics config file",
    )
    parser.add_argument(
        "--explanation_methods",
        type=str,
        help="Explanation method to evaluate. Multiple methods can be evaluated by separating them with a comma.",
        nargs="*",  # 0 or more
        default=None,
    )

    parser.add_argument(
        "--gpu",
        type=int,
        help="GPU to use",
        default=3,
    )

    args = parser.parse_args()

    main(
        args.config_path,
        training=args.training,
        explanations=args.explanations,
        evaluations=args.evaluations,
        visualizations=args.visualizations,
        debug=args.debug,
        debug_explanations_bool=args.debug_explanations,
        metrics_config_path=args.metrics_config_path,
        explanation_methods=args.explanation_methods,
        gpu=args.gpu,
    )
