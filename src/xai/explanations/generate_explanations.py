import torch
import tqdm
import yaml

from data.data_utils import (
    get_loader_for_datamodule,
    load_data_module,
)
from data.zarr_handler import get_zarr_dataloader
from models.get_models import get_model
from utility.cluster_logging import logger
from xai.explanations.explanation_manager import ExplanationsManager


def generate_explanations(cfg: dict):
    logger.debug("Generating explanations")
    logger.debug(f"Using CUDA: {torch.cuda.is_available()}")
    logger.debug(f"Using device: {cfg['device']}")
    cfg["method"] = "explain"

    if cfg["debug"] and cfg["dataset_name"] == "deepglobe":
        # model = model.double()
        data_loader, _ = get_zarr_dataloader(
            cfg,
            filter_keys=[
                "x_data",
                "y_data",
                "index_data",
                "s_data",
            ],  # we only need the actual data
        )

    else:
        # load datamodule
        data_module, cfg = load_data_module(cfg)
        data_loader = get_loader_for_datamodule(data_module, loader_name="test")

        logger.debug(f"Samples in test loader: {len(data_loader)}")

    # load model
    model = get_model(
        cfg,
        num_classes=cfg["num_classes"],
        input_channels=cfg["input_channels"],  # data_module.dims[0],
        self_trained=True,
    ).to(cfg["device"])

    model.eval()
    explanation_manager = ExplanationsManager(cfg, model)

    for batch in tqdm.tqdm(data_loader):
        explanation_manager.explain_batch(batch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/explanations_config.yml",
    )

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generate_explanations(config)
