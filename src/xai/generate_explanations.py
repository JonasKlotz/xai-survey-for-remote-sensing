import numpy as np
import torch
import tqdm
import yaml


from data.data_utils import (
    get_loader_for_datamodule,
    load_data_module,
    calculate_dataset_distribution,
)
from data.zarr_handler import load_most_recent_batches
from models.get_models import get_model
from utility.cluster_logging import logger
from xai.xai_methods.explanation_manager import ExplanationsManager


def generate_explanations(cfg: dict):
    logger.debug("Generating explanations")

    # load model
    model = get_model(
        cfg,
        num_classes=cfg["num_classes"],
        input_channels=cfg["input_channels"],  # data_module.dims[0],
        pretrained=True,
    )
    model.eval()
    if cfg["debug"]:
        model = model.double()

        all_zarrs = load_most_recent_batches(results_dir=cfg["results_path"])
        for key, value in all_zarrs.items():
            all_zarrs[key] = value[:]

        index = np.arange(0, len(all_zarrs["x_batch"]))
        batchsize = cfg["data"]["batch_size"]
        explanation_manager = ExplanationsManager(cfg, model)

        for i in range(int(len(index) / batchsize) - 1):
            batch = (
                all_zarrs["x_batch"][i * batchsize : (i + 1) * batchsize],
                all_zarrs["y_batch"][i * batchsize : (i + 1) * batchsize],
                index[i * batchsize : (i + 1) * batchsize],
                all_zarrs["s_batch"][i * batchsize : (i + 1) * batchsize],
            )
            # convert to tensor
            batch = tuple(map(torch.tensor, batch))
            # convert to double
            batch = tuple(map(lambda x: x.double(), batch))
            # if cuda available, move to cuda
            if torch.cuda.is_available():
                batch = tuple(map(lambda x: x.cuda(), batch))

            explanation_manager.explain_batch(batch)

    else:
        # load datamodule
        data_module = load_data_module(cfg)
        train__loader = get_loader_for_datamodule(data_module, loader_name="train")
        test_loader = get_loader_for_datamodule(data_module, loader_name="test")
        val_loader = get_loader_for_datamodule(data_module, loader_name="val")

        logger.debug(f"Samples in train loader: {len(train__loader)}")
        logger.debug(f"Samples in test loader: {len(test_loader)}")
        logger.debug(f"Samples in val loader: {len(val_loader)}")
        if cfg["debug"]:
            calculate_dataset_distribution(cfg, train__loader)
            calculate_dataset_distribution(cfg, test_loader)
            calculate_dataset_distribution(cfg, val_loader)

        explanation_manager = ExplanationsManager(cfg, model)

        for batch in tqdm.tqdm(test_loader):
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
