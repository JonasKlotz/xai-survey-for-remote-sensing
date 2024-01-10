import torch
import yaml
from tqdm import tqdm

from data.zarr_handler import get_zarr_dataloader
from models.get_models import get_model
from utility.cluster_logging import logger
from xai.metrics.metrics_manager import MetricsManager
from xai.explanations.explanation_manager import _explanation_methods


device_string = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_explanation_methods(cfg: dict, load_precomputed: bool = True):
    """
    Evaluate Explanation Methods

    """
    logger.debug(f"Evaluating explanations from {cfg['zarr_path']}")
    # set batch size to 1 for evaluation
    cfg["data"]["batch_size"] = 4
    data_loader, keys = get_zarr_dataloader(
        cfg,
    )
    model = get_model(
        cfg,
        num_classes=cfg["num_classes"],
        input_channels=cfg["input_channels"],  # data_module.dims[0],
        self_trained=True,
    )
    model.eval()

    metrics_manager_dict = {}
    for explanation_name in cfg["explanation_methods"]:
        metrics_manager_dict[explanation_name] = MetricsManager(
            model=model,
            explanation=_explanation_methods[explanation_name](model),
            aggregate=True,
            device_string=device_string,
            log=True,
            log_dir=cfg["models_path"],
        )

    for batch in tqdm(data_loader):
        batch_dict = dict(zip(keys, batch))
        image_tensor = batch_dict.pop("x_data")
        label_tensor = batch_dict.pop("y_data")
        _ = batch_dict.pop("y_pred_data")
        segments_tensor = batch_dict.pop("s_data")
        _ = batch_dict.pop("index_data")

        for explanation_name in cfg["explanation_methods"]:
            results = metrics_manager_dict[explanation_name].evaluate_batch_mlc(
                x_batch=image_tensor,
                y_batch=label_tensor,
                a_batch=batch_dict[explanation_name + "_data"],
                s_batch=segments_tensor,
            )
            logger.debug(f"Results: {results}")


def main():
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

    evaluate_explanation_methods(config["evaluations"])


if __name__ == "__main__":
    main()
