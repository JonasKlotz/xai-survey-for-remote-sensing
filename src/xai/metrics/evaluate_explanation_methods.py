from datetime import datetime

import torch
import yaml
from tqdm import tqdm

from data.zarr_handler import get_zarr_dataloader
from models.get_models import get_model
from utility.cluster_logging import logger
from xai.explanations.explanation_manager import _explanation_methods
from xai.metrics.metrics_manager import MetricsManager


def evaluate_explanation_methods(
    cfg: dict, metrics_cfg: dict, load_precomputed: bool = True
):
    """
    Evaluate Explanation Methods

    """
    logger.debug(f"Evaluating explanations from {cfg['zarr_path']}")
    # set batch size to 2 for evaluation
    cfg["data"][
        "batch_size"
    ] = 2  # batchsize should not be 1 as we squeeze the batch dimension in the metrics manager
    data_loader, keys = get_zarr_dataloader(
        cfg,
    )
    model = get_model(cfg, self_trained=True).to(cfg["device"])
    model.eval()

    log_dir = cfg["results_path"] + "/metrics/" + cfg["experiment_name"]
    # todo debug check does the model apply softmax?
    metrics_manager_dict = {}
    for explanation_name in cfg["explanation_methods"]:
        # Initialize the explanation method
        explanation = _explanation_methods[explanation_name](
            model,
            device=cfg["device"],
            multilabel=False,  # todo adapt to multilabel
            num_classes=cfg["num_classes"],
        )
        # Initialize the metrics manager for the explanation method
        metrics_manager_dict[explanation_name] = MetricsManager(
            model=model,
            metrics_config=metrics_cfg,
            explanation=explanation,
            aggregate=True,
            device=cfg["device"],
            log=True,
            log_dir=log_dir,
            task=cfg["task"],
        )
    start_time = datetime.now()
    for batch in tqdm(data_loader):
        #
        batch_dict = dict(zip(keys, batch))
        image_tensor = batch_dict.pop("x_data").numpy(force=True)
        _ = batch_dict.pop("y_data").numpy(force=True)
        predicted_label_tensor = batch_dict.pop("y_pred_data").numpy(force=True)
        if "s_data" in batch_dict:
            segments_tensor = batch_dict.pop("s_data").numpy(force=True)
        else:
            segments_tensor = None
        _ = batch_dict.pop("index_data")

        for explanation_name in cfg["explanation_methods"]:
            # sum a_batch from 3 dim to 1
            # todo: this is only necessary for the current zarr, as the a_batch is saved as 3 channel image however
            # the metrics manager expects a 1 channel image. Changed for the next zarrs
            a_batch = batch_dict["a_" + explanation_name + "_data"].squeeze(dim=1)
            a_batch = torch.sum(a_batch, dim=1).numpy(force=True)  # sum over channels

            _, _ = metrics_manager_dict[explanation_name].evaluate_batch(
                x_batch=image_tensor,
                y_batch=predicted_label_tensor,
                a_batch=a_batch,
                s_batch=segments_tensor,
            )
        break  # todo remove this break
    end_time = datetime.now()
    logger.debug(f"Time for evaluation: {end_time - start_time}")


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
