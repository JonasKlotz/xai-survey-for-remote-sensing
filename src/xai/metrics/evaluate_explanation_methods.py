from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from data.data_utils import reverse_one_hot_encoding
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

    (
        data_loader,
        explanations_dict,
        keys,
        metrics_manager_dict,
    ) = setup_metrics_evaluation(cfg, metrics_cfg)

    start_time = datetime.now()
    for batch in tqdm(data_loader):
        (
            image_tensor,
            predicted_label_tensor,
            segments_tensor,
            batch_dict,
        ) = _parse_batch(batch, keys)
        # skip if we have a single label
        if np.sum(predicted_label_tensor) <= len(predicted_label_tensor):
            continue

        # todo: remove this it is mostly for debugging? or create other file for this
        generate_attributions = cfg["dataset_name"] == "deepglobe"
        if generate_attributions:
            predicted_label_tensor = reverse_one_hot_encoding(predicted_label_tensor)

        for explanation_name in cfg["explanation_methods"]:
            # sum a_batch from 3 dim to 1
            # todo: this is only necessary for the current zarr, as the a_batch is saved as 3 channel image however
            # the metrics manager expects a 1 channel image. Changed for the next zarrs
            a_batch = batch_dict["a_" + explanation_name + "_data"].squeeze(dim=1)
            a_batch = torch.sum(a_batch, dim=1).numpy(force=True)  # sum over channels

            if generate_attributions:
                # generate attributions
                a_batch = (
                    explanations_dict[explanation_name]
                    .explain_batch(
                        tensor_batch=torch.tensor(image_tensor),
                        target_batch=predicted_label_tensor,
                    )
                    .numpy(force=True)
                )

            _, _ = metrics_manager_dict[explanation_name].evaluate_batch(
                x_batch=image_tensor,
                y_batch=predicted_label_tensor,
                a_batch=a_batch,
                s_batch=segments_tensor,
            )
        break  # todo remove this break
    end_time = datetime.now()
    logger.debug(f"Time for evaluation: {end_time - start_time}")


def _parse_batch(batch, keys):
    batch_dict = dict(zip(keys, batch))
    image_tensor = batch_dict.pop("x_data").numpy(force=True)
    _ = batch_dict.pop("y_data").numpy(force=True)
    predicted_label_tensor = batch_dict.pop("y_pred_data").numpy(force=True)
    if "s_data" in batch_dict:
        segments_tensor = batch_dict.pop("s_data").numpy(force=True)
    else:
        segments_tensor = None
    _ = batch_dict.pop("index_data")
    return image_tensor, predicted_label_tensor, segments_tensor, batch_dict


def setup_metrics_evaluation(cfg, metrics_cfg):
    # set batch size to 2 for evaluation
    cfg["data"][
        "batch_size"
    ] = 2  # batchsize should not be 1 as we squeeze the batch dimension in the metrics manager
    data_loader, keys = get_zarr_dataloader(
        cfg,
    )
    image_shape = (3, 120, 120) if cfg["dataset_name"] == "deepglobe" else (3, 224, 224)
    model = get_model(cfg, self_trained=True).to(cfg["device"])
    model.eval()
    multi_label = cfg["task"] == "multilabel"
    log_dir = cfg["results_path"] + "/metrics/" + cfg["experiment_name"]
    explanations_dict = {}
    # todo debug check does the model apply softmax?
    metrics_manager_dict = {}
    for explanation_name in cfg["explanation_methods"]:
        # Initialize the explanation method
        explanation = _explanation_methods[explanation_name](
            model,
            device=cfg["device"],
            multi_label=multi_label,
            num_classes=cfg["num_classes"],
        )
        explanations_dict[explanation_name] = explanation
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
            image_shape=image_shape,
            num_classes=cfg["num_classes"],
        )
    return data_loader, explanations_dict, keys, metrics_manager_dict
