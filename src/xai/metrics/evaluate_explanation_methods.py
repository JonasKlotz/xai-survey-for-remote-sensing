from datetime import datetime
from typing import Union, List, Dict

import numpy as np
import torch
from tqdm import tqdm

from data.data_utils import reverse_one_hot_encoding, get_dataloader_from_cfg
from models.get_models import get_model
from utility.cluster_logging import logger
from xai.explanations.explanation_manager import (
    _explanation_methods,
    ExplanationsManager,
)
from xai.metrics.metrics_manager import MetricsManager


def evaluate_explanation_methods(
    cfg: dict,
    metrics_cfg: dict,
):
    """
    Evaluate Explanation Methods

    """
    logger.debug("Running the metrics for the attributions")
    cfg["method"] = "explain"
    cfg["data"]["num_workers"] = 0
    image_shape = (3, 120, 120) if cfg["dataset_name"] == "deepglobe" else (3, 224, 224)
    one_hot_encoding = cfg["dataset_name"] == "deepglobe"

    cfg, data_loader = get_dataloader_from_cfg(cfg)

    # Load Model
    model = get_model(cfg, self_trained=True).to(cfg["device"])
    model.eval()

    metrics_manager_dict, explanation_manager = setup_metrics_manager(
        cfg=cfg,
        image_shape=image_shape,
        metrics_cfg=metrics_cfg,
        model=model,
    )

    start_time = datetime.now()
    for batch in tqdm(data_loader):
        (
            image_tensor,
            predicted_label_tensor,
            segments_tensor,
            attributions_dict,
        ) = parse_batch(batch)

        if predicted_label_tensor is None:
            predicted_label_tensor = model.prediction_step(image_tensor)

        if attributions_dict is None:
            attributions_dict = explanation_manager.explain_batch(batch)

        if one_hot_encoding:
            predicted_label_tensor = reverse_one_hot_encoding(predicted_label_tensor)

        evaluate_metrics_batch(
            cfg,
            metrics_manager_dict,
            image_tensor,
            predicted_label_tensor,
            segments_tensor,
            attributions_dict,
        )

    end_time = datetime.now()
    logger.debug(f"Time for evaluation: {end_time - start_time}")


def parse_batch(batch: dict):
    # batch can either come from the zarr or from the dataloader
    if "features" in batch.keys():
        # we have a batch form dataloaders
        return _parse_dataloader_batch(batch)
    elif "x_data" in batch.keys():
        # we parse it as zarr batch
        return _parse_zarr_batch(batch)

    raise ValueError("Batch cannot be parsed...")


def setup_metrics_manager(
    cfg,
    image_shape,
    metrics_cfg,
    model,
):
    log_dir = cfg["results_path"] + "/metrics/" + cfg["experiment_name"]

    explanations_dict = {}
    metrics_manager_dict = {}
    for explanation_name in cfg["explanation_methods"]:
        # Initialize the explanation method
        explanation = _explanation_methods[explanation_name](
            model,
            device=cfg["device"],
            multi_label=cfg["task"] == "multilabel",
            num_classes=cfg["num_classes"],
        )
        explanations_dict[explanation_name] = explanation
        # Initialize the metrics manager for the explanation method
        metrics_manager_dict[explanation_name] = MetricsManager(
            model=model,
            cfg=cfg,
            metrics_config=metrics_cfg,
            explanation=explanation,
            log_dir=log_dir,
            image_shape=image_shape,
        )
    explanation_manager = ExplanationsManager(cfg, model)
    return metrics_manager_dict, explanation_manager


def evaluate_metrics_batch(
    cfg: dict,
    metrics_manager_dict: dict,
    image_tensor: Union[np.ndarray, torch.Tensor],
    predicted_label_tensor: Union[List, np.ndarray, torch.Tensor],
    segments_tensor: Union[np.ndarray, torch.Tensor],
    attributions_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
):
    image_tensor = _to_numpy_array(image_tensor)
    segments_tensor = _to_numpy_array(segments_tensor)
    predicted_label_tensor = _to_numpy_array(predicted_label_tensor)

    all_results = {}
    all_time_spend = {}

    for explanation_name in cfg["explanation_methods"]:
        a_batch = attributions_dict["a_" + explanation_name + "_data"]

        a_batch = _to_numpy_array(a_batch)

        results, time_spend = metrics_manager_dict[explanation_name].evaluate_batch(
            x_batch=image_tensor,
            y_batch=predicted_label_tensor,
            a_batch=a_batch,
            s_batch=segments_tensor,
        )
        all_results[explanation_name] = results
        all_time_spend[explanation_name] = time_spend

    return all_results, all_time_spend


def _parse_dataloader_batch(batch: dict):
    image_tensor = batch["features"]
    segments_tensor = batch["segmentations"]
    return image_tensor, None, segments_tensor, None


def _parse_zarr_batch(batch: dict):
    # batch_dict = dict(zip(keys, batch))
    image_tensor = batch.pop("x_data").numpy(force=True)
    _ = batch.pop("y_data").numpy(force=True)
    predicted_label_tensor = batch.pop("y_pred_data").numpy(force=True)
    if "s_data" in batch:
        segments_tensor = batch.pop("s_data").numpy(force=True)
    else:
        segments_tensor = None

    _ = batch.pop("index_data")
    attributions_dict = batch  # rename for clarity
    return image_tensor, predicted_label_tensor, segments_tensor, attributions_dict


def _to_numpy_array(input_tensor):
    if input_tensor is None:
        return None
    elif isinstance(input_tensor, np.ndarray):
        return input_tensor
    elif isinstance(input_tensor, torch.Tensor):
        return input_tensor.numpy(force=True)
    elif isinstance(input_tensor, list):
        return [_to_numpy_array(x) for x in input_tensor]
    else:
        return np.asarray(input_tensor)
