from datetime import datetime
from typing import Union, List, Dict

import numpy as np
import torch
from tqdm import tqdm

from data.data_utils import (
    reverse_one_hot_encoding,
    get_dataloader_from_cfg,
    _parse_segments,
    parse_batch,
)
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
    Evaluate Explanation Methods.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing all the necessary parameters.
    metrics_cfg : dict
        Configuration dictionary containing all the necessary parameters for metrics.
    """
    logger.debug("Running the metrics for the attributions")
    cfg["method"] = "explain"

    # enforce single batch size and no workers
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 1

    image_shape = (3, 120, 120) if cfg["dataset_name"] == "deepglobe" else (3, 224, 224)

    one_hot_encoding = cfg["dataset_name"] == "deepglobe"
    multilabel = cfg["task"] == "multilabel"

    recompute_attributions = True
    skip_wrong_predictions = True
    skip_samples_with_single_label = True

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

    # max_iterations = 200 // cfg["data"]["batch_size"]
    i = 0

    start_time = datetime.now()
    for batch in tqdm(data_loader):
        (
            image_tensor,
            true_labels,
            predicted_label_tensor,
            segments_tensor,
            _,
            attributions_dict,
        ) = parse_batch(batch)

        if predicted_label_tensor is None:
            predicted_label_tensor = model.prediction_step(
                image_tensor.to(cfg["device"])
            )

        # Cast all to NP
        image_tensor = _to_numpy_array(image_tensor)
        segments_tensor = _to_numpy_array(segments_tensor)
        predicted_label_tensor = _to_numpy_array(predicted_label_tensor)
        true_labels = _to_numpy_array(true_labels)

        if _skip_batch(
            multilabel,
            predicted_label_tensor,
            skip_samples_with_single_label,
            skip_wrong_predictions,
            true_labels,
        ):
            continue

        if attributions_dict is None or recompute_attributions:
            attributions_dict = explanation_manager.explain_batch(batch)

        if one_hot_encoding:
            # The quantus framework expects the targets not to be one-hot-encoded.
            predicted_label_tensor = reverse_one_hot_encoding(predicted_label_tensor)

        if segments_tensor is not None:
            # parse the segments to quantus format
            segments_tensor = _parse_segments(cfg, segments_tensor)

        try:
            evaluate_metrics_batch(
                cfg,
                metrics_manager_dict,
                image_tensor,
                predicted_label_tensor,
                segments_tensor,
                attributions_dict,
                true_labels,
            )

        except Exception as e:
            logger.error(f"Error in batch {i}: {e}")
            continue

        i += cfg["data"]["batch_size"]

    end_time = datetime.now()
    logger.debug(f"Time for evaluation: {end_time - start_time}")


def setup_metrics_manager(
    cfg,
    image_shape,
    metrics_cfg,
    model,
):
    """
    Sets up the metrics manager.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing all the necessary parameters.
    image_shape : tuple
        Shape of the input images.
    metrics_cfg : dict
        Configuration dictionary containing all the necessary parameters for metrics.
    model : torch.nn.Module
        The model to be used.

    Returns
    -------
    dict, ExplanationsManager
        Dictionary of MetricsManager objects and an ExplanationsManager object.
    """
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
    explanation_manager = ExplanationsManager(cfg, model, save=False)
    return metrics_manager_dict, explanation_manager


def evaluate_metrics_batch(
    cfg: dict,
    metrics_manager_dict: dict,
    image_tensor: Union[np.ndarray, torch.Tensor],
    predicted_label_tensor: Union[List, np.ndarray, torch.Tensor],
    segments_tensor: Union[np.ndarray, torch.Tensor],
    attributions_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
    true_labels: Union[np.ndarray, torch.Tensor],
):
    """
    Evaluates the metrics for a batch.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing all the necessary parameters.
    metrics_manager_dict : dict
        Dictionary of MetricsManager objects.
    image_tensor : Union[np.ndarray, torch.Tensor]
        Tensor of images.
    predicted_label_tensor : Union[List, np.ndarray, torch.Tensor]
        Tensor of predicted labels.
    segments_tensor : Union[np.ndarray, torch.Tensor]
        Tensor of segments.
    attributions_dict : Dict[str, Union[np.ndarray, torch.Tensor]]
        Dictionary of attributions.

    Returns
    -------
    dict, dict
        Dictionary of results and dictionary of time spent for each explanation method.
    """
    image_tensor = _to_numpy_array(image_tensor)
    segments_tensor = _to_numpy_array(segments_tensor)
    predicted_label_tensor = _to_numpy_array(predicted_label_tensor)

    all_results = {}
    all_time_spend = {}

    for explanation_name in cfg["explanation_methods"]:
        a_batch = _to_numpy_array(attributions_dict["a_" + explanation_name + "_data"])
        results, time_spend = metrics_manager_dict[explanation_name].evaluate_batch(
            x_batch=image_tensor,
            y_batch=predicted_label_tensor,
            a_batch=a_batch,
            s_batch=segments_tensor,
            y_true_batch=true_labels,
        )
        all_results[explanation_name] = results
        all_time_spend[explanation_name] = time_spend

    return all_results, all_time_spend


def _to_numpy_array(input_tensor):
    """
    Converts the input tensor to a numpy array.

    Parameters
    ----------
    input_tensor : Union[np.ndarray, torch.Tensor, list]
        Input tensor.

    Returns
    -------
    Union[np.ndarray, list]
        Numpy array or list of numpy arrays.
    """
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


def _skip_batch(
    multilabel,
    predicted_label_tensor,
    skip_samples_with_single_label,
    skip_wrong_predictions,
    true_labels,
):
    """
    Determines whether to skip the current batch based on the given conditions.

    Parameters
    ----------
    multilabel : bool
        Indicates whether the task is multilabel or not.
    predicted_label_tensor : tensor
        Tensor of predicted labels.
    skip_samples_with_single_label : bool
        Flag to skip samples with a single label.
    skip_wrong_predictions : bool
        Flag to skip wrong predictions.
    true_labels : tensor
        Tensor of true labels.

    Returns
    -------
    bool
        True if the batch should be skipped, False otherwise.
    """
    if skip_wrong_predictions and _is_wrong_prediction(
        multilabel, predicted_label_tensor, true_labels
    ):
        return True
    if skip_samples_with_single_label and multilabel:
        if isinstance(predicted_label_tensor, np.ndarray):
            return np.sum(predicted_label_tensor) == 1
        if isinstance(predicted_label_tensor, torch.Tensor):
            return torch.sum(predicted_label_tensor) == 1
    return False


def _is_wrong_prediction(multilabel, predicted_label_tensor, true_labels):
    """
    Checks if the prediction is wrong.

    Parameters
    ----------
    multilabel : bool
        Indicates whether the task is multilabel or not.
    predicted_label_tensor : tensor
        Tensor of predicted labels.
    true_labels : tensor
        Tensor of true labels.

    Returns
    -------
    bool
        True if the prediction is wrong, False otherwise.
    """
    if multilabel:
        return _is_multilabel_wrong_prediction(predicted_label_tensor, true_labels)
    return predicted_label_tensor != true_labels


def _is_multilabel_wrong_prediction(predicted_label_tensor, true_labels):
    """
    Checks if the multilabel prediction is wrong.

    Parameters
    ----------
    predicted_label_tensor : tensor
        Tensor of predicted labels.
    true_labels : tensor
        Tensor of true labels.

    Returns
    -------
    bool
        True if the multilabel prediction is wrong, False otherwise.
    """
    if isinstance(predicted_label_tensor, np.ndarray):
        return not np.all(np.equal(predicted_label_tensor, true_labels))
    if isinstance(predicted_label_tensor, torch.Tensor):
        return not torch.equal(
            predicted_label_tensor, true_labels.to(predicted_label_tensor.device)
        )
    return False
