import os

import torch
import tqdm
import yaml

from data.data_utils import (
    get_dataloader_from_cfg,
)
from data.lmdb_handler import LMDBDataHandler
from models.get_models import get_model
from utility.cluster_logging import logger
from xai.explanations.explanation_manager import ExplanationsManager


def generate_explanations(cfg: dict):
    logger.debug("Generating explanations")
    logger.debug(f"Using CUDA: {torch.cuda.is_available()}")
    logger.debug(f"Using device: {cfg['device']}")
    cfg["method"] = "explain"
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 1  # for debugging

    cfg, data_loader = get_dataloader_from_cfg(cfg, loader_name="train")

    # load model
    model = get_model(cfg, self_trained=True).to(cfg["device"])
    model.eval()

    explanation_manager = ExplanationsManager(cfg, model, save=False)

    save_as_segmentations = True

    if save_as_segmentations:
        segmentation_handler_dict = _create_lmdb_handlers(cfg, explanation_manager)

    for batch in tqdm.tqdm(data_loader):
        batch_dict = explanation_manager.explain_batch(batch, explain_all=True)
        if save_as_segmentations:
            _save_segmentations_to_lmdb(
                batch_dict,
                segmentation_handler_dict,
                explanation_manager.explanations.keys(),
            )


def _create_lmdb_handlers(cfg, explanation_manager):
    segmentation_handler_dict = {}
    base_path = f"{cfg['results_path']}/{cfg['experiment_name']}"
    os.makedirs(base_path, exist_ok=True)
    for explanation_method_name in explanation_manager.explanations.keys():
        lmdb_path = f"{base_path}/{explanation_method_name}.lmdb"
        segmentation_handler_dict[explanation_method_name] = LMDBDataHandler(
            path=lmdb_path, write_only=True
        )
    return segmentation_handler_dict


def _save_segmentations_to_lmdb(
    batch_dict, segmentation_handler_dict, explanation_method_names
):
    for explanation_method_name in explanation_method_names:
        attribution_maps = batch_dict[f"a_{explanation_method_name}_data"]
        index = batch_dict["index_data"]
        for ind in range(len(index)):
            write_index = str(index[ind].item())
            write_attribution_map = attribution_maps[ind]
            segmentation_handler_dict[explanation_method_name].append(
                write_index, write_attribution_map
            )


def minmax_normalize_tensor(tensor):
    """
    Min-Max normalizes a PyTorch tensor along all axes.

    Parameters:
    - tensor (torch.Tensor): The input tensor to normalize.

    Returns:
    - torch.Tensor: The Min-Max normalized tensor.
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    # Avoid division by zero
    if tensor_min == tensor_max:
        return torch.zeros_like(tensor)
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor


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
