import torch
import tqdm
import yaml

from data.data_utils import (
    get_dataloader_from_cfg,
)
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

    for batch in tqdm.tqdm(data_loader):
        explanation_manager.explain_batch(batch, explain_all=True)


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
