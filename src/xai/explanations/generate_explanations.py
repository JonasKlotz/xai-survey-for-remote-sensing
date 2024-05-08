import torch
import tqdm

from data.data_utils import (
    get_dataloader_from_cfg,
    get_index_to_name,
)
from models.get_models import get_model
from utility.cluster_logging import logger
from visualization.explanation_visualizer import ExplanationVisualizer
from xai.explanations.explanation_manager import ExplanationsManager


def generate_explanations(cfg: dict):
    logger.debug("Generating explanations")
    logger.debug(f"Using CUDA: {torch.cuda.is_available()}")
    logger.debug(f"Using device: {cfg['device']}")
    cfg["method"] = "explain"
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 1  # batch size must be 1 for explanations

    cfg["results_path"] = f"/media/storagecube/jonasklotz/results/{cfg["dataset_name"]}"
    logger.debug(f"Updated for Results path: {cfg['results_path']}")

    cfg, data_loader = get_dataloader_from_cfg(cfg, loader_name="test")

    # load model
    model = get_model(cfg, self_trained=True).to(cfg["device"])
    model.eval()

    explanation_manager = ExplanationsManager(cfg, model, save=True)
    explanation_visualizer = ExplanationVisualizer(cfg, get_index_to_name(cfg))

    for batch in tqdm.tqdm(data_loader):
        batch_dict = explanation_manager.explain_batch(batch)
        explanation_visualizer.visualize_from_batch_dict(
            batch_dict, show=False, skip_non_multilabel=False
        )
        explanation_visualizer.save_last_fig(
            name=f"sample_{batch_dict["index_data"][0]}", format="png"
        )
