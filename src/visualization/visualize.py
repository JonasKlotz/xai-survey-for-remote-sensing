import torch
from tqdm import tqdm

from data.data_utils import get_index_to_name, parse_batch
from data.zarr_handler import get_zarr_dataloader
from models.get_models import get_model
from utility.cluster_logging import logger
from visualization.explanation_visualizer import ExplanationVisualizer


def visualize(cfg: dict):
    logger.debug(f"Visualizing explanations from {cfg['zarr_path']}")
    cfg["method"] = "visualize"

    # load model
    model = get_model(cfg, self_trained=True)

    model = model.double()
    model.eval()

    cfg["data"]["batch_size"] = 1  # we only want to visualize one sample at a time
    # no filter keys as we need all data
    data_loader = get_zarr_dataloader(
        cfg,
    )
    _ = data_loader.zarr_keys
    index2name = get_index_to_name(cfg)
    explanation_visualizer = ExplanationVisualizer(cfg, model, index2name)
    cfg[
        "cgf_save_path"
    ] = f"{cfg['results_path']}/visualizations/{cfg['experiment_name']}"

    for batch in tqdm(data_loader):
        #
        batch = parse_batch(batch)
        batch_tensor_list = batch[:-1]
        batch_tensor_list = [torch.tensor(t).squeeze() for t in batch_tensor_list]

        attributions_dict = batch[-1]
        attributions_dict = {
            k: torch.tensor(v).squeeze() for k, v in attributions_dict.items()
        }

        # unpack the batch
        (
            image_tensor,
            true_labels,
            predicted_label_tensor,
            segments_tensor,
            index_tensor,
        ) = batch_tensor_list

        # here we can either supply the labels or the predictions
        explanation_visualizer.visualize(
            attrs=attributions_dict,
            image_tensor=image_tensor,
            label_tensor=true_labels,
            segmentation_tensor=segments_tensor,
            predictions_tensor=predicted_label_tensor,
            show=True,
            task=cfg["task"],
        )
        explanation_visualizer.save_last_fig(name=f"sample_{index_tensor}")
        break
