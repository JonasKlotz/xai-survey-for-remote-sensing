from tqdm import tqdm

from data.data_utils import get_index_to_name
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
    keys = data_loader.zarr_keys
    index2name = get_index_to_name(cfg)
    explanation_visualizer = ExplanationVisualizer(cfg, model, index2name)
    cfg[
        "cgf_save_path"
    ] = f"{cfg['results_path']}/visualizations/{cfg['experiment_name']}"

    for batch in tqdm(data_loader):
        # squeeze whole batch
        batch = [b.squeeze() for b in batch]
        # split batch with keys
        batch_dict = dict(zip(keys, batch))
        image_tensor = batch_dict.pop("x_data")
        label_tensor = batch_dict.pop("y_data")
        predictions_tensor = batch_dict.pop("y_pred_data")
        if "s_data" in batch_dict.keys():
            segments_tensor = batch_dict.pop("s_data")
        else:
            segments_tensor = None
        index_tensor = batch_dict.pop("index_data")

        # here we can either supply the labels or the predictions
        explanation_visualizer.visualize(
            attrs=batch_dict,
            image_tensor=image_tensor,
            label_tensor=label_tensor,
            segmentation_tensor=segments_tensor,
            predictions_tensor=predictions_tensor,
            show=False,
            task=cfg["task"],
        )
        explanation_visualizer.save_last_fig(name=f"sample_{index_tensor}")
