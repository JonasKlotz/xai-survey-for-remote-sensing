from tqdm import tqdm

from data.data_utils import get_index_to_name, get_dataloader_from_cfg
from data.zarr_handler import get_zarr_dataloader
from models.get_models import get_model
from utility.cluster_logging import logger
from visualization.explanation_visualizer import ExplanationVisualizer

import plotly.io as pio
import plotly.graph_objects as go

from xai.explanations.explanation_manager import ExplanationsManager

pio.templates["my_modification"] = go.layout.Template(layout=dict(font={"size": 20}))

# Then combine your modification with any of the
# available themes like this:
pio.templates.default = "plotly_white+my_modification"


def visualize(cfg: dict, model=None):
    cfg["method"] = "visualize"

    if not model:
        # load model
        model = get_model(cfg, self_trained=True)

        model = model.double()
        model.eval()

    cfg["data"]["batch_size"] = 1  # we only want to visualize one sample at a time

    if cfg["generate_explanations"]:
        logger.info("Generating explanations")
        explanation_manager = ExplanationsManager(cfg, model)
        cfg, data_loader = get_dataloader_from_cfg(cfg, loader_name="val")
    else:
        logger.info(
            f"Not generating explanations, loading from zarr: {cfg['zarr_path']}"
        )
        # no filter keys as we need all data
        data_loader = get_zarr_dataloader(
            cfg,
        )
        _ = data_loader.zarr_keys

    index2name = get_index_to_name(cfg)

    explanation_visualizer = ExplanationVisualizer(cfg, model, index2name)
    logger.info("Starting visualization")
    logger.info(f"Saving to {explanation_visualizer.output_path}")
    for i, batch_dict in enumerate(tqdm(data_loader)):
        if cfg["generate_explanations"]:
            batch_dict = explanation_manager.explain_batch(
                batch_dict, explain_all=False
            )
        explanation_visualizer.visualize_from_batch_dict(batch_dict, show=False)

        explanation_visualizer.save_last_fig(
            name=f"sample_{batch_dict["index_data"].item()}", format="png"
        )
