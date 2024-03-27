from tqdm import tqdm

from data.data_utils import get_index_to_name
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

    explanation_manager = ExplanationsManager(cfg, model)

    cfg[
        "cgf_save_path"
    ] = f"{cfg['results_path']}/visualizations/{cfg['experiment_name']}"

    for i, batch_dict in enumerate(tqdm(data_loader)):
        batch_dict = explanation_manager.explain_batch(batch_dict, explain_all=False)
        explanation_visualizer.visualize_from_batch_dict(batch_dict, show=True)

        explanation_visualizer.save_last_fig(
            name=f"sample_{batch_dict["index_data"].item()}", format="png"
        )
        break
