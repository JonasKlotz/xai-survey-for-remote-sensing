import torch
from tqdm import tqdm

from data.data_utils import get_index_to_name, parse_batch
from data.zarr_handler import get_zarr_dataloader
from models.get_models import get_model
from utility.cluster_logging import logger
from visualization.explanation_visualizer import ExplanationVisualizer

import plotly.io as pio
import plotly.graph_objects as go

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
    cfg[
        "cgf_save_path"
    ] = f"{cfg['results_path']}/visualizations/{cfg['experiment_name']}"

    for i, batch in enumerate(tqdm(data_loader)):
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

        # explanation_visualizer.visualize(
        #     attribution_dict=attributions_dict,
        #     image_tensor=image_tensor,
        #     label_tensor=true_labels,
        #     segmentation_tensor=segments_tensor,
        #     predictions_tensor=predicted_label_tensor,
        #     show=True,
        #     task=cfg["task"],
        # )
        #
        # explanation_visualizer.visualize_top_k_attributions(
        #     attribution_dict=attributions_dict,
        #     image_tensor=image_tensor,
        #     label_tensor=true_labels,
        #     segmentation_tensor=segments_tensor,
        #     predictions_tensor=predicted_label_tensor,
        #     show=True,
        #     task=cfg["task"],
        #     k=0.1,
        #     largest=True,
        # )
        # explanation_visualizer.visualize_top_k_attributions(
        #     attribution_dict=attributions_dict,
        #     image_tensor=image_tensor,
        #     label_tensor=true_labels,
        #     segmentation_tensor=segments_tensor,
        #     predictions_tensor=predicted_label_tensor,
        #     show=True,
        #     task=cfg["task"],
        #     k=0.9,
        #     largest=False,
        # )
        k_list = [0.05, 0.1, 0.15, 0.2]
        explanation_visualizer.visualize_top_k_attributions_with_predictions(
            attribution_dict=attributions_dict,
            image_tensor=image_tensor,
            label_tensor=true_labels,
            segmentation_tensor=segments_tensor,
            predictions_tensor=predicted_label_tensor,
            show=True,
            k_list=k_list,
            remove_top_k_features=False,
            model=model,
            title="Top k attributions present in the Image",
            save_name=f"sample_{index_tensor}",
        )

        explanation_visualizer.visualize_top_k_attributions_with_predictions(
            attribution_dict=attributions_dict,
            image_tensor=image_tensor,
            label_tensor=true_labels,
            segmentation_tensor=segments_tensor,
            predictions_tensor=predicted_label_tensor,
            show=True,
            k_list=k_list,
            remove_top_k_features=True,
            model=model,
            title="Top k attributions removed from the Image",
            save_name=f"sample_{index_tensor}",
        )
        print(f"Visualizing sample {index_tensor}")
        if i == 5:
            break

        # explanation_visualizer.save_last_fig(name=f"sample_{index_tensor}")
