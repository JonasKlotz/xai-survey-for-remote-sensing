import numpy as np

from data.constants import DEEPGLOBE_IDX2NAME
from data.zarr_handler import load_most_recent_batches
from models.get_models import get_model
from utility.cluster_logging import logger
from visualization.explanation_visualizer import ExplanationVisualizer


def visualize(cfg: dict):
    logger.debug("Visualizing explanations")

    # load model
    model = get_model(
        cfg,
        num_classes=cfg["num_classes"],
        input_channels=cfg["input_channels"],  # data_module.dims[0],
        pretrained=True,
    )
    model.eval()

    if cfg["debug"]:
        model = model.double()

        all_zarrs = load_most_recent_batches(results_dir=cfg["results_path"])

        explanation_visualizer = ExplanationVisualizer(cfg, model, DEEPGLOBE_IDX2NAME)
        sample_size = len(all_zarrs["x_batch"])

        for i in range(sample_size):
            image_tensor = all_zarrs["x_batch"][i][:]
            label_tensor = all_zarrs["y_batch"][i][:]
            segmentation_tensor = all_zarrs["s_batch"][i][:]
            # all zarrs with a key starting with a_batch are attributions
            attributions = {}
            for key, value in all_zarrs.items():
                if key.startswith("a_batch"):
                    attributions[key] = value[i][:]
            if np.sum(label_tensor) < 2:
                continue
            explanation_visualizer.visualize_multi_label_classification(
                image_tensor=image_tensor,
                label_tensor=label_tensor,
                segmentation_tensor=segmentation_tensor,
                attrs=attributions,
                show=False,
            )

            explanation_visualizer.save_last_fig(name=f"sample_{i}")