import torch
import yaml

from data.constants import DEEPGLOBE_IDX2NAME
from data.zarr_handler import load_batches
from models.get_models import get_model
from utility.cluster_logging import logger
from visualization.explanation_visualizer import ExplanationVisualizer
from xai.explanations.generate_explanations import generate_explanations
from xai.metrics.metrics_manager import MetricsManager
from xai.explanations.explanation_methods.gradcam_impl import GradCamImpl

device_string = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_explanation_methods(cfg: dict, load_precomputed: bool = True):
    """
    Evaluate Explanation Methods

    """
    if not load_precomputed:
        generate_explanations(cfg)

    logger.debug("Loading batches as zarr")
    all_zarrs = load_batches(cfg)
    for key, value in all_zarrs.items():
        all_zarrs[key] = value[:]  # convert to numpy

        # convert y and sbatch to int
        if key in ["y_batch", "s_batch"]:
            all_zarrs[key] = all_zarrs[key].astype(int)

    x_batch = all_zarrs["x_batch"]
    y_batch = all_zarrs["y_batch"]
    s_batch = all_zarrs["s_batch"]

    # test batch
    a_batch_gradcam = all_zarrs["a_batch_gradcam"]
    a_batch_deeplift = all_zarrs["a_batch_deeplift"]
    a_batch_integrated_gradients = all_zarrs["a_batch_integrated_gradients"]

    batch_size = cfg["data"]["batch_size"]
    num_classes = cfg["num_classes"]
    input_channels = cfg["input_channels"]

    vis = ExplanationVisualizer(
        cfg, explanation_method_name="GradCAM", index_to_name=DEEPGLOBE_IDX2NAME
    )

    for batch_index in range(batch_size):
        img = x_batch[batch_index]
        labels = y_batch[batch_index]
        segments = s_batch[batch_index]

        # # if more than two labels are 1
        # if sum(labels) <= 1:
        #     continue

        gradcam_attrs = a_batch_gradcam[batch_index]  # # num classes attributions
        deeplift_attrs = a_batch_deeplift[batch_index]  # # num classes attributions
        integrated_gradients_attrs = a_batch_integrated_gradients[batch_index]
        all_attrs = {
            "gradcam": gradcam_attrs,
            "deeplift": deeplift_attrs,
            "integrated_gradients": integrated_gradients_attrs,
        }
        try:
            vis.visualize_multi_label_classification(
                all_attrs, img, segmentation_tensor=segments, label_tensor=labels
            )
        except Exception as e:
            logger.error(f"Error visualizing: {e}")
            continue

    # load model
    model = get_model(cfg, num_classes=num_classes, input_channels=input_channels)
    model.eval()
    explanation = GradCamImpl(model)

    metrics_manager = MetricsManager(
        model=model,
        explanation=explanation,
        aggregate=True,
        device_string=device_string,
        log=True,
        log_dir=cfg["models_path"],
    )

    all_results = metrics_manager.evaluate_batch(
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch_gradcam,
        s_batch=s_batch,
    )

    logger.debug(all_results)


def main():
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

    evaluate_explanation_methods(config["evaluations"])


if __name__ == "__main__":
    main()
