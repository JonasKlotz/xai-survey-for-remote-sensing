from tqdm import tqdm

from data.constants import DEEPGLOBE_IDX2NAME
from data.zarr_handler import get_zarr_dataloader
from models.get_models import get_model
from utility.cluster_logging import logger
from visualization.explanation_visualizer import ExplanationVisualizer


def visualize(cfg: dict):
    logger.debug(f"Visualizing explanations from {cfg['zarr_path']}")

    # load model
    model = get_model(
        cfg,
        num_classes=cfg["num_classes"],
        input_channels=cfg["input_channels"],  # data_module.dims[0],
        self_trained=True,
    )
    model.eval()

    if cfg["debug"]:
        model = model.double()
        model.eval()

        model = model.double()
        cfg["data"]["batch_size"] = 1  # we only want to visualize one sample at a time
        # no filter keys as we need all data
        data_loader, keys = get_zarr_dataloader(
            cfg,
        )

        explanation_visualizer = ExplanationVisualizer(cfg, model, DEEPGLOBE_IDX2NAME)

        for batch in tqdm(data_loader):
            # squeeze whole batch
            batch = [b.squeeze() for b in batch]
            # split batch with keys
            batch_dict = dict(zip(keys, batch))
            image_tensor = batch_dict.pop("x_data")
            label_tensor = batch_dict.pop("y_data")
            predictions_tensor = batch_dict.pop("y_pred_data")
            segments_tensor = batch_dict.pop("s_data")
            index_tensor = batch_dict.pop("index_data")

            #

            # here we can either supply the labels or the predictions
            explanation_visualizer.visualize_multi_label_classification(
                attrs=batch_dict,
                image_tensor=image_tensor,
                label_tensor=label_tensor,
                segmentation_tensor=segments_tensor,
                predictions_tensor=predictions_tensor,
                show=True,
            )
            break

            explanation_visualizer.save_last_fig(name=f"sample_{index_tensor}")
