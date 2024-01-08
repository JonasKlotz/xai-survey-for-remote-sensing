from tqdm import tqdm

from data.constants import DEEPGLOBE_IDX2NAME
from data.zarr_handler import get_zarr_dataloader
from models.get_models import get_model
from utility.cluster_logging import logger
from visualization.explanation_visualizer import ExplanationVisualizer


def visualize(cfg: dict):
    logger.debug(f"Visualizing explanations from {cfg['batches_path']}")

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

        # load model
        model = get_model(
            cfg,
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],  # data_module.dims[0],
            self_trained=True,
        )
        model.eval()

        model = model.double()
        # no filter keys as we need all data
        data_loader, keys = get_zarr_dataloader(
            cfg,
        )

        explanation_visualizer = ExplanationVisualizer(cfg, model, DEEPGLOBE_IDX2NAME)

        for batch in tqdm(data_loader):
            # split batch with keys
            batch_dict = dict(zip(keys, batch))
            # attr_dict =

            # here we can either supply the labels or the predictions
            explanation_visualizer.visualize_multi_label_classification(
                image_tensor=batch_dict["x_data"],
                label_tensor=batch_dict["y_data"],
                prediction_tensor=batch_dict["y_pred_data"],
                segments_tensor=batch_dict["s_data"],
            )

            explanation_visualizer.save_last_fig(
                name=f"sample_{batch_dict['index_data']}"
            )
