import torch
import tqdm

from data.data_utils import (
    get_loader_for_datamodule,
    load_data_module,
    get_index_to_name,
)
from data.zarr_handler import get_zarr_dataloader
from models.get_models import get_model
from utility.cluster_logging import logger
from visualization.explanation_visualizer import ExplanationVisualizer
from xai.explanations.explanation_manager import ExplanationsManager


def debug_explanations(cfg: dict):
    logger.debug("Generating explanations")
    logger.debug(f"Using CUDA: {torch.cuda.is_available()}")
    logger.debug(f"Using device: {cfg['device']}")
    cfg["method"] = "explain"
    # set batchsize to 1
    cfg["data"]["batch_size"] = 1

    if cfg["debug"] and cfg["dataset_name"] == "deepglobe":
        # model = model.double()
        data_loader, _ = get_zarr_dataloader(
            cfg,
            filter_keys=[
                "x_data",
                "y_data",
                "index_data",
                "s_data",
            ],  # we only need the actual data
        )

    else:
        # load datamodule
        data_module, cfg = load_data_module(cfg)
        data_loader = get_loader_for_datamodule(data_module, loader_name="test")

        logger.debug(f"Samples in test loader: {len(data_loader)}")

    # load model
    model = get_model(cfg, self_trained=True).to(cfg["device"])

    # from torchvision import models
    # model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    # num_classes = 101
    # input_channels = 3
    # model.features[0] = torch.nn.Conv2d(
    #     input_channels,
    #     64,
    #     kernel_size=(7, 7),
    #     stride=(1, 1),
    #     padding=(1, 1),
    #     bias=False,
    # )
    # model.classifier[-1] = torch.nn.Linear(4096, num_classes)

    model.eval()
    explanation_manager = ExplanationsManager(cfg, model)
    index2name = get_index_to_name(cfg)
    explanation_visualizer = ExplanationVisualizer(cfg, model, index2name)

    for batch in tqdm.tqdm(data_loader):
        batch_dict = explanation_manager.explain_batch(batch)
        batch_dict = {k: v.squeeze() for k, v in batch_dict.items()}

        image_tensor = batch_dict.pop("x_data")
        label_tensor = batch_dict.pop("y_data")
        predictions_tensor = batch_dict.pop("y_pred_data")
        if "s_data" in batch_dict.keys():
            _ = batch_dict.pop("s_data")

        _ = batch_dict.pop("index_data")  # pop that its not handed to the visualizer

        # here we can either supply the labels or the predictions
        explanation_visualizer.visualize(
            attrs=batch_dict,
            image_tensor=image_tensor,
            label_tensor=label_tensor,
            segmentation_tensor=None,
            predictions_tensor=predictions_tensor,
            show=True,
            task=cfg["task"],
        )
        print("Done")
