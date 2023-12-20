import numpy as np
import torch
import tqdm
import yaml
from PIL import Image
from torchvision import transforms

from data.data_utils import (
    get_loader_for_datamodule,
    load_data_module,
    calculate_dataset_distribution,
)
from data.zarr_handler import load_most_recent_batches
from models.get_models import get_model
from utility.cluster_logging import logger
from xai.xai_methods.explanation_manager import ExplanationsManager


def load_test_imagenet_image(idx_to_labels, image_tensor):
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # inv_normalize = transforms.Normalize(
    #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    #     std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    # )
    img = Image.open("/home/jonasklotz/Studys/MASTERS/XAI_PLAYGROUND/img/swan.jpeg")
    transformed_img = transform(img)
    image_tensor = transform_normalize(transformed_img)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def generate_explanations(cfg: dict):
    logger.debug("Generating explanations")

    # load model

    from_zarr = True

    if from_zarr:
        model = get_model(
            cfg,
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],  # data_module.dims[0],
            pretrained=True,
        ).double()
        model.eval()

        all_zarrs = load_most_recent_batches(results_dir=cfg["results_path"])
        for key, value in all_zarrs.items():
            all_zarrs[key] = value[:]

        index = np.arange(0, len(all_zarrs["x_batch"]))
        batchsize = cfg["data"]["batch_size"]
        explanation_manager = ExplanationsManager(cfg, model)

        for i in range(int(len(index) / batchsize) - 1):
            batch = (
                all_zarrs["x_batch"][i * batchsize : (i + 1) * batchsize],
                all_zarrs["y_batch"][i * batchsize : (i + 1) * batchsize],
                index[i * batchsize : (i + 1) * batchsize],
                all_zarrs["s_batch"][i * batchsize : (i + 1) * batchsize],
            )
            # convert to tensor
            batch = tuple(map(torch.tensor, batch))
            # convert to double
            batch = tuple(map(lambda x: x.double(), batch))
            # if cude available, move to cuda
            if torch.cuda.is_available():
                batch = tuple(map(lambda x: x.cuda(), batch))

            explanation_manager.explain_batch(batch)

    else:
        # load datamodule
        data_module = load_data_module(cfg)
        train__loader = get_loader_for_datamodule(data_module, loader_name="train")
        test_loader = get_loader_for_datamodule(data_module, loader_name="test")
        val_loader = get_loader_for_datamodule(data_module, loader_name="val")

        logger.debug(f"Samples in train loader: {len(train__loader)}")
        logger.debug(f"Samples in test loader: {len(test_loader)}")
        logger.debug(f"Samples in val loader: {len(val_loader)}")
        if from_zarr:
            calculate_dataset_distribution(cfg, train__loader)
            calculate_dataset_distribution(cfg, test_loader)
            calculate_dataset_distribution(cfg, val_loader)

        i = 0
        model = None
        explanation_manager = ExplanationsManager(cfg, model)

        for batch in tqdm.tqdm(test_loader):
            explanation_manager.explain_batch(batch)
            i += 1
            if i == 10:
                break


if __name__ == "__main__":
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

    generate_explanations(config)
