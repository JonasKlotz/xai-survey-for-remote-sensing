import os
from typing import List

import numpy as np
import torch
import tqdm
from pytorch_lightning import LightningDataModule
from torchvision import transforms

from data.constants import DEEPGLOBE_IDX2NAME
from data.datamodule import DeepGlobeDataModule
from data.torch_vis.torch_vis_datamodules import Caltech101DataModule, MNISTDataModule
from data.torch_vis.torchvis_CONSTANTS import CALTECH101_IDX2NAME, MNIST_IDX2NAME
from data.zarr_handler import get_zarr_dataloader
from utility.cluster_logging import logger


def get_loader_for_datamodule(
    data_module: LightningDataModule, loader_name: str = "test"
):
    """
    This function get a data module and return a dictionary of loaders
    \nInputs=> A data module
    \nOutputs=> A dictionary of loaders
    """
    data_module.prepare_data()
    data_module.setup()

    if loader_name == "test":
        return data_module.test_dataloader()
    elif loader_name == "train":
        return data_module.train_dataloader()
    elif loader_name == "val":
        return data_module.val_dataloader()
    raise ValueError(f"Loader name {loader_name} not supported.")


DATA_PATH = os.path.join(os.getcwd(), "..", "..", "data")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = 8 if torch.cuda.is_available() else 8


def load_data_module(cfg: dict):
    logger.info(f"Loading data module {cfg['dataset_name']}")
    dataset_name = cfg["dataset_name"]
    if dataset_name in dataset_cards:
        data_module = dataset_cards[dataset_name](cfg)
        cfg["task"] = data_module.task
        cfg["num_classes"] = data_module.num_classes
        cfg["input_channels"] = data_module.dims[0]
        return data_module, cfg
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")


def get_mnist_data_module(cfg):
    module = MNISTDataModule(
        data_dir=cfg["da"], num_workers=NUM_WORKERS, batch_size=BATCH_SIZE
    )

    return module


def get_deepglobe_data_module(cfg):
    means = [0.4095, 0.3808, 0.2836]
    stds = [0.1509, 0.1187, 0.1081]

    # create normal torch transforms
    transforms_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds),
        ]
    )

    data_module = DeepGlobeDataModule(cfg, transforms_train, transforms_train)

    return data_module


def get_caltech101_data_module(cfg):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # define transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    datamodule = Caltech101DataModule(cfg, train_transform)
    return datamodule


dataset_cards = {
    "mnist": get_mnist_data_module,
    "deepglobe": get_deepglobe_data_module,
    "caltech101": get_caltech101_data_module,
}


def calculate_dataset_distribution(cfg, data_loader):
    """
    Calculate the distribution of the dataset

    Calculate the distribution of the dataset using the dataloader
    Also calculate the mean and std of the dataset
    also calculate the pixel distribution of the dataset

    """
    batchsize = cfg["data"]["batch_size"]
    label_counts = torch.zeros(6)
    pixel_counts = torch.zeros(6)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_batches = 0

    for batch in tqdm.tqdm(data_loader, total=len(data_loader)):
        x_batch, y_batch, idx, segmentation = batch

        unique, counts = torch.unique(segmentation, return_counts=True)
        # ensure that the counts have the same shape as the label counts
        counts = torch.cat(
            [
                counts,
                torch.zeros(label_counts.shape[0] - counts.shape[0], dtype=torch.int64),
            ]
        )

        label_counts += y_batch.sum(dim=0) / batchsize
        pixel_counts += counts / batchsize
        mean += x_batch.mean(dim=(0, 2, 3))
        std += x_batch.std(dim=(0, 2, 3))
        num_batches += 1

    mean /= num_batches
    std /= num_batches

    logger.info(f"Label counts: {label_counts}")
    logger.info(f"Pixel counts: {pixel_counts}")
    logger.info(f"Mean: {mean}")
    logger.info(f"Std: {std}")
    return label_counts, pixel_counts, mean, std


def get_index_to_name(cfg):
    """
    This function returns the index to name mapping of the dataset
    """
    if cfg["dataset_name"] == "deepglobe":
        return DEEPGLOBE_IDX2NAME
    elif cfg["dataset_name"] == "caltech101":
        return CALTECH101_IDX2NAME
    elif cfg["dataset_name"] == "mnist":
        return MNIST_IDX2NAME
    else:
        raise ValueError(f"Dataset {cfg['dataset_name']} not supported.")


def reverse_one_hot_encoding(batches: List[torch.Tensor]):
    """
    This function takes a one hot encoded tensor and returns the class index.

    It works for both multi-class and multi-label classification. The list can contain multiple batches with different shapes.
    so creating a numpy array or tensor is not possible. Therefore, we use a list comprehension to iterate over the batches.
    """
    result = [
        torch.tensor(batch) if not isinstance(batch, torch.Tensor) else batch
        for batch in batches
    ]
    result = [torch.nonzero(batch, as_tuple=False)[:, 0] for batch in result]
    return result


def get_dataloader_from_cfg(cfg, filter_keys=None):
    if cfg["load_from_zarr"]:
        logger.debug(f"Loading dataloader from zarr: {cfg["zarr_path"]}")
        data_loader = get_zarr_dataloader(cfg, filter_keys)

    else:
        # load datamodule
        logger.debug(
            f"Loading dataloader from regular datamodule: {cfg["dataset_name"]}"
        )
        data_module, cfg = load_data_module(cfg)
        data_loader = get_loader_for_datamodule(data_module, loader_name="test")

    logger.debug(
        f"Samples in test loader: {len(data_loader)}, \n"
        f"with batchsize {cfg["data"]["batch_size"]}"
    )
    return cfg, data_loader


def _parse_segments(segments_tensor, dataset_name, num_classes):
    # The quantus framework expects the segments to be boolean tensors.
    if dataset_name == "caltech101":
        # threshold the segments
        segments_tensor = segments_tensor > 0.5
        # convert to 0 1 binary tensor
        segments_tensor = segments_tensor.int()

    elif dataset_name == "deepglobe":
        # create tensor empty with shape (batchsize, num_classes, 1, 120, 120) from (batchsize, 1, 120, 120)
        unsqueezed_segments = np.zeros(
            shape=(
                segments_tensor.shape[0],
                num_classes,
                segments_tensor.shape[1],
                segments_tensor.shape[2],
            )
        )
        for class_index in range(num_classes):
            unsqueezed_segments[:, class_index, :, :] = segments_tensor == class_index
        segments_tensor = unsqueezed_segments
    else:
        raise ValueError("Unknown dataset")
    return segments_tensor


def parse_batch(batch: dict):
    # batch can either come from the zarr or from the dataloader
    if "features" in batch.keys():
        # we have a batch form dataloaders
        return _parse_dataloader_batch(batch)
    elif "x_data" in batch.keys():
        # we parse it as zarr batch
        return _parse_zarr_batch(batch)

    raise ValueError("Batch cannot be parsed...")


def _parse_dataloader_batch(batch: dict):
    image_tensor = batch["features"]
    segments_tensor = batch["segmentations"]
    labels_tensor = batch["targets"]
    index_tensor = batch["index"]
    return image_tensor, labels_tensor, None, segments_tensor, index_tensor, None


def _parse_zarr_batch(batch: dict):
    tmp_batch = batch.copy()
    # batch_dict = dict(zip(keys, batch))
    image_tensor = tmp_batch.pop("x_data").numpy(force=True)
    label_tensor = tmp_batch.pop("y_data").numpy(force=True)
    predicted_label_tensor = tmp_batch.pop("y_pred_data").numpy(force=True)
    if "s_data" in tmp_batch:
        segments_tensor = tmp_batch.pop("s_data").numpy(force=True)
    else:
        segments_tensor = None

    index_tensor = tmp_batch.pop("index_data")
    attributions_dict = tmp_batch  # rename for clarity
    return (
        image_tensor,
        label_tensor,
        predicted_label_tensor,
        segments_tensor,
        index_tensor,
        attributions_dict,
    )
