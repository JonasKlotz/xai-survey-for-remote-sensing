import os

import torch
from pytorch_lightning import LightningDataModule
from torchvision import transforms
import tqdm
from data.datamodule import MNISTDataModule, DeepGlobeDataModule
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
        return dataset_cards[dataset_name](cfg)
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


dataset_cards = {
    "mnist": get_mnist_data_module,
    "deepglobe": get_deepglobe_data_module,
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
