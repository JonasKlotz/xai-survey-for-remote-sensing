import os

import torch
from pl_bolts.datamodules import CIFAR10DataModule, CityscapesDataModule

from src.data.data_modules import MNISTDataModule



DATA_PATH = os.path.join(os.getcwd(), "..", "..", "data")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = 8 if torch.cuda.is_available() else 8


def load_data_module(dataset_name: str):
    if dataset_name in dataset_cards:
        return dataset_cards[dataset_name]()
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")


def get_mnist_data_module(data_dir: str = DATA_PATH):
    module =  MNISTDataModule(data_dir=data_dir, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

    return module


def get_cifar_data_module(data_dir: str = DATA_PATH):
    return CIFAR10DataModule(data_dir=data_dir, num_workers=4, batch_size=BATCH_SIZE)


def get_cityscapes_data_module(data_dir: str = DATA_PATH):
    return CityscapesDataModule(data_dir=data_dir, num_workers=4, batch_size=BATCH_SIZE)


dataset_cards = {
    "mnist": get_mnist_data_module,
    "cifar10": get_cifar_data_module,
    "cityscapes": get_cityscapes_data_module,
}


# def get_EuroSAT_data_module(data_dir: str = '/home/jonasklotz/Studys/MASTERS/XAI_PLAYGROUND/data/eurosat',
#                             batch_size=64, num_workers=4, download=True):
#     from torchgeo.datamodules import EuroSATDataModule
#     LABELS = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
#               'Residential', 'River', 'SeaLake']
#     module = EuroSATDataModule(root=data_dir, num_workers=num_workers, batch_size=batch_size, download=download)
#     module.num_classes = len(LABELS)
#     module.labels = LABELS
#     return module
