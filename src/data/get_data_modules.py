import os

import torch

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
    module = MNISTDataModule(
        data_dir=data_dir, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE
    )

    return module


dataset_cards = {
    "mnist": get_mnist_data_module,
}
