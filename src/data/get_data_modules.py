import os

import torch

from src.data.data_modules import MNISTDataModule

DATA_PATH = os.path.join(os.getcwd(), "..", "..", "data")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = 8 if torch.cuda.is_available() else 8


def load_data_module(cfg: dict):
    print("Loading data module")
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
    from torchvision import transforms
    from data.tom_data.datamodule import DeepGlobeDataModule
    from data.tom_data.transformations_impl import TransformationsImpl

    transforms_train = TransformationsImpl(
        cfg,
        [
            transforms.ToTensor(),
        ],
    )
    data_module = DeepGlobeDataModule(cfg, transforms_train, transforms_train)

    return data_module


dataset_cards = {
    "mnist": get_mnist_data_module,
    "deepglobe": get_deepglobe_data_module,
}
