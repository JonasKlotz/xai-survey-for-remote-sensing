from abc import abstractmethod

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from data.torch_vis.torch_vis_datasets import Caltech101Dataset


class TorchVisDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_transforms, target_transforms):
        """DataModule for Data

        Parameters
        ----------
        cfg : dict
        transform_tr : Transform
        transform_te : Transform
        """
        super().__init__()
        self.cfg = cfg
        self.train_transforms = train_transforms
        self.target_transforms = target_transforms

    @abstractmethod
    def get_dataset(
        self,
        dataset_path: str,
        train_transforms,
        target_transforms,
    ):
        raise NotImplementedError

    def setup(self, stage=None):
        # Load the entire dataset
        dataset = self.get_dataset(
            self.cfg["data_path"], self.train_transforms, self.target_transforms
        )

        # Calculate the lengths of the splits
        total_length = len(dataset)
        train_length = int(total_length * 0.7)
        val_length = int(total_length * 0.15)
        test_length = total_length - train_length - val_length

        # Split the dataset
        self.trainset_tr, self.valset, self.testset = random_split(
            dataset, [train_length, val_length, test_length]
        )

    def get_loader(self, dataset, drop_last):
        shuffle = True if dataset == self.trainset_tr else False

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg["data"]["batch_size"],
            num_workers=self.cfg["data"]["num_workers"],
            shuffle=shuffle,
            pin_memory=self.cfg["data"]["pin_memory"],
            drop_last=drop_last,
        )
        return dataloader

    def train_dataloader(self, drop_last=False):
        return self.get_loader(self.trainset_tr, drop_last)

    def val_dataloader(self, drop_last=False):
        return self.get_loader(self.valset, drop_last)

    def test_dataloader(self, drop_last=False):
        return self.get_loader(self.testset, drop_last)


class Caltech101DataModule(TorchVisDataModule):
    def __init__(self, cfg, train_transforms, target_transforms=None):
        self.num_classes = 101
        self.dims = (3, 224, 224)
        self.task = "multiclass"

        super().__init__(cfg, train_transforms, target_transforms)

    def get_dataset(
        self, dataset_path: str, train_transforms=None, target_transforms=None
    ):
        return Caltech101Dataset(
            dataset_path=dataset_path,
            transform=train_transforms,
            target_transform=target_transforms,
        )
