import copy
import os
from abc import abstractmethod

import pytorch_lightning as pl
from torch.utils.data import DataLoader

# from data.utils import add_mixed_noise
from src.data.tom_data.constants import ACTIVE_CLASSES
from src.data.tom_data.constants import BAND_99TH_PERCENTILES, EUROSAT_99TH_PERCENTILES
from src.data.tom_data.constants import BAND_NORM_STATS, EUROSAT_NORM_STATS
from src.data.tom_data.dataset import (
    Ben19Dataset,
    DeepGlobeDataset,
    EuroSATDataset,
)


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, transform_tr, transform_te):
        """DataModule for BigEarthNet, DeepGlobe, EuroSAT.

        Parameters
        ----------
        cfg : dict
        transform_tr : Transform
        transform_te : Transform
        """
        super().__init__()
        self.cfg = cfg
        self.transform_tr = transform_tr
        self.transform_te = transform_te

    @abstractmethod
    def get_dataset(
        self,
        lmdb_path,
        csv_path,
        labels_path,
        transform,
        temporal_views_path=None,
        segmentations_lmdb_path=None,
    ):
        raise NotImplementedError

    def setup(self, stage=None):
        # if self.cfg['data'].eval_on_test:
        #     self.cfg['data'].val_csv = self.cfg['data'].test_csv
        #     self.transform_val = self.transform_te

        self.trainset_tr = self.get_dataset(
            lmdb_path=self.cfg["data"]["images_lmdb_path"],
            csv_path=self.cfg["data"]["train_csv"],
            labels_path=self.cfg["data"]["labels_path"],
            temporal_views_path=self.cfg["data"].get("temporal_views_path", None),
            transform=self.transform_tr,
            segmentations_lmdb_path=self.cfg["data"].get(
                "segmentations_lmdb_path", None
            ),
        )
        self.valset = self.get_dataset(
            lmdb_path=self.cfg["data"]["images_lmdb_path"],
            csv_path=self.cfg["data"]["train_csv"],
            labels_path=self.cfg["data"]["labels_path"],
            temporal_views_path=self.cfg["data"].get("temporal_views_path", None),
            transform=self.transform_te,
            segmentations_lmdb_path=self.cfg["data"].get(
                "segmentations_lmdb_path", None
            ),
        )
        self.testset = self.get_dataset(
            lmdb_path=self.cfg["data"]["images_lmdb_path"],
            csv_path=self.cfg["data"]["train_csv"],
            labels_path=self.cfg["data"]["labels_path"],
            temporal_views_path=self.cfg["data"].get("temporal_views_path", None),
            transform=self.transform_te,
            segmentations_lmdb_path=self.cfg["data"].get(
                "segmentations_lmdb_path", None
            ),
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
        return self.get_loader(
            self.valset, drop_last
        )  # list(map(lambda x: self.get_loader(x, drop_last), self.valset))

    def test_dataloader(self, drop_last=False):
        return self.get_loader(
            self.testset, drop_last
        )  # list(map(lambda x: self.get_loader(x, drop_last), self.testset))


class BigEarthNetDataModule(DataModule):
    def __init__(self, cfg, transform_tr, transform_te):
        super().__init__(cfg, transform_tr, transform_te)
        self.cfg = cfg
        self.init_transforms()
        self.init_active_classes()

    def get_dataset(
        self,
        lmdb_path,
        csv_path,
        labels_path,
        transform,
        temporal_views_path=None,
        segmentations_lmdb_path=None,
    ):
        return Ben19Dataset(
            lmdb_path,
            csv_path,
            labels_path,
            temporal_views_path,
            transform,
            self.active_classes,
        )

    def init_transforms(self):
        self.train_country = (
            os.path.basename(self.cfg["data"].train_csv).split("_")[0].capitalize()
        )
        self.test_country = list(
            map(
                lambda x: os.path.basename(x).split("_")[0].capitalize(),
                self.cfg["data"].test_csv,
            )
        )
        if self.cfg["data"].global_pctl:
            print("Global Pre-Normalization.")
            tr_percentiles = 10000
        elif self.cfg["data"].all_percentiles and not self.cfg["data"].global_pctl:
            print("All BEN Pre-Normalization.")
            tr_percentiles = list(BAND_99TH_PERCENTILES["All"].values())
        else:
            print("Country Pre-Normalization.")
            tr_percentiles = list(BAND_99TH_PERCENTILES[self.train_country].values())
        # te_percentiles = list(map(lambda x: list(BAND_99TH_PERCENTILES[x].values()), self.train_country))

        # currently train and test normalized by train norms (!)
        channel_global = "Global" if self.cfg["data"].global_pctl else "Channel"
        if self.cfg["data"].all_percentiles:
            print("All Percentiles", channel_global)
            means = list(BAND_NORM_STATS[channel_global]["All"]["mean"].values())
            stds = list(BAND_NORM_STATS[channel_global]["All"]["std"].values())
        else:
            print("Country Percentiles", channel_global)
            means = list(
                BAND_NORM_STATS[channel_global][self.train_country]["mean"].values()
            )
            stds = list(
                BAND_NORM_STATS[channel_global][self.train_country]["std"].values()
            )

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(
            means, stds, tr_percentiles, sentinel2=True
        )
        self.transform_tr.setup_compose()

        transform_va_list = []
        for te_percentile in range(len(self.test_country)):
            transform = copy.deepcopy(self.transform_te)
            transform.add_data_transforms(means, stds, tr_percentiles, sentinel2=True)
            transform.setup_compose()
            transform_va_list.append(transform)
        self.transform_val = transform_va_list

        # add data transforms to (all) transform_te's
        transform_te_list = []
        for te_percentile in range(len(self.test_country)):
            transform = copy.deepcopy(self.transform_te)
            transform.add_data_transforms(means, stds, tr_percentiles, sentinel2=True)
            transform.setup_compose()
            transform_te_list.append(transform)
        self.transform_te = transform_te_list

    def init_active_classes(self):
        active_classes_tr = set(ACTIVE_CLASSES[self.train_country])
        active_classes_te = list(
            map(lambda x: set(ACTIVE_CLASSES[x]), self.test_country)
        )
        active_classes_te = set.intersection(*active_classes_te)
        self.active_classes = list(active_classes_tr.intersection(active_classes_te))

        if self.cfg["data"].intersection_8country:
            self.active_classes = ACTIVE_CLASSES["Intersection_8Country"]
        self.num_cls = len(self.active_classes)


class DeepGlobeDataModule(DataModule):
    def __init__(self, cfg, transform_tr, transform_te):
        super().__init__(cfg, transform_tr, transform_te)
        self.num_classes = 6
        self.dims = (3, 256, 256)
        self.cfg = cfg
        self.num_cls = 6
        self.init_transforms()

    def get_dataset(
        self,
        lmdb_path,
        csv_path,
        labels_path,
        transform,
        temporal_views_path=None,
        segmentations_lmdb_path=None,
    ):
        return DeepGlobeDataset(
            images_lmdb_path=lmdb_path,
            csv_path=csv_path,
            labels_path=labels_path,
            transform=transform,
            temporal_views_path=temporal_views_path,
            segmentations_lmdb_path=segmentations_lmdb_path,
        )

    def init_transforms(self):
        """Currently hard-coded always only one test-dataloader."""
        means = [0.4095, 0.3808, 0.2836]
        stds = [0.1509, 0.1187, 0.1081]

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds)
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        self.transform_te.add_data_transforms(means, stds)
        self.transform_te.setup_compose()


class EuroSATDataModule(DataModule):
    def __init__(self, cfg, transform_tr, transform_te):
        super().__init__(cfg, transform_tr, transform_te)
        self.cfg = cfg
        self.num_cls = 10
        self.init_transforms()

    def get_dataset(
        self,
        lmdb_path,
        csv_path,
        labels_path,
        transform,
        temporal_views_path=None,
        segmentations_lmdb_path=None,
    ):
        return EuroSATDataset(
            lmdb_path,
            csv_path,
            labels_path,
            temporal_views_path,
            transform,
            segmentations_lmdb_path,
        )

    def init_transforms(self):
        if self.cfg["data"].pretrain_norm:
            percentiles = list(BAND_99TH_PERCENTILES["All"].values())
            # currently train and test normalized by train norms (!)
            means = list(BAND_NORM_STATS["Channel"]["All"]["mean"].values())
            stds = list(BAND_NORM_STATS["Channel"]["All"]["std"].values())
        else:
            percentiles = list(EUROSAT_99TH_PERCENTILES.values())
            # currently train and test normalized by train norms (!)
            means = list(EUROSAT_NORM_STATS["mean"].values())
            stds = list(EUROSAT_NORM_STATS["std"].values())

        # add data transforms to transform_tr
        self.transform_tr.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_tr.setup_compose()

        # add data transforms to (all) transform_te's
        self.transform_te.add_data_transforms(means, stds, percentiles, sentinel2=True)
        self.transform_te.setup_compose()
        self.transform_te = [self.transform_te]


def get_datamodule(dataset):
    if dataset == "BigEarthNet":
        return BigEarthNetDataModule
    if dataset == "DeepGlobe":
        return DeepGlobeDataModule
    if dataset == "EuroSAT":
        return EuroSATDataModule
