import pickle

import lmdb
import numpy as np
import pandas as pd
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch
from skimage.transform import resize
from torch.utils.data import Dataset

from constants import (
    BEN19_NAME2IDX,
    DEEPGLOBE_NAME2IDX,
    EUROSAT_NAME2IDX,
)


class BaseDataset(Dataset):
    def __init__(self, lmdb_path, csv_path, labels_path, temporal_views_path=None, transform=None):
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.env = None
        self.lmdb_path = lmdb_path
        self.patch_names = self.read_csv(csv_path)
        self.labels = self.read_labels(labels_path, self.patch_names)
        self.transform = transform
        self.temporal_views = self.read_temporal_views(temporal_views_path)

    def read_csv(self, csv_data):
        return pd.read_csv(csv_data, header=None).to_numpy()[:, 0]

    def read_labels(self, meta_data_path, patch_names):
        df = pd.read_parquet(meta_data_path)
        df_subset = df.set_index('name').loc[self.patch_names].reset_index(inplace=False)
        string_labels = df_subset.labels.tolist()
        multihot_labels = np.array(list(map(self.convert_to_multihot, string_labels)))
        return multihot_labels

    def read_temporal_views(self, temporal_views_path):
        temporal_views = None
        if temporal_views_path is not None:
            temporal_views = pd.read_parquet(temporal_views_path)
        return temporal_views

    def convert_to_multihot(self, labels):
        raise NotImplementedError

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        patch = pickle.loads(byteflow)
        label = self.labels[idx]
        patch = self.transform(patch) if self.transform is not None else patch

        return patch, label, idx

    def __len__(self):
        """Get length of Dataset."""
        return len(self.patch_names)


class Ben19Dataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, temporal_views_path=None, transform=None, active_classes=None,
                 rgb_only=False, discard_empty_labels=True):
        super().__init__(lmdb_path, csv_path, labels_path, temporal_views_path, transform)
        """
        Parameter
        ---------
        lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.active_classes = active_classes
        self.labels = self.select_active_classes(self.labels)
        self.rgb_only = rgb_only
        self.band_ordering = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

        if discard_empty_labels:
            self.discard_empty_labels()

    def read_labels(self, meta_data_path, patch_names):
        df = pd.read_parquet(meta_data_path)
        df_subset = df.set_index('name').loc[self.patch_names].reset_index(inplace=False)
        string_labels = df_subset.new_labels.tolist()
        multihot_labels = np.array(list(map(self.convert_to_multihot, string_labels)))
        return multihot_labels

    def convert_to_multihot(self, labels):
        multihot = np.zeros(19)
        indices = [BEN19_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot

    def interpolate_bands(self, bands, img10_shape=[120, 120]):
        """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
        bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
        for i in range(bands.shape[0]):
            bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode='reflect') * 30000
        return bands_interp

    def select_active_classes(self, multihot):
        if self.active_classes is not None:
            multihot = multihot[:, self.active_classes]
        return multihot

    def discard_empty_labels(self):
        empty_idx = np.argwhere(self.labels.sum(axis=1) == 0).flatten()
        print("Removing {} patches without any active label.".format(len(empty_idx)))
        self.patch_names = np.delete(self.patch_names, empty_idx)
        self.labels = np.delete(self.labels, empty_idx, axis=0)

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        s2_patch = BigEarthNet_S2_Patch.loads(byteflow)
        label = self.labels[idx]

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = self.interpolate_bands(bands20)
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
        patch = self.transform(patch)
        if self.rgb_only:
            patch = patch[[2, 1, 0], ...]

        return patch, label, idx


class DeepGlobeDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, temporal_views_path=None, transform=None):
        super().__init__(lmdb_path, csv_path, labels_path, temporal_views_path, transform)

    def convert_to_multihot(self, labels):
        multihot = np.zeros(6)
        indices = [DEEPGLOBE_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot


class EuroSATDataset(BaseDataset):
    def __init__(self, lmdb_path, csv_path, labels_path, temporal_views_path=None, transform=None):
        super().__init__(lmdb_path, csv_path, labels_path, temporal_views_path, transform)
        self.band_ordering = [
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B010', 'B11', 'B12'
        ]
        self.ben19_ordering = [1, 2, 3, 7, 4, 5, 6, 8, 11, 12]

    def convert_to_multihot(self, labels):
        return EUROSAT_NAME2IDX[labels[0]]  # hack for single-label classification

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.patch_names[idx]

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode('utf-8'))

        patch = pickle.loads(byteflow)

        label = self.labels[idx]
        patch = patch[:, :, self.ben19_ordering]
        patch = self.transform(patch) if self.transform is not None else patch

        return patch, label, idx
