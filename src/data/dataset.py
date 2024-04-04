import os
import pickle

import lmdb
import numpy as np
import pandas as pd
from skimage.transform import resize
from torch.utils.data import Dataset

from src.data.constants import (
    BEN19_NAME2IDX,
    DEEPGLOBE_NAME2IDX,
    EUROSAT_NAME2IDX,
)


def read_csv(csv_path):
    # if file exists, read it
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path, header=None).to_numpy()[:, 0]
    raise FileNotFoundError(f"CSV file not found at {csv_path}")


def read_temporal_views(temporal_views_path):
    temporal_views = None
    if temporal_views_path is not None:
        temporal_views = pd.read_parquet(temporal_views_path)
    return temporal_views


class BaseDataset(Dataset):
    def __init__(
        self,
        images_lmdb_path,
        csv_path,
        labels_path,
        temporal_views_path=None,
        transform=None,
        segmentations_lmdb_path=None,
        xai_segmentations_lmdb_path=None,
    ):
        """
        Parameter
        ---------
        images_lmdb_path      : path to the LMDB file for efficiently loading the patches.
        csv_path       : path to a csv file containing the patch names that will make up this split
        transform_mode:  specifies the image transform mode which determines the augmentations
                         to be applied to the image
        """
        self.lmdb_path = images_lmdb_path
        self.images_env = None

        self.segmentations_lmdb_path = segmentations_lmdb_path
        self.segmentations_env = None

        self.xai_segmentations_lmdb_path = xai_segmentations_lmdb_path
        self.xai_segmentations_env = None

        self.patch_names = read_csv(csv_path)
        self.labels = self.read_labels(labels_path, self.patch_names)
        self.transform = transform
        self.temporal_views = read_temporal_views(temporal_views_path)

    def read_labels(self, meta_data_path, patch_names):
        df = pd.read_parquet(meta_data_path)
        df_subset = (
            df.set_index("name").loc[self.patch_names].reset_index(inplace=False)
        )
        string_labels = df_subset.labels.tolist()
        multihot_labels = np.array(list(map(self.convert_to_multihot, string_labels)))
        return multihot_labels

    def convert_to_multihot(self, labels):
        raise NotImplementedError

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        patch, self.images_env = self._extract_patch_from_lmdb(
            idx, self.images_env, self.lmdb_path
        )

        segmentation_patch = []
        try:
            if self.segmentations_lmdb_path is not None:
                (
                    segmentation_patch,
                    self.segmentations_env,
                ) = self._extract_patch_from_lmdb(
                    idx, self.segmentations_env, self.segmentations_lmdb_path
                )
        except ValueError:
            pass

        xai_segmentation_patch = []
        try:
            if self.xai_segmentations_lmdb_path is not None:
                (
                    xai_segmentation_patch,
                    self.xai_segmentations_env,
                ) = self._extract_patch_from_lmdb(
                    idx, self.xai_segmentations_env, self.xai_segmentations_lmdb_path
                )
        except ValueError:
            # We don't have XAI segmentations for all patches
            pass

        label = self.labels[idx]
        # divide by 255 to get values between 0 and 1
        patch = patch / 255
        patch = self.transform(patch) if self.transform is not None else patch
        return {
            "features": patch,
            "targets": label,
            "index": idx,
            "segmentations": segmentation_patch,
            "xai_segmentations": xai_segmentation_patch,
        }

    def get_patch_name(self, idx):
        return self.patch_names[idx]

    def _extract_patch_from_lmdb(self, idx, env, lmdb_path):
        """Extract patch from LMDB."""
        if env is None:
            # write_lmdb_keys_to_file(lmdb_path, "lmdb_keys.txt")
            env = lmdb.open(
                str(lmdb_path),
                max_dbs=1,
                readonly=True,
                lock=False,
                meminit=False,
                readahead=True,
            )

        patch_name = self.get_patch_name(idx)
        with env.begin(write=False) as txn:
            byte_flow = txn.get(patch_name.encode("utf-8"))
        if byte_flow is None:
            raise ValueError(f"Patch {patch_name} not found in LMDB.")

        patch = pickle.loads(byte_flow)

        return patch, env

    def __len__(self):
        """Get length of Dataset."""
        return len(self.patch_names)


def print_first_5_indices(lmdb_path):
    """Print the first 5 indices from an LMDB file."""
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        meminit=False,
        readahead=True,
    )

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for idx, (key, value) in enumerate(cursor):
            print(f"Index {idx}: {key.decode('utf-8')}")
            if idx == 4:  # Stop after printing 5 indices
                break

    env.close()


def write_lmdb_keys_to_file(lmdb_path, output_file_path):
    """Write all keys from an LMDB database to a file."""
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        meminit=False,
        readahead=True,
    )
    with env.begin(write=False) as txn, open(output_file_path, "w") as f:
        cursor = txn.cursor()
        for key, _ in cursor:
            f.write(f"{key.decode('utf-8')}\n")
    env.close()
    print(f"Keys written to {output_file_path}")


def interpolate_bands(bands, img10_shape=(120, 120)):
    """Interpolate bands. See: https://github.com/lanha/DSen2/blob/master/utils/patches.py."""
    bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)
    for i in range(bands.shape[0]):
        bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode="reflect") * 30000
    return bands_interp


class Ben19Dataset(BaseDataset):
    def __init__(
        self,
        images_lmdb_path,
        csv_path,
        labels_path,
        temporal_views_path=None,
        transform=None,
        active_classes=None,
        rgb_only=False,
        discard_empty_labels=True,
        segmentations_lmdb_path=None,
        xai_segmentations_lmdb_path=None,
    ):
        super().__init__(
            images_lmdb_path,
            csv_path,
            labels_path,
            temporal_views_path,
            transform,
            segmentations_lmdb_path,
        )
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
        self.band_ordering = [
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
        ]

        if discard_empty_labels:
            self.discard_empty_labels()

    def read_labels(self, meta_data_path, patch_names):
        df = pd.read_parquet(meta_data_path)
        df_subset = (
            df.set_index("name").loc[self.patch_names].reset_index(inplace=False)
        )
        string_labels = df_subset.new_labels.tolist()
        multihot_labels = np.array(list(map(self.convert_to_multihot, string_labels)))
        return multihot_labels

    def convert_to_multihot(self, labels):
        multihot = np.zeros(19)
        indices = [BEN19_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot

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
        s2_patch, self.images_env = self._extract_patch_from_lmdb(
            idx, self.images_env, self.lmdb_path
        )
        label = self.labels[idx]

        bands10 = s2_patch.get_stacked_10m_bands()
        bands10 = bands10.astype(np.float32)
        bands20 = s2_patch.get_stacked_20m_bands()
        bands20 = interpolate_bands(bands20)
        bands20 = bands20.astype(np.float32)

        # put channel to last axis s.t. toTensor can flip them to first axis again
        patch = np.moveaxis(np.concatenate([bands10, bands20]), 0, -1)
        patch = self.transform(patch)
        if self.rgb_only:
            patch = patch[[2, 1, 0], ...]

        return patch, label, idx, None


class DeepGlobeDataset(BaseDataset):
    def __init__(
        self,
        images_lmdb_path,
        csv_path,
        labels_path,
        temporal_views_path=None,
        transform=None,
        segmentations_lmdb_path=None,
        xai_segmentations_lmdb_path=None,
    ):
        super().__init__(
            images_lmdb_path,
            csv_path,
            labels_path,
            temporal_views_path,
            transform,
            segmentations_lmdb_path,
            xai_segmentations_lmdb_path,
        )

    def convert_to_multihot(self, labels):
        multihot = np.zeros(6)
        indices = [DEEPGLOBE_NAME2IDX[label] for label in labels]
        multihot[indices] = 1
        return multihot


class EuroSATDataset(BaseDataset):
    def __init__(
        self,
        images_lmdb_path,
        csv_path,
        labels_path,
        temporal_views_path=None,
        transform=None,
        segmentations_lmdb_path=None,
        xai_segmentations_lmdb_path=None,
    ):
        super().__init__(
            images_lmdb_path,
            csv_path,
            labels_path,
            temporal_views_path,
            transform,
            segmentations_lmdb_path,
        )
        self.band_ordering = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B010",
            "B11",
            "B12",
        ]
        self.ben19_ordering = [1, 2, 3, 7, 4, 5, 6, 8, 11, 12]

    def convert_to_multihot(self, labels):
        return EUROSAT_NAME2IDX[labels[0]]  # hack for single-label classification

    def __getitem__(self, idx):
        """Get item at position idx of Dataset."""
        patch, self.images_env = self._extract_patch_from_lmdb(
            idx, self.images_env, self.lmdb_path
        )

        label = self.labels[idx]
        patch = patch[:, :, self.ben19_ordering]
        patch = self.transform(patch) if self.transform is not None else patch

        return patch, label, idx, None
