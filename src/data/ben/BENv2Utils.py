from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import lmdb
import numpy as np
import pandas as pd
from safetensors.numpy import load as safetensor_load

_s1_bandnames = ["VH", "VV"]
_s2_bandnames = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B11",
    "B12",
    "B8A",
]
_all_bandnames = _s2_bandnames + _s1_bandnames


def _numpy_level_aggreagate(df, key_col, val_col):
    """
    Optimized version of df.groupby(key_col)[val_col].apply(list).reset_index(name=val_col)
    Credits to B. M. @
    https://stackoverflow.com/questions/22219004/how-to-group-dataframe-rows-into-list-in-pandas-groupby
    """
    keys, values = df.sort_values(key_col).values.T
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index[1:])
    df2 = pd.DataFrame({key_col: ukeys, val_col: [list(a) for a in arrays]})
    return df2


NEW_LABELS_ORIGINAL_ORDER = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)
NEW_LABELS = sorted(NEW_LABELS_ORIGINAL_ORDER)
_val2idx_new_labels = {x: NEW_LABELS.index(x) for x in NEW_LABELS}
_val2idx_new_labels_original_order = {
    x: NEW_LABELS_ORIGINAL_ORDER.index(x) for x in NEW_LABELS_ORIGINAL_ORDER
}

BEN_IDX2NAME = {i: name for i, name in enumerate(NEW_LABELS)}


def ben_19_labels_to_multi_hot(
    labels: Iterable[str], lex_sorted: bool = True
) -> np.array:
    """
    Convenience function that converts an input iterable of labels into
    a multi-hot encoded vector.
    If `lex_sorted` is True (default) the classes are lexigraphically ordered, as they are
    in `constants.NEW_LABELS`.
    If `lex_sorted` is False, the original order from the BigEarthNet paper is used, as
    they are given in `constants.NEW_LABELS_ORIGINAL_ORDER`.

    If an unknown label is given, a `KeyError` is raised.

    Be aware that this approach assumes that **all** labels are actually used in the dataset!
    This is not necessarily the case if you are using a subset!

    :param labels: iterable of labels
    :param lex_sorted: whether to lexigraphically sort the labels, defaults to True
    :return: multi-hot encoded vector as np.array of shape (len(NEW_LABELS),) where the i-th entry is 1 if the i-th
        label is present in the input iterable and 0 otherwise
    """
    lbls_to_idx = (
        _val2idx_new_labels if lex_sorted else _val2idx_new_labels_original_order
    )
    idxs = [lbls_to_idx[label] for label in labels]
    multi_hot = np.zeros(len(NEW_LABELS))
    multi_hot[idxs] = 1.0
    return multi_hot


class BENv2LDMBReader:
    def __init__(
        self,
        image_lmdb_file: Union[str, Path],
        label_file: Union[str, Path],
        s2s1_mapping_file: Optional[Union[str, Path]] = None,
        bands: Optional[Iterable[str]] = None,
        process_bands_fn: Optional[
            Callable[[Dict[str, np.ndarray], List[str]], Any]
        ] = None,
        process_labels_fn: Optional[Callable[[List[str]], Any]] = None,
        print_info: bool = False,
    ):
        """
        :param image_lmdb_file: path to the lmdb file containing the images as safetensors in numpy format
        :param label_file: path to the parquet file containing the labels
        :param s2s1_mapping_file: path to the parquet file containing the mapping from S2v2 to S1. Only needed if S1
            bands are used.
        :param bands: list of bands to use, defaults to all bands in order [B01, B02, ..., B12, B8A, VH, VV]
        :param process_bands_fn: function to process the bands, defaults to stack_and_interpolate with nearest
            interpolation. Must accept a dict of the form {bandname: np.ndarray} and a list of bandnames and may return
            anything.
        :param process_labels_fn: function to process the labels, defaults to ben_19_labels_to_multi_hot. Must accept
            a list of strings and may return anything.
        :param print_info: whether to print some info during setup
        """
        self.image_lmdb_file = image_lmdb_file
        self.env = None
        self.print_info_toggle = print_info

        self.bands = bands if bands is not None else _all_bandnames
        self.uses_s1 = any([x in _s1_bandnames for x in self.bands])
        self.uses_s2 = any([x in _s2_bandnames for x in self.bands])

        if s2s1_mapping_file is None:
            assert (
                not self.uses_s1
            ), "If you want to use S1 bands, please provide a s2s1_mapping_file"
            self.mapping = None
        else:
            # read and create mapping S2v2 name -> S1 name
            self._print_info("Reading mapping ...")
            mapping = pd.read_parquet(str(s2s1_mapping_file))
            self._print_info("Creating mapping dict ...")
            self.mapping = dict(
                zip(mapping.new_name, mapping.s1_name)
            )  # naming of the columns is hardcoded
            del mapping

        # read labels and create mapping S2v2 name -> List[label]
        self._print_info("Reading labels ...")
        lbls = pd.read_parquet(str(label_file))
        self._print_info("Aggregating label list ...")
        lbls = _numpy_level_aggreagate(lbls, "patch", "lbl_19")
        # lbls = lbls.groupby('patch')['lbl_19'].apply(list).reset_index(name='lbl_19')
        self._print_info("Creating label dict ...")
        self.lbls = dict(
            zip(lbls.patch, lbls.lbl_19)
        )  # naming of the columns is hardcoded
        self.lbl_key_set = set(self.lbls.keys())
        del lbls

        # set mean and std based on bands selected
        self.mean = None
        self.std = None

        self.process_bands_fn = (
            process_bands_fn if process_bands_fn is not None else lambda x, y: x
        )
        self.process_labels_fn = (
            process_labels_fn if process_labels_fn is not None else lambda x: x
        )

        self._keys = None
        self._S2_keys = None
        self._S1_keys = None

    def _print_info(self, info: str):
        if self.print_info_toggle:
            print(info)

    def _open_env(self):
        """
        Opens the lmdb environment if not opened yet
        """
        if self.env is None:
            self._print_info("Opening LMDB environment ...")
            self.env = lmdb.open(
                str(self.image_lmdb_file),
                readonly=True,
                lock=False,
                meminit=False,
                # readahead=True,
                map_size=8 * 1024**3,  # 8GB blocked for caching
            )

    def keys(self, update: bool = False):
        """
        Returns all keys in the lmdb file, i.e. all S2v2 patch names as well as all S1 patch names.

        :param update: whether to update the cached keys
        """
        self._open_env()
        if self._keys is None or update:
            self._print_info("(Re-)Reading keys ...")
            with self.env.begin() as txn:
                self._keys = set(txn.cursor().iternext(values=False))
            self._keys = set([x.decode() for x in self._keys])
        return self._keys

    def S2_keys(self, update: bool = False):
        """
        Returns all keys in the lmdb file that are S2v2 patch names (i.e. all keys that start with 'S2').

        :param update: whether to update the cached keys
        """
        if self._S2_keys is None or update:
            self._print_info("(Re-)Reading S2 keys ..")
            self._S2_keys = set(
                [key for key in self.keys(update) if key.startswith("S2")]
            )
        return self._S2_keys

    def S1_keys(self, update: bool = False):
        """
        Returns all keys in the lmdb file that are S1 patch names (i.e. all keys that start with 'S1').

        :param update: whether to update the cached keys
        """
        if self._S1_keys is None or update:
            self._print_info("(Re-)Reading S1 keys")
            self._S1_keys = set(
                [key for key in self.keys(update) if key.startswith("S1")]
            )
        return self._S1_keys

    def __getitem__(self, key: str):
        """
        Returns the image data and labels for a given key.

        :param key: the key to use as S2v2 patch name

        :return: a tuple of (img_data, labels) where img_data is a dict of the form {bandname: np.ndarray} and labels as
            a list of strings. The return types change if process_bands_fn or process_labels_fn are provided during
            init and return the return values of those functions in these cases instead in the same order.

        Note: If S1 bands are used, the key is mapped to the corresponding S1 patch name and the image data for the S1
            patch is returned as well.
        Note: The key is always the S2v2 patch name, even if only S1 bands are used.
        """
        # the key is the name of the S2v2 patch

        # open lmdb file if not opened yet
        self._open_env()
        img_data_dict = {}
        if self.uses_s2:
            # read image data for S2v2
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        if self.uses_s1:
            # read image data for S1
            s1_key = self.mapping[key]
            with self.env.begin(write=False, buffers=True) as txn:
                byte_data = txn.get(s1_key.encode())
            img_data_dict.update(safetensor_load(bytes(byte_data)))

        img_data_dict = {k: v for k, v in img_data_dict.items() if k in self.bands}

        img_data = self.process_bands_fn(img_data_dict, self.bands)
        labels = self.lbls[key] if key in self.lbl_key_set else []
        labels = self.process_labels_fn(labels)

        return img_data, labels
