from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union


import numpy as np
from torch.utils.data import Dataset

from BENv2Utils import BENv2LDMBReader


class BENv2DataSet(Dataset):
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
        transforms: Optional[Callable] = None,
        keys: Optional[List[str]] = None,
        return_patchname: bool = False,
        verbose: bool = False,
    ):
        """
        :param image_lmdb_file: path to the lmdb file containing the images as safetensors in numpy format
        :param label_file: path to the parquet file containing the labels
        :param s2s1_mapping_file: path to the parquet file containing the mapping from S2v2 to S1
        :param bands: list of bands to use, defaults to all bands in order [B01, B02, ..., B12, B8A, VH, VV]
        :param process_bands_fn: function to process the bands, defaults to stack_and_interpolate with nearest
            interpolation. Must accept a dict of the form {bandname: np.ndarray} and a list of bandnames and may return
            anything.
        :param process_labels_fn: function to process the labels, defaults to ben_19_labels_to_multi_hot. Must accept
            a list of strings and may return anything.
        :param transforms: transforms to apply to the images after processing, defaults to None
        :param keys: keys to use from the lmdb, defaults to all keys in the lmdb (S2 keys)
        :param return_patchname: whether to return the patchname as well, defaults to False. If set to True, the return
            value is (img, lbl, patchname) instead of (img, lbl)
        :param verbose: whether to print some info about the dataset, defaults to False
        """
        self.transforms = transforms
        self.return_patchname = return_patchname
        self.reader = BENv2LDMBReader(
            image_lmdb_file=image_lmdb_file,
            label_file=label_file,
            s2s1_mapping_file=s2s1_mapping_file,
            bands=bands,
            process_bands_fn=process_bands_fn,
            process_labels_fn=process_labels_fn,
            print_info=verbose,
        )
        if keys is None:
            # read from a temp reader to not have problems with partially copied envs etc.
            tmp_reader = BENv2LDMBReader(
                image_lmdb_file=image_lmdb_file,
                label_file=label_file,
                s2s1_mapping_file=s2s1_mapping_file,
                bands=bands,
                process_bands_fn=process_bands_fn,
                process_labels_fn=process_labels_fn,
                print_info=verbose,
            )
            self.keys = tmp_reader.S2_keys()
        else:
            self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img, lbl = self.reader[key]
        if self.transforms is not None:
            img = self.transforms(img)
        if self.return_patchname:
            return img, lbl, key
        return img, lbl
