from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
# from lightning.pytorch import LightningDataModule

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.ben.BENv2DataSet import BENv2DataSet
from data.ben.BENv2Stats import means, stds
from data.ben.BENv2TorchUtils import ben_19_labels_to_multi_hot
from data.ben.BENv2TorchUtils import stack_and_interpolate
from data.ben.BENv2Utils import _all_bandnames


class BENv2DataModule(pl.LightningDataModule):
    # This setting is dependent on the system being used
    pin_memory = False

    def __init__(
        self,
        image_lmdb_file: Union[str, Path],
        label_file: Union[str, Path],
        s2s1_mapping_file: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 8,
        bands: Optional[Iterable[str]] = _all_bandnames,
        process_bands_fn: Optional[
            Callable[[Dict[str, np.ndarray], List[str]], Any]
        ] = partial(stack_and_interpolate, img_size=120, upsample_mode="nearest"),
        process_labels_fn: Optional[
            Callable[[List[str]], Any]
        ] = ben_19_labels_to_multi_hot,
        train_transforms: Optional[Callable] = None,
        eval_transforms: Optional[Callable] = None,
        split_file: Optional[Union[str, Path]] = None,
        keys: Optional[Dict[str, List[str]]] = None,
        img_size: Optional[
            int
        ] = 120,  # only used for mean/std retrieval in case no transform is provided
        interpolation_mode: Optional[
            str
        ] = "nearest",  # only used for mean/std retrieval in case no transform is provided
        verbose: bool = False,
        xai_segmentations_lmdb_path=None,
        **kwargs,
    ):
        """
        :param image_lmdb_file: path to the lmdb file containing the images as safetensors in numpy format
        :param label_file: path to the parquet file containing the labels
        :param s2s1_mapping_file: path to the parquet file containing the mapping from S2v2 to S1
        :param batch_size: batch size for the dataloaders
        :param num_workers: number of workers for the dataloaders
        :param bands: list of bands to use, defaults to all bands in order [B01, B02, ..., B12, B8A, VH, VV]
        :param process_bands_fn: function to process the bands, defaults to stack_and_interpolate with nearest
            interpolation. Must accept a dict of the form {bandname: np.ndarray} and a list of bandnames and may return
            anything.
        :param process_labels_fn: function to process the labels, defaults to ben_19_labels_to_multi_hot. Must accept
            a list of strings and may return anything.
        :param train_transforms: transforms to use for training, defaults to random horizontal and vertical flips and
        normalization
        :param eval_transforms: transforms to use for evaluation, defaults to normalization
        :param split_file: path to the csv file containing the split information. Only used if keys is None
        :param keys: dictionary containing the keys for the different splits. Should be of the form
            {"train": [key1, key2, ...], "validation": [key1, key2, ...], "test": [key1, key2, ...]}
        :param img_size: image size to use for mean and std retrieval. Only used if no transforms are provided. If set
            to None, the values for no interpolation are used.
        :param interpolation_mode: interpolation mode to use for mean and std retrieval. Only used if no transforms are
            provided. If set to None, the values for no interpolation are used.
        :param verbose: print info during setup
        :param kwargs: additional kwargs for the BENv2DataSet
        """
        super().__init__()
        self.image_lmdb_file = image_lmdb_file
        self.label_file = label_file
        self.s2s1_mapping_file = s2s1_mapping_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bands = bands
        self.process_bands_fn = process_bands_fn
        self.process_labels_fn = process_labels_fn
        self.verbose = verbose
        self.xai_segmentations_lmdb_path = xai_segmentations_lmdb_path
        self.kwargs = kwargs

        self.task = "multilabel"
        self.num_classes = 19
        self.dims = [len(bands), img_size, img_size]

        # there are different combinations depending on the mode available, select the right mode
        interpolation_method = "no_interpolation"
        if img_size is not None and interpolation_mode is not None:
            interpolation_method = f"{img_size}_{interpolation_mode}"
        mean = means[interpolation_method]
        std = stds[interpolation_method]
        # only select the right bands
        mean = [mean[b] for b in bands]
        std = [std[b] for b in bands]

        if train_transforms is None:
            from torchvision import transforms

            self._print_info("Using default train transformation.")

            # use default transforms
            self.train_transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # torchvision.transforms.RandomRotation(180),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            self.train_transforms = train_transforms

        if eval_transforms is None:
            from torchvision import transforms

            self._print_info("Using default eval transformation.")

            # use default transforms
            self.eval_transforms = transforms.Compose(
                [
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            self.eval_transforms = eval_transforms

        if keys is not None:
            for s in ["train", "validation", "test"]:
                assert (
                    s in keys
                ), f"The keys dict needs to have a {s} key containing the list of patches for split {s}"
            self.keys = keys
        elif split_file is not None:
            # key retrieval from label and split file
            import pandas as pd

            self._print_info("Using default split from split csv file and labels.")

            # read the csv
            df = pd.read_csv(split_file)
            # get all patches without dublicates
            # -> this means not all labels are in this table but we dont use it for label information at this point anyways
            lbls = pd.read_parquet(label_file).drop_duplicates(["patch"])
            # get only patches that also have a label, drop the additional columns
            df = df.merge(lbls, how="inner", left_on=["name"], right_on=["patch"]).drop(
                ["lbl_19", "patch"], axis=1
            )
            self.keys = {
                split: sorted(list(df[df.split == split].name.values))
                for split in ["train", "validation", "test"]
            }
        else:
            raise RuntimeError(
                "Please provide a dictionary for splits or the split file so that the split can be retrieved automatically"
            )


    def _print_info(self, info: str):
        if self.verbose:
            print(info)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = BENv2DataSet(
                image_lmdb_file=self.image_lmdb_file,
                label_file=self.label_file,
                s2s1_mapping_file=self.s2s1_mapping_file,
                bands=self.bands,
                process_bands_fn=self.process_bands_fn,
                process_labels_fn=self.process_labels_fn,
                transforms=self.train_transforms,
                keys=self.keys["train"],
                return_patchname=False,
                verbose=self.verbose,
                xai_segmentations_lmdb_path=self.xai_segmentations_lmdb_path,
                **self.kwargs,
            )
            self.val_dataset = BENv2DataSet(
                image_lmdb_file=self.image_lmdb_file,
                label_file=self.label_file,
                s2s1_mapping_file=self.s2s1_mapping_file,
                bands=self.bands,
                process_bands_fn=self.process_bands_fn,
                process_labels_fn=self.process_labels_fn,
                transforms=self.eval_transforms,
                keys=self.keys["validation"],
                return_patchname=False,
                verbose=self.verbose,
                **self.kwargs,
            )
        if stage == "test" or stage is None:
            self.test_dataset = BENv2DataSet(
                image_lmdb_file=self.image_lmdb_file,
                label_file=self.label_file,
                s2s1_mapping_file=self.s2s1_mapping_file,
                bands=self.bands,
                process_bands_fn=self.process_bands_fn,
                process_labels_fn=self.process_labels_fn,
                transforms=self.eval_transforms,
                keys=self.keys["test"],
                return_patchname=False,
                verbose=self.verbose,
                **self.kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
