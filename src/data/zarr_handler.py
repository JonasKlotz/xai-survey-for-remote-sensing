import os
from typing import Union, List

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class ZarrHandler:
    def __init__(
        self,
        name: str,
        results_dir: str,
        chunks: tuple = None,
        folder_name: str = None,
    ):
        self.name = name
        self.results_dir = results_dir
        self.chunks = chunks
        self.zarr = None
        self.path = os.path.join(results_dir, folder_name)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.path = os.path.join(self.path, name + ".zarr")

    def append(self, data: Union[torch.Tensor, np.ndarray], batchify: bool = False):
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()

        if batchify:
            # batchify data
            data = np.expand_dims(data, axis=0)

        if not self.zarr:
            self.zarr = zarr.open_array(self.path, mode="w", shape=data.shape)
            self.zarr[:] = data
        else:
            self.zarr.append(data, axis=0)

    def __getitem__(self, index):
        return self.zarr[index]

    @property
    def shape(self):
        return self.zarr.shape


class ZarrGroupHandler:
    def __init__(self, path: str, keys: List[str], storage_system="zarr"):
        self.path = path
        self.keys = keys
        if not path.endswith(f".{storage_system}"):
            path += f".{storage_system}"
        if storage_system == "zarr":
            self.store = zarr.DirectoryStore(path)
        elif storage_system == "lmdb":
            self.store = zarr.LMDBStore(path)
        else:
            raise ValueError(f"Storage system {storage_system} not supported")
        self.zarr_group = None

    def __getitem__(self, index):
        items = {}
        for array_name in self.zarr_group.keys():
            items[array_name] = self.zarr_group[array_name][index]
        return items

    def __len__(self):
        array_sizes = [
            len(self.zarr_group[array_name]) for array_name in self.zarr_group.keys()
        ]
        assert (
            len(set(array_sizes)) == 1
        ), "All arrays in the group must have the same length"
        return array_sizes[0]

    def append(self, data: dict):
        if not self.zarr_group:
            self.zarr_group = zarr.group(store=self.store)
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().detach().numpy()
                self.zarr_group.create_dataset(
                    key, data=value, chunks=True, overwrite=True
                )
        else:
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().detach().numpy()
                self.zarr_group[key].append(value, axis=0)


class ZarrGroupDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        filter_keys: List[str] = None,
    ):
        self.file_path = file_path
        self.zarr_group = self.load_zarr_group()
        self.filter_keys = filter_keys

    def load_zarr_group(self):
        group = zarr.open_group(self.file_path, mode="r")
        return group

    def __getitem__(self, index):
        items = []
        if self.filter_keys:
            keys = self.filter_keys
        else:
            keys = self.zarr_group.keys()

        for array_name in keys:
            items.append(self.zarr_group[array_name][index])
        return items

    def __len__(self):
        array_sizes = [
            len(self.zarr_group[array_name]) for array_name in self.zarr_group.keys()
        ]
        assert (
            len(set(array_sizes)) == 1
        ), "All arrays in the group must have the same length"
        return array_sizes[0]

    def __repr__(self):
        return f"ZarrDataset({self.file_path})"

    def __str__(self):
        return f"ZarrDataset({self.file_path})"

    def get_keys(self):
        if self.filter_keys:
            return self.filter_keys
        return self.zarr_group.keys()


def get_zarr_dataloader(
    cfg: dict,
    filter_keys: List[str] = None,
):
    dataset = ZarrGroupDataset(cfg["zarr_path"], filter_keys=filter_keys)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        shuffle=False,
    )
    return dataloader, dataset.get_keys()


def load_batches(cfg: dict):
    if not cfg["zarr_path"]:
        return load_most_recent_batches(results_dir=cfg["results_path"])

    else:
        return _load_group(cfg["zarr_path"])


def load_most_recent_batches(results_dir: str):
    """
    Load the most recent batches of data stored in .zarr files from a specified directory.

    This function lists all folders in the specified directory, filters for folders starting with "batches_",
    and sorts them to get the most recent folder. If the directory is empty, it gets the next folder.
    It then lists all files in the last folder, filters for files ending with .zarr, and loads the zarr files.
    It also removes the .zarr ending from file names and creates a dictionary with zarr files and the corresponding explanation name.

    Args:
        results_dir (str): The directory from which to load the most recent batches.

    Returns:
        dict: A dictionary where the keys are the names of the explanations and the values are the loaded zarr files.
    """
    # get all folders in results_dir
    folders = os.listdir(results_dir)
    # filter for folders starting with a_batch
    folders = [
        folder
        for folder in folders
        if folder.startswith("explanations_") or folder.startswith("batches")
    ]
    # get the most recent folder, sort the string name
    folders.sort(key=lambda x: "{0:0>8}".format(x).lower())
    # get the last folder
    last_folder = folders[-1]
    # while directory is empty, get the next folder
    while not os.listdir(os.path.join(results_dir, last_folder)):
        folders.pop()
        # remove the last folder
        os.rmdir(os.path.join(results_dir, last_folder))
        last_folder = folders[-1]
    # get all files in the last folder
    files_path = os.path.join(results_dir, last_folder)

    group = _load_group(files_path)

    return group


def _load_group(group_path):
    group = zarr.open_group(group_path, mode="r")
    return group
