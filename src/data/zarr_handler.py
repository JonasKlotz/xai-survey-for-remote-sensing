import os
from typing import Union

import numpy as np
import torch
import zarr


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
    folders = [folder for folder in folders if folder.startswith("batches_")]
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
    files = os.listdir(os.path.join(results_dir, last_folder))
    # filter for files ending with .zarr
    files = [file for file in files if file.endswith(".zarr")]
    # load the zarr files
    zarr_files = [
        zarr.open_array(os.path.join(results_dir, last_folder, file)) for file in files
    ]
    # remove .zarr ending from file names
    files = [file[:-5] for file in files]
    # create dict with zarr files and the corresponsing explanation name
    zarr_files = dict(zip(files, zarr_files))

    return zarr_files
