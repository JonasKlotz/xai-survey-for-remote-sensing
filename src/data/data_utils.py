import os

import rasterio
from pytorch_lightning import LightningDataModule
import zarr


def write_tiff(directory, matrix, metadata):
    """
    This function get Numpy ndarray and related metadata to create
    a .tif file as the results of process.
    \nInputs=> A directory for exported .tif file, Numpy and meta
    \nOutputs=> A path to result .tif file
    """
    file_name = os.path.split(directory)[1]
    kwargs = metadata
    kwargs.update(dtype=rasterio.float32, count=1)
    try:
        with rasterio.open(directory, "w", **kwargs) as dst:
            dst.write_band(1, matrix.astype(rasterio.float32))
            print("\n File was created successfully.\n%s" % file_name)
    except:
        raise Exception("Error in exporting dataset to .tif!")


def read_tif(path):
    """Read a tif file

    :param path:  path to the tif file
    :return: np array of the image and the metadata
    """
    with rasterio.open(path) as image:
        arr, meta = image.read(), image.meta
        return arr, meta


def get_loader_for_datamodule(data_module: LightningDataModule):
    """
    This function get a data module and return a dictionary of loaders
    \nInputs=> A data module
    \nOutputs=> A dictionary of loaders
    """
    data_module.prepare_data()
    data_module.setup()

    loaders = {
        "train": data_module.train_dataloader(),
        "val": data_module.val_dataloader(),
        "test": data_module.test_dataloader(),
    }
    return loaders
