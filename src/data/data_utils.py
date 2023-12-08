import os
from pytorch_lightning import LightningDataModule




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
