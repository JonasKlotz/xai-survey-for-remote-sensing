from pytorch_lightning import LightningDataModule


def get_loader_for_datamodule(
    data_module: LightningDataModule, loader_name: str = "test"
):
    """
    This function get a data module and return a dictionary of loaders
    \nInputs=> A data module
    \nOutputs=> A dictionary of loaders
    """
    data_module.prepare_data()
    data_module.setup()

    if loader_name == "test":
        return data_module.test_dataloader()
    elif loader_name == "train":
        return data_module.train_dataloader()
    elif loader_name == "val":
        return data_module.val_dataloader()
    raise ValueError(f"Loader name {loader_name} not supported.")
