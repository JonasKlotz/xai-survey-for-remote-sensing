import glob
import os

import torch

from models.lightningresnet import LightningResnet


def get_model(
    config: dict, num_classes: int, input_channels: int, pretrained: bool = False
):
    if config["model_name"] == "resnet":
        return get_lightning_resnet(
            config, num_classes, input_channels, pretrained=pretrained
        )

    raise ValueError(f"Model name {config['model_name']} not supported.")


def get_lightning_resnet(
    cfg: dict, num_classes: int, input_channels: int, pretrained: bool
):
    model = LightningResnet(
        num_classes=num_classes,
        input_channels=input_channels,
        resnet_layers=cfg["layer_number"],
        lr=cfg["learning_rate"],
        batch_size=cfg["data"]["batch_size"],
    )
    if pretrained:
        model.load_state_dict(load_most_recent_model(cfg))
    return model


def load_most_recent_model(cfg: dict):
    """
    Load the most recent model of a specific type from the models directory.

    This function lists all the .pt files in the models directory that match the model name pattern
    constructed from the 'model_name', 'layer_number', and 'dataset_name' keys in the cfg dictionary.
    It then sorts these files based on their modification time and loads the most recent one.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Must contain 'models_path', 'model_name', 'layer_number', and 'dataset_name' keys.

    Returns
    -------
    model : torch.nn.Module or None
        The state dict of the most recent model.

    Raises
    ------
    FileNotFoundError
        If the models directory does not exist or is not a directory.
    """
    # Get the path to the models directory
    models_path = cfg.get("models_path")
    if not models_path:
        raise FileNotFoundError("Models path is not provided in the configuration.")

    # Construct the model name pattern
    model_name_pattern = (
        f"{cfg['model_name']}{cfg['layer_number']}_{cfg['dataset_name']}*.pt"
    )

    # List the .pt files in the models directory that match the model name pattern
    model_files = glob.glob(os.path.join(models_path, model_name_pattern))
    if not model_files:
        raise FileNotFoundError(
            f"No model files found in the models directory that match the pattern {model_name_pattern}."
        )

    # Sort the model files based on the timestamp in their names
    model_files.sort(key=os.path.getmtime, reverse=True)

    # Load the most recent model
    most_recent_model_file = model_files[0]
    model = torch.load(most_recent_model_file)

    return model
