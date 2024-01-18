import torch

from models.lightningresnet import LightningResnet
from utility.cluster_logging import logger


def get_model(
    config: dict,
    num_classes: int,
    input_channels: int,
    self_trained: bool = False,
    pretrained: bool = False,
):
    if config["model_name"] == "resnet":
        return get_lightning_resnet(
            config,
            num_classes,
            input_channels,
            self_trained=self_trained,
            pretrained=pretrained,
        )

    raise ValueError(f"Model name {config['model_name']} not supported.")


def get_lightning_resnet(
    cfg: dict,
    num_classes: int,
    input_channels: int,
    self_trained: bool,
    pretrained: bool,
):
    model = LightningResnet(
        num_classes=num_classes,
        input_channels=input_channels,
        resnet_layers=cfg["layer_number"],
        lr=cfg["learning_rate"],
        batch_size=cfg["data"]["batch_size"],
        pretrained=pretrained,
        loss_name=cfg["loss_name"],
    )
    if self_trained:
        state_dict = load_model(cfg)
        model.load_state_dict(
            state_dict,
            strict=False,
        )

    logger.debug(
        f"Loaded model {cfg['model_name']}, pretrained from imagenet: {pretrained}, self trained: {self_trained}"
    )
    return model


def load_model(cfg: dict):
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
    models_path = cfg.get("model_path")
    if not models_path:
        raise FileNotFoundError("Models path is not provided in the configuration.")

    model = torch.load(models_path, map_location=torch.device(cfg["device"]))

    return model
