import torch


from models.lightning_models import LightningResnet, LightningVGG
from utility.cluster_logging import logger


def get_model(config: dict, self_trained: bool = False, pretrained: bool = False):
    if config["model_name"] == "resnet":
        return get_lightning_resnet(
            config,
            self_trained=self_trained,
            pretrained=pretrained,
        )
    elif config["model_name"] == "vgg":
        return get_lightning_vgg(
            config,
            self_trained=self_trained,
            pretrained=pretrained,
        )

    raise ValueError(f"Model name {config['model_name']} not supported.")


def get_lightning_resnet(
    cfg: dict,
    self_trained: bool,
    pretrained: bool,
):
    model = LightningResnet(
        config=cfg,
    )
    if self_trained:
        state_dict = load_state_dict(cfg)
        model.load_state_dict(
            state_dict,
            strict=False,
        )

    logger.debug(
        f"Loaded model {cfg['model_name']}, pretrained from imagenet: {pretrained}, self trained: {self_trained}"
    )
    return model


def get_lightning_vgg(
    cfg: dict,
    self_trained: bool,
    pretrained: bool,
):
    model = LightningVGG(
        config=cfg,
    )
    if self_trained:
        state_dict = load_state_dict(cfg)
        model.load_state_dict(
            state_dict,
            strict=False,
        )

    logger.debug(
        f"Loaded model {cfg['model_name']}, pretrained from imagenet: {pretrained}, self trained: {self_trained}"
    )
    return model


def load_state_dict(cfg: dict):
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

    # check if path ends with .pt
    if models_path.endswith(".pt"):
        model = torch.load(models_path, map_location=torch.device(cfg["device"]))
    elif models_path.endswith(".ckpt"):
        model = torch.load(models_path, map_location=torch.device(cfg["device"]))[
            "state_dict"
        ]
    else:
        raise ValueError("Model path must end with .pt or .ckpt")

    return model
