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
    else:
        raise ValueError(f"Model name {config['model_name']} not supported.")


def get_lightning_resnet(
    config, num_classes: int, input_channels: int, pretrained: bool
):
    model = LightningResnet(
        num_classes=num_classes,
        input_channels=input_channels,
        resnet_layers=config["layer_number"],
        lr=config["learning_rate"],
        batch_size=config["batch_size"],
    )
    if pretrained:
        model_name = f"resnet18_{config['dataset_name']}.pt"
        model_path = os.path.join(config["models_path"], model_name)
        model.load_state_dict(torch.load(model_path))
    return model
