import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from data.get_data_modules import load_data_module
from models.get_models import get_model

from utility.cluster_logging import logger


# Note - you must have torchvision installed for this example


def train(
    cfg: dict,
):
    # load datamodule
    data_module = load_data_module(cfg)
    logger.debug(f"Loaded data module {data_module}")
    # setup data module

    #
    # data_module = DeepGlobeDataModule(cfg, transforms_train, transforms_train)
    # load model
    model = get_model(
        cfg, num_classes=data_module.num_classes, input_channels=data_module.dims[0]
    )
    logger.debug("Start Training")
    # init trainer
    trainer = Trainer(
        max_epochs=cfg["max_epochs"],
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, data_module)

    save_model(cfg, model)


def save_model(cfg: dict, model: torch.nn.Module):
    # get timestamp
    model_name = f"{cfg['model_name']}{cfg['layer_number']}_{cfg['dataset_name']}_epochs{cfg['max_epochs']}_{cfg['timestamp']}.pt"
    model_path = os.path.join(cfg["models_path"], model_name)
    torch.save(model.state_dict(), model_path)
    logger.debug(f"Saved model to {model_path}")
