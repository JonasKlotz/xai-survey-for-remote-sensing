import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from data.get_data_modules import load_data_module
from models.get_models import get_model
from models.lightningresnet import LightningResnet
from training.rrr_loss import RightForRightReasonsLoss
from xai.xai_methods.gradcam_impl import GradCamImpl

# Note - you must have torchvision installed for this example


def train(
    cfg: dict,
):
    # load datamodule
    data_module = load_data_module(cfg["dataset_name"])
    # load model
    model = get_model(cfg, num_classes=data_module.num_classes, input_channels=data_module.dims[0])

    # init trainer
    trainer = Trainer(
        max_epochs=cfg['max_epochs'],
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, data_module)
    model_name = f"{cfg['model_name']}18_{cfg['dataset_name']}1.pt"
    model_path = os.path.join(cfg['model_dir_path'], model_name)
    torch.save(model.state_dict(), model_path)
