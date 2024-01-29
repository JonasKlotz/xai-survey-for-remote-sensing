import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    StochasticWeightAveraging,
    ModelCheckpoint,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from data.data_utils import load_data_module
from models.get_models import get_model
from utility.cluster_logging import logger


def train(
    cfg: dict,
):
    # load datamodule
    data_module = load_data_module(cfg)
    logger.debug(f"Loaded data module {data_module}")

    # load model
    model = get_model(
        cfg,
        num_classes=data_module.num_classes,
        input_channels=data_module.dims[0],
        pretrained=True,
    )
    logger.debug("Start Training")
    prefix_name = f"{cfg['model_name']}{cfg['layer_number']}_{cfg['dataset_name']}_{cfg['timestamp']}"
    cfg["models_path"] = os.path.join(cfg["models_path"], prefix_name)
    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        # GradientAccumulationScheduler(scheduling={0: 4, 4: 2, 6: 1}),
        StochasticWeightAveraging(swa_lrs=1e-2),
        ModelCheckpoint(
            dirpath=cfg["models_path"],
            filename="{epoch}-{val_loss:.2f}",
        ),
    ]
    multiGPU = False
    strategy = "auto"
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            multiGPU = True
            logger.debug(f"Using {torch.cuda.device_count()} GPUs")
        else:
            logger.debug(f"Using {torch.cuda.device_count()} GPU")
        if multiGPU:
            model = torch.nn.parallel.DistributedDataParallel(model)
            strategy = "ddp"
    else:
        strategy = None

    # init trainer
    trainer = Trainer(
        max_epochs=cfg["max_epochs"],
        callbacks=callbacks,
        accelerator="auto",
        strategy=strategy,
    )

    trainer.fit(model, data_module)
    try:
        save_model(cfg, model)
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def save_model(cfg: dict, model: torch.nn.Module):
    # get timestamp
    model_name = f"epochs_ {cfg['max_epochs']}.pt"
    model_path = os.path.join(cfg["models_path"], model_name)
    torch.save(model.state_dict(), model_path)
    logger.debug(f"Saved model to {model_path}")
