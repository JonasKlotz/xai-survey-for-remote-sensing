import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    StochasticWeightAveraging,
    ModelCheckpoint,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.tuner.tuning import Tuner

from data.data_utils import load_data_module
from models.get_models import get_model
from utility.cluster_logging import logger


def train(
    cfg: dict,
):
    cfg["method"] = "train"

    # load datamodule
    data_module, cfg = load_data_module(cfg)
    logger.debug(f"Loaded data module {data_module}")

    # load model
    model = get_model(cfg, pretrained=True)
    logger.debug("Start Training")
    prefix_name = f"{cfg['model_name']}_{cfg['dataset_name']}_{cfg['timestamp']}"
    cfg["models_path"] = os.path.join(cfg["models_path"], prefix_name)

    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        # GradientAccumulationScheduler(scheduling={0: 4, 4: 2, 6: 1}),
        StochasticWeightAveraging(swa_lrs=1e-2),
        ModelCheckpoint(
            dirpath=cfg["training_root_path"],
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
            # currently i do not want to use ddp
            model = torch.nn.parallel.DistributedDataParallel(model)
            strategy = "ddp"
    else:
        strategy = "auto"

    # init trainer
    trainer = Trainer(
        default_root_dir=cfg["training_root_path"],
        max_epochs=cfg["max_epochs"],
        callbacks=callbacks,
        accelerator="auto",
        strategy=strategy,
        gradient_clip_val=1,
        log_every_n_steps=20,
    )
    tune_trainer(cfg, data_module, model, trainer, tune_learning_rate=True)

    trainer.fit(model, data_module)
    model.metrics_manager.plot(stage="val")

    trainer.test(model, data_module)
    model.metrics_manager.plot(stage="test")

    try:
        save_model(cfg, model)
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def tune_trainer(
    cfg, data_module, model, trainer, tune_batch_size=False, tune_learning_rate=False
):
    tuner = Tuner(trainer)

    if tune_batch_size:
        tuner.scale_batch_size(model, datamodule=data_module)

    if tune_learning_rate:
        # Run learning rate finder
        lr_finder = tuner.lr_find(model, datamodule=data_module)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig(os.path.join(cfg["training_root_path"], "lr_finder.png"))

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        logger.debug(f"New learning rate: {new_lr}")
        model.learning_rate = new_lr


def save_model(cfg: dict, model: torch.nn.Module):
    # get timestamp
    model_name = "final_model.pt"
    model_path = os.path.join(cfg["training_root_path"], model_name)
    torch.save(model.state_dict(), model_path)
    logger.debug(f"Saved model to {model_path}")
