import json
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import (
    StochasticWeightAveraging,
    ModelCheckpoint,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
import wandb

from data.data_utils import load_data_module, parse_batch, get_dataloader_from_cfg
from models.get_models import get_model
from models.lightning_models import LightningBaseModel
from utility.cluster_logging import logger


def train(cfg: dict, tune=False):
    cfg["method"] = "train"

    # load datamodule
    data_module, cfg = load_data_module(cfg)
    logger.debug(f"Loaded data module {data_module}")

    # load model
    model = get_model(cfg, pretrained=True)
    model.learning_rate = cfg["learning_rate"]

    logger.debug("Start Training")
    prefix_name = f"{cfg['model_name']}_{cfg['dataset_name']}_{cfg['timestamp']}"

    if cfg.get("rrr_explanation", None) is not None:
        prefix_name = f"{prefix_name}_{cfg['rrr_explanation']}"

    cfg["models_path"] = os.path.join(cfg["models_path"], prefix_name)

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    cfg, val_loader = get_dataloader_from_cfg(cfg, loader_name="val")

    val_batch = next(iter(val_loader))
    image_tensor, labels_tensor, _, _, index_tensor, _ = parse_batch(val_batch)

    # start a new wandb run to track this script
    # group is dataset_mode_explanation_method
    group_name = f"{cfg['dataset_name']}_{cfg['mode']}"
    if cfg["mode"] != "normal":
        if cfg.get("normal_segmentations", None):
            group_name += "_normal_segmentations"
        else:
            group_name += f"_{cfg['explanation_methods'][0]}"

    wandb_logger = WandbLogger(
        project="xai_for_rs",
        name=cfg["experiment_name"],
        log_model=True,
        group=group_name,
        tags=[
            cfg["explanation_methods"][0],
            cfg["dataset_name"],
            cfg["mode"],
        ],
    )

    # log all hyperparameters
    wandb_logger.experiment.config.update(cfg)

    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        # GradientAccumulationScheduler(scheduling={0: 4, 4: 2, 6: 1}),
        StochasticWeightAveraging(swa_lrs=1e-2),
        ModelCheckpoint(
            monitor="val_accuracy",
            mode="max",
            dirpath=cfg["training_root_path"],
            filename="{epoch}-{val_accuracy:.2f}",
        ),
        # image_preds_logger,
    ]
    strategy = "auto"

    # init trainer
    trainer = Trainer(
        default_root_dir=cfg["training_root_path"],
        max_epochs=cfg["max_epochs"],
        callbacks=callbacks,
        accelerator="auto",
        strategy=strategy,
        gradient_clip_val=1,
        # devices=[gpu],
        inference_mode=False,  # we always want gradients for RRR
        # log_every_n_steps=20,
        logger=wandb_logger,
    )
    if tune:
        tune_trainer(
            cfg,
            data_module,
            model,
            trainer,
            tune_learning_rate=True,
            tune_batch_size=True,
        )

    trainer.fit(model, data_module)
    model.metrics_manager.plot(stage="val")

    trainer.test(model, data_module)
    model.metrics_manager.plot(stage="test")

    save_model(cfg, model)


def tune_trainer(
    cfg,
    data_module,
    model,
    trainer,
    tune_batch_size=False,
    tune_learning_rate=False,
    max_lr=0.025,
):
    tuner = Tuner(trainer)

    if tune_batch_size:
        tuner.scale_batch_size(
            model, datamodule=data_module, max_trials=7
        )  # max batchsize 256

    if tune_learning_rate:
        # Run learning rate finder
        lr_finder = tuner.lr_find(model, datamodule=data_module)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig(os.path.join(cfg["training_root_path"], "lr_finder.png"))

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        # new_lr = max_lr if new_lr > max_lr else new_lr

        logger.debug(f"New learning rate: {new_lr}")
        model.learning_rate = new_lr


def save_model(cfg: dict, model: torch.nn.Module):
    # get timestamp
    model_name = "final_model.pt"
    model_path = os.path.join(cfg["training_root_path"], model_name)
    torch.save(model.state_dict(), model_path)
    logger.debug(f"Saved model to {model_path}")


class ImagePredictionLogger(Callback):
    def __init__(
        self,
        val_images: torch.Tensor,
        val_labels: torch.Tensor,
        val_indices: torch.Tensor = None,
        num_samples: int = 5,
    ):
        super().__init__()
        self.val_imgs, self.val_labels = val_images, val_labels
        self.val_indices = (
            val_indices if val_indices is not None else torch.arange(len(val_images))
        )
        self.num_samples = num_samples
        self.test_table = wandb.Table(
            columns=["id", "image", "prediction", "label", "logits"]
        )

    def on_validation_epoch_end(self, trainer, pl_module: LightningBaseModel):
        log_images = self.val_imgs.to(device=pl_module.device)[: self.num_samples]
        log_labels = self.val_labels.to(device=pl_module.device)[: self.num_samples]
        log_indices = self.val_indices[: self.num_samples]

        # Get model prediction
        log_preds, log_logits = pl_module.prediction_step(log_images)

        assert torch.allclose((log_logits > 0.5).long(), log_preds)

        for idx, (img, label, pred, logit) in enumerate(
            zip(log_images, log_labels, log_preds, log_logits)
        ):
            img_id = str(log_indices[idx].item())

            # Add data to table, converting lists to JSON strings for readability if necessary
            self.test_table.add_data(
                img_id,
                wandb.Image(img.cpu()),  # Ensure image tensor is on CPU
                json.dumps(pred.int().cpu().tolist()),  # Convert list to JSON string
                json.dumps(label.int().cpu().tolist()),
                json.dumps(
                    torch.round(logit, decimals=2).cpu().tolist()
                ),  # Convert logits to JSON string
            )
        # annoying workaround for https://github.com/wandb/wandb/issues/2981
        new_table = wandb.Table(
            columns=self.test_table.columns, data=self.test_table.data
        )
        trainer.logger.experiment.log({"val_predictions": new_table}, commit=False)
