import os

from torchgeo.datamodules import EuroSATDataModule

import ssl
import torch.nn as nn
import math
import torch
from collections import OrderedDict
from torch.utils import model_zoo
import tempfile
import torch.nn.functional as F
import timm

from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.functional import accuracy
from torchgeo.models import ResNet18_Weights


class SurrogateModel(LightningModule):
    def __init__(self, blackbox, surrogate, lr=0.001, loss=nn.BCEWithLogitsLoss()):
        super(SurrogateModel, self).__init__()
        # self.save_hyperparameters()
        self.lr = lr
        self.blackbox = blackbox
        # freeze blackbox
        for param in blackbox.parameters():
            param.requires_grad = False

        self.surrogate = surrogate
        self.loss = loss

    def forward(self, x):
        out = self.surrogate(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)

        blackbox_logits = self.blackbox(x)

        loss = self.loss(logits, blackbox_logits)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        blackbox_logits = self.blackbox(x)
        loss = self.loss(logits, blackbox_logits)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.surrogate.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
