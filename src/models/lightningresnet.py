import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import (
    AveragePrecision,
    Accuracy,
    F1Score,
)

from src.models.interpretable_resnet import get_resnet


class LightningResnet(LightningModule):
    def __init__(
        self,
        resnet_layers=18,
        input_channels=3,
        num_classes=10,
        lr=0.001,
        batch_size=32,
        freeze=False,
    ):
        super(LightningResnet, self).__init__()
        self.save_hyperparameters(ignore=["loss"])

        self.model = get_resnet(resnet_layers)

        # replace the first conv layer
        self.model.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.metrics_calculator = MetricsCalculator(num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # for deepglobe:
        images, target, idx, segments = batch

        y_hat = self.model(images)
        loss = self.loss(y_hat, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def predict(self, batch):
        images, target, idx, segments = batch

        y_hat = self.model(images)
        # todo find if name logits is right, sigmoid is normalization
        logits = torch.sigmoid(y_hat)
        return logits, y_hat

    def evaluate(self, batch, stage=None):
        images, target, idx, segments = batch

        logits, y_hat = self.predict(batch)

        # loss cannot be part of self as it would be interpreted as layer and need a ruling
        loss = nn.BCEWithLogitsLoss()(y_hat, target)

        target = target.long()
        metrics = self.metrics_calculator.calculate_metrics(logits, target)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            for name, value in metrics.items():
                self.log(f"{stage}_{name}", value, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class MetricsCalculator:
    def __init__(self, num_classes):
        self.metrics = {
            "f1_score_macro": F1Score(
                task="multilabel",
                average="macro",
                threshold=0.5,
                num_labels=num_classes,
            ),
            "f1_score_micro": F1Score(
                task="multilabel",
                average="micro",
                threshold=0.5,
                num_labels=num_classes,
            ),
            "average_precision_macro": AveragePrecision(
                task="multilabel", average="macro", num_labels=num_classes
            ),
            "accuracy": Accuracy(
                task="multilabel", threshold=0.5, num_labels=num_classes
            ),
        }

    def calculate_metrics(self, logits, target):
        result = {}
        for name, metric in self.metrics.items():
            result[name] = metric(logits, target)
        return result
