import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import (
    AveragePrecision,
    Accuracy,
    F1Score,
)

from src.models.interpretable_resnet import get_resnet
from training.rrr_loss import RightForRightReasonsLoss
from xai.explanations.explanation_methods.deeplift_impl import DeepLiftImpl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LightningResnet(LightningModule):
    def __init__(
        self,
        resnet_layers=18,
        input_channels=3,
        num_classes=10,
        lr=0.001,
        batch_size=32,
        freeze=False,
        pretrained=False,
        loss_name="bce",
    ):
        super(LightningResnet, self).__init__()
        self.save_hyperparameters(ignore=["loss"])

        self.model = get_resnet(resnet_layers, pretrained=pretrained)

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
        self.loss_name = loss_name

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # for deepglobe:
        images, target, idx, segments = batch

        y_hat = self.model(images)
        # loss cannot be part of self as it would be interpreted as layer and need a ruling
        if self.loss_name == "RRR":
            print("RRR")
            base_loss = nn.BCEWithLogitsLoss()(y_hat, target)

            rrr = RightForRightReasonsLoss(
                lambda_=1, explanation_method=DeepLiftImpl(self)
            )
            self.log("base_loss", base_loss, prog_bar=True)
            loss = rrr(
                x_batch=images,
                y_batch=target,
                s_batch=segments,
                regular_loss_value=base_loss,
            )
            _ = self.model(images)
        else:
            loss = nn.BCEWithLogitsLoss()(y_hat, target)

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
        if self.loss_name == "rrr":
            print("RRR")
            base_loss = nn.BCEWithLogitsLoss()(y_hat, target)

            rrr = RightForRightReasonsLoss(
                lambda_=1, explanation_method=DeepLiftImpl(self)
            )
            if stage:
                self.log(f"{stage}_base_loss", base_loss, prog_bar=True)
            loss = rrr(
                x_batch=images,
                y_batch=target,
                s_batch=segments,
                regular_loss_value=base_loss,
            )
            logits, y_hat = self.predict(batch)  # re-establish grads
        else:
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
    # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metrics-and-devices
    def __init__(self, num_classes: int):
        self.metrics = {
            "f1_score_macro": F1Score(
                task="multilabel",
                average="macro",
                threshold=0.5,
                num_labels=num_classes,
            ).to(device),
            "f1_score_micro": F1Score(
                task="multilabel",
                average="micro",
                threshold=0.5,
                num_labels=num_classes,
            ).to(device),
            "average_precision_macro": AveragePrecision(
                task="multilabel", average="macro", num_labels=num_classes
            ).to(device),
            "accuracy": Accuracy(
                task="multilabel", threshold=0.5, num_labels=num_classes
            ).to(device),
        }

    def calculate_metrics(self, logits, target):
        result = {}
        for name, metric in self.metrics.items():
            result[name] = metric(logits, target)
        return result
