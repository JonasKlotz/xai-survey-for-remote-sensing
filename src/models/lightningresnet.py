import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import (
    AveragePrecision,
    Accuracy,
    F1Score,
    MetricCollection,
)

from src.models.interpretable_resnet import get_resnet

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
        task="multilabel",
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

        # replace the last layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # unfreeze the model
        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = True

        if task == "multilabel":
            metrics = MetricCollection(
                {
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
            )
        elif task == "multiclass":
            metrics = MetricCollection(
                {
                    "f1_score_macro": F1Score(
                        task="multiclass",
                        average="macro",
                        num_classes=num_classes,
                    ),
                    "f1_score_micro": F1Score(
                        task="multiclass",
                        average="micro",
                        num_classes=num_classes,
                    ),
                    "average_precision_macro": AveragePrecision(
                        task="multiclass", average="macro", num_classes=num_classes
                    ),
                    "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                }
            )

        self.task = task
        print(metrics)
        # self.valid_metrics = metrics.clone(
        #     prefix="val_"
        # )  # todo comment when calculating LRP
        self.valid_metrics = metrics  # None

        self.loss_name = loss_name

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # for deepglobe:
        images, target, _ = batch

        y_hat = self.model(images)

        loss, logits = self._calc_loss(y_hat, target)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def evaluate(self, batch, stage=None):
        print(batch.shape)
        images, target, _ = batch

        y_hat = self.model(images)

        loss, logits = self._calc_loss(y_hat, target)

        target = target.long()
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log_dict(
                self.valid_metrics(logits, target), prog_bar=True, sync_dist=True
            )

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

    def on_validation_epoch_end(self):
        pass  # self.valid_metrics.reset()

    def _calc_loss(self, y_hat, target):
        if self.task == "multilabel":
            loss = nn.BCEWithLogitsLoss()(y_hat, target)
            logits = torch.sigmoid(y_hat)
        elif self.task == "multiclass":
            loss = nn.CrossEntropyLoss()(y_hat, target)
            logits = torch.softmax(y_hat, dim=1)
        else:
            raise ValueError(f"Task {self.task} not supported.")
        return loss, logits
