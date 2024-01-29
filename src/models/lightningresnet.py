import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from src.models.interpretable_resnet import get_resnet
from training.metrics import TrainingMetricsManager

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
        config: dict = None,
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

        self.metrics_manager = TrainingMetricsManager(config)

        self.task = task

        self.loss_name = loss_name

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target, _ = batch

        y_hat = self.model(images)

        loss, logits = self._calc_loss(y_hat, target)

        stage = "train"

        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.metrics_manager.update(logits, target, stage=stage)

        return loss

    def evaluate(self, batch, stage=None):
        images, target, _ = batch

        y_hat = self.model(images)
        loss, logits = self._calc_loss(y_hat, target)
        target = target.long()
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.metrics_manager.update(logits, target, stage=stage)

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

    def on_train_epoch_start(self):
        self.metrics_manager.increment("train")

    def on_validation_epoch_start(self):
        self.metrics_manager.increment("val")

    def on_test_epoch_start(self):
        self.metrics_manager.increment("test")

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            self.metrics_manager.compute(stage="val"), prog_bar=True, sync_dist=True
        )

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            self.metrics_manager.compute(stage="test"), prog_bar=True, sync_dist=True
        )

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
