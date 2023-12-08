import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from torchmetrics import AveragePrecision, Accuracy, F1Score, Precision, Recall, ConfusionMatrix

from src.models.interpretable_resnet import get_resnet
from src.xai.xai_methods.deeplift_impl import DeepLiftImpl
from training.metrics import calculate_metrics
from utility.cluster_logging import logger


class LightningResnet(LightningModule):
    def __init__(
            self,
            resnet_layers=18,
            input_channels=3,
            num_classes=10,
            lr=0.001,
            batch_size=32,
            freeze=False,
            loss=nn.BCEWithLogitsLoss(),
    ):
        super(LightningResnet, self).__init__()
        self.save_hyperparameters(ignore=['loss'])

        self.model = get_resnet(resnet_layers)
        self.loss = loss

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

        self.f1_score_macro = F1Score(task='multilabel', average='macro', threshold=0.5, num_labels=num_classes)
        self.f1_score_micro = F1Score(task='multilabel', average='micro', threshold=0.5, num_labels=num_classes)
        self.average_precision_macro = AveragePrecision(task='multilabel', average='macro', num_labels=num_classes)
        self.accuracy = Accuracy(task='multilabel',threshold=0.5,  num_labels=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # for deepglobe:
        images, labels, _ = batch

        y_hat = self.model(images)
        loss = self.loss(y_hat, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def predict(self, batch):
        images, target, idx = batch

        y_hat = self.model(images)
        #todo find if name logits is right, sigmoid is normalization
        logits = torch.sigmoid(y_hat)
        return logits, y_hat


    def evaluate(self, batch, stage=None):
        # for deepglobe:
        # x = batch["image"]
        # y = batch["mask"]
        # for tom:
        images, target, idx = batch

        logits, y_hat = self.predict(batch)

        loss = self.loss(y_hat, target)


        target = target.long()
        maf1 = self.f1_score_macro(logits, target)
        mif1 = self.f1_score_micro(logits, target)
        maMAP = self.average_precision_macro(logits, target)
        acc = self.accuracy(logits, target)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_maf1", maf1, prog_bar=True)
            self.log(f"{stage}_mif1", mif1, prog_bar=True)
            self.log(f"{stage}_maMAP", maMAP, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

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
