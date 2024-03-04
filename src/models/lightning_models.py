from abc import abstractmethod

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchvision import models
from torchvision.models import VGG16_Weights

from src.models.interpretable_resnet import get_resnet
from training.metrics import TrainingMetricsManager
from training.rrr_loss import RightForRightReasonsLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LightningBaseModel(LightningModule):
    def __init__(
        self,
        freeze=False,
        config: dict = None,
    ):
        super(LightningBaseModel, self).__init__()
        self.save_hyperparameters(ignore=["loss"])
        self.task = config["task"]
        self.threshold = config["threshold"]
        self.max_epochs = config["max_epochs"]

        # Parameters for RRR loss
        self.loss_name = config.get("loss", "regular")
        self.rrr_explanation = config.get("rrr_explanation", None)
        self.rrr_lambda = config.get("rrr_lambda", 1)
        self.dataset_name = config.get("dataset_name", "unknown")

        # Hyperparameters
        self.learning_rate = config["learning_rate"]
        self.momentum = config["momentum"]
        self.input_channels = config["input_channels"]
        self.num_classes = config["num_classes"]

        self.backbone = self.get_backbone(config)

        self.trained_epochs = 0

        self.config = config

        # unfreeze the model
        if not freeze:
            for param in self.backbone.parameters():
                param.requires_grad = True

        if config.get("method") == "train":
            self.metrics_manager = TrainingMetricsManager(config)
        else:
            self.metrics_manager = None

    @abstractmethod
    def get_backbone(self, cfg):
        raise NotImplementedError

    def forward(self, x):
        return self.backbone(x)

    def prediction_step(self, images):
        y_hat = self.backbone(images)
        if self.task == "multiclass":
            logits = torch.softmax(y_hat, dim=1)
            predictions = torch.argmax(logits, dim=1)
        elif self.task == "multilabel":
            logits = torch.sigmoid(y_hat)
            predictions = (logits > self.threshold).long()
        else:
            raise ValueError(f"Task {self.task} not supported.")

        return predictions, logits

    def training_step(self, batch, batch_idx, stage="train"):
        images = batch["features"]
        target = batch["targets"]
        segmentations = batch["segmentations"]

        loss, normalized_probabilities = self.calc_loss(
            images, segmentations, stage, target
        )

        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.metrics_manager.update(normalized_probabilities, target, stage=stage)

        return loss

    def evaluate(self, batch, stage=None):
        images = batch["features"]
        target = batch["targets"]
        segmentations = batch["segmentations"]

        loss, normalized_probabilities = self.calc_loss(
            images, segmentations, stage, target
        )

        target = target.long()
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.metrics_manager.update(normalized_probabilities, target, stage=stage)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=5e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(self.max_epochs * 0.3),
        )

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

    def on_train_epoch_end(self) -> None:
        self.trained_epochs = +1

    def calc_loss(self, images, segmentations, stage, target):
        # After 5 epochs we enable the RRR loss, as it is not stable in the beginning
        if self.loss_name == "rrr" and self.trained_epochs > 5:
            # enable gradients for the explanation loss (if we are in test mode)
            self.requires_grad_(True)
            images.requires_grad = True
            target.requires_grad = True
            segmentations.requires_grad = True

            y_hat = self.backbone(images)
            loss, normalized_probabilities, explanation_loss = self._calc_rrr_loss(
                images, segmentations, target, y_hat
            )
            # log the explanation loss
            self.log(
                f"{stage}_explanation_loss",
                explanation_loss,
                prog_bar=True,
                sync_dist=True,
            )
            # We have to predict again as some of the explanation methods change the gradients
            _ = self.backbone(images)
        else:
            y_hat = self.backbone(images)
            loss, normalized_probabilities = self._calc_regular_loss(y_hat, target)
        return loss, normalized_probabilities

    def _calc_regular_loss(self, unnormalized_logits, target):
        if self.task == "multilabel":
            loss = nn.BCEWithLogitsLoss()(unnormalized_logits, target)
            normalized_probabilities = torch.sigmoid(unnormalized_logits)
        elif self.task == "multiclass":
            loss = nn.CrossEntropyLoss()(unnormalized_logits, target)
            normalized_probabilities = torch.softmax(unnormalized_logits, dim=1)
        else:
            raise ValueError(f"Task {self.task} not supported.")
        return loss, normalized_probabilities

    def _calc_rrr_loss(self, images, segmentations, target, y_hat):
        regular_loss, normalized_probabilities = self._calc_regular_loss(y_hat, target)
        if self.task == "multiclass":
            predictions = torch.argmax(normalized_probabilities, dim=1)
        elif self.task == "multilabel":
            predictions = (normalized_probabilities > self.threshold).long()
        else:
            raise ValueError(f"Task {self.task} not supported.")
        # get prediction values
        rrr = RightForRightReasonsLoss(
            task=self.task,
            num_classes=self.num_classes,
            lambda_=self.rrr_lambda,
            dataset_name=self.dataset_name,
            explanation_method_name=self.rrr_explanation,
            explanation_kwargs=self.config.get("explanation_kwargs", None),
        )
        explanation_loss = rrr(self, images, predictions, segmentations)
        loss = regular_loss + explanation_loss
        return loss, normalized_probabilities, explanation_loss

    @property
    def out_channels(self):
        # dirty hack for quantus input invariance metric, as we have a model wrapper this is necessary
        return self.backbone.features[0].out_channels


class LightningResnet(LightningBaseModel):
    def __init__(
        self,
        freeze=False,
        config: dict = None,
    ):
        super(LightningResnet, self).__init__(
            freeze=freeze,
            config=config,
        )

    def get_backbone(self, cfg):
        backbone = get_resnet(resnet_layers=34, pretrained=True)

        # replace the first conv layer
        backbone.conv1 = nn.Conv2d(
            self.input_channels,
            64,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        # replace the last layer
        backbone.fc = nn.Linear(backbone.fc.in_features, self.num_classes)
        return backbone


class LightningVGG(LightningBaseModel):
    def __init__(
        self,
        freeze=False,
        config: dict = None,
    ):
        super(LightningVGG, self).__init__(
            freeze=freeze,
            config=config,
        )

    def get_backbone(self, cfg):
        backbone = models.vgg16(weights=VGG16_Weights.DEFAULT)

        backbone.features[0] = torch.nn.Conv2d(
            self.input_channels,
            64,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        backbone.classifier[-1] = torch.nn.Linear(4096, self.num_classes)
        return backbone
