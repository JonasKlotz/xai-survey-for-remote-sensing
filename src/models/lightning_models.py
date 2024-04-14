import json
import os.path
from abc import abstractmethod

import torch
import torch.nn as nn
import wandb
from einops import einops
from pytorch_lightning import LightningModule
from torchvision import models
from torchvision.models import VGG16_Weights

from data.constants import DEEPGLOBE_IDX2NAME
from src.models.interpretable_resnet import get_resnet
from training.augmentations import CutMix_Xai, CutMix_segmentations
from training.metrics import TrainingMetricsManager
from training.rrr_loss import RightForRightReasonsLoss
from utility.cluster_logging import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LightningBaseModel(LightningModule):
    def __init__(
        self,
        freeze=False,
        config: dict = None,
    ):
        super(LightningBaseModel, self).__init__()
        self.log_next = False
        self.save_hyperparameters(ignore=["loss"])
        self.task = config["task"]
        self.cutmix_threshold = 0.5
        self.max_epochs = config["max_epochs"]

        self.mode = config.get("mode", "normal")

        # Parameters for RRR loss
        self.loss_name = config.get("loss", "regular")
        self.rrr_explanation = config.get("rrr_explanation", None)
        self.rrr_lambda = config.get("rrr_lambda", 1)
        self.dataset_name = config.get("dataset_name", "unknown")
        self.rrr_logged = False

        # Hyperparameters
        self.learning_rate = config["learning_rate"]
        self.momentum = config["momentum"]
        self.input_channels = config["input_channels"]
        self.num_classes = config["num_classes"]

        # Parameters for the cutmix augmentation
        if self.mode == "cutmix":
            if config.get("normal_segmentations", False):
                self.cutmix_mode = "segmentations"
            else:
                self.cutmix_mode = "XAI"

            logger.debug(f"Cutmix mode: {self.cutmix_mode}")

        self.max_aug_area = config.get("max_aug_area", 0.5)
        self.min_aug_area = config.get("min_aug_area", 0.1)

        self.backbone = self.get_backbone(config)

        self.trained_epochs = 0

        self.config = config

        self.optimizer = None

        self.first_batch_save_path = os.path.join(
            config["results_path"], config["experiment_name"], "first_batch.pth"
        )
        os.makedirs(os.path.dirname(self.first_batch_save_path), exist_ok=True)
        self.first_batch_saved = False

        # unfreeze the model
        if not freeze:
            logger.debug("Unfreezing the model")
            for param in self.backbone.parameters():
                param.requires_grad = True

        if config.get("method") == "train":
            self.metrics_manager = TrainingMetricsManager(config)
        else:
            self.metrics_manager = None

        # setup explanations if we want to generate them in the training loop
        self.generate_explanations = config.get("setup_explanations", False)
        if self.generate_explanations:
            self.config["setup_explanations"] = False
            # hacky cracky shit
            from models.get_models import get_lightning_vgg
            from xai.explanations.explanation_manager import ExplanationsManager

            model = get_lightning_vgg(config, self_trained=True, pretrained=False)
            self.explanation_manager = ExplanationsManager(config, model, save=False)

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
            predictions = (logits > 0.5).long()
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
        if self.metrics_manager is not None:
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
        if self.metrics_manager is not None:
            self.metrics_manager.update(normalized_probabilities, target, stage=stage)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=5e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(self.max_epochs * 0.3),
        )

        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler}

    def on_train_epoch_start(self):
        if self.metrics_manager is not None:
            self.metrics_manager.increment("train")

    def on_validation_epoch_start(self):
        if self.metrics_manager is not None:
            self.metrics_manager.increment("val")

    def on_test_epoch_start(self):
        if self.metrics_manager is not None:
            self.metrics_manager.increment("test")

    def on_validation_epoch_end(self) -> None:
        if self.metrics_manager is not None:
            self.log_dict(
                self.metrics_manager.compute(stage="val"), prog_bar=True, sync_dist=True
            )
        self.log_next = True

    def on_test_epoch_end(self) -> None:
        if self.metrics_manager is not None:
            self.log_dict(
                self.metrics_manager.compute(stage="test"),
                prog_bar=True,
                sync_dist=True,
            )

    def on_train_epoch_end(self) -> None:
        self.trained_epochs += 1

    def calc_loss(self, images, segmentations, stage, target):
        y_hat = self.backbone(images)
        if self.loss_name == "rrr" and stage == "train" and self.trained_epochs >= 3:
            if not self.rrr_logged:
                logger.debug("Starting RRR loss")
                self.rrr_logged = True

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
            self.log(
                f"{stage}_regular_loss",
                loss - explanation_loss,
                prog_bar=True,
                sync_dist=True,
            )
        else:
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
            predictions = (normalized_probabilities > self.cutmix_threshold).long()
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

    ####################################################################################################################
    #                                     Functions for the augmentation
    ####################################################################################################################
    def _augment(self, batch):
        # image only augmentation

        self.aug_p: float = 0.5
        self.overhead: int = 10
        self.cutmix_threshold = 10  # TMAP Threshhold
        self.segmentation_threshold = 0  # 0.1

        self._assert_augmentations(batch)

        if self.cutmix_mode == "segmentations":
            batch = CutMix_segmentations(
                batch=batch,
                max_aug_area=self.max_aug_area,
                min_aug_area=self.min_aug_area,
                aug_p=self.aug_p,
                overhead=self.overhead,
            )
        elif self.cutmix_mode == "XAI":
            batch = CutMix_Xai(
                batch=batch,
                max_aug_area=self.max_aug_area,
                min_aug_area=self.min_aug_area,
                aug_p=self.aug_p,
                overhead=self.overhead,
                threshold=self.cutmix_threshold,
                segmentation_threshold=self.segmentation_threshold,
            )

        return batch

    def _assert_augmentations(self, batch):
        if self.min_aug_area > self.max_aug_area:
            raise ValueError(
                f"""The value of max_aug_area: {self.max_aug_area} is less or equal than min_aug_area: {self.min_aug_area},
                but max_aug_area should be strictly greater than min_aug_area!"""
            )
        assert (
            "features" in batch.keys()
            and "targets" in batch.keys()
            and "xai_segmentations" in batch.keys()
        ), f"The batch should contain the keys 'features', 'targets' and 'xai_segmentations'! The keys in the batch are: {batch.keys()}"

    def on_before_batch_transfer(self, batch, dataloader_idx=0):
        """
        This function is called before the batch is transferred to the device. It is used to augment the batch with
        cutmix and xai explanations.

        Parameters
        ----------
        batch
        dataloader_idx

        Returns
        -------

        """
        # we dont want any augmentation for multiclass
        if self.task == "multiclass":
            return batch
        if self.mode == "cutmix":
            return self._on_before_batch_transfer_cutmix(batch)
        elif self.mode == "rrr":
            return self._on_before_batch_transfer_rrr(batch)
        if self.log_next:
            self._log_segmentation(batch)
            self.log_next = False
        return batch

    def _on_before_batch_transfer_cutmix(self, batch):
        if self.training and not self.first_batch_saved:
            old_labels = batch["targets"].clone().detach()
            old_images = batch["features"].clone().detach()
        if self.training:
            if self.generate_explanations:
                attr = self._generate_explanations_as_masks(batch)
                batch["xai_segmentations"] = attr
            # else we use what is provided in the batch
            batch = self._augment(batch)
        # checking if first batch should be saved
        if self.training and not self.first_batch_saved:
            new_labels = batch["targets"]
            new_images = batch["features"]

            self._log_labels(
                new_labels,
                old_labels,
                new_images,
                old_images,
            )
            print("Save first batch....")

            self.log_xai_segmentations(batch)

            # we dont want to override existing files
            if os.path.exists(self.first_batch_save_path):
                raise FileExistsError(
                    f"The path {self.first_batch_save_path} already exists. Please specify a different location by specifying first_batch_save_path!"
                )

            torch.save(batch, self.first_batch_save_path)
            self.first_batch_saved = True
        return batch

    def _on_before_batch_transfer_rrr(self, batch):
        if self.log_next:
            self._log_rrr_segmentation(batch)
            self.log_next = False
        return batch

    def _log_segmentation(self, batch):
        # explanation_method_name = (self.rrr_explanation,)
        # explanation_kwargs = self.config.get("explanation_kwargs", None)
        # from src.xai.explanations.explanation_manager import _explanation_methods
        #
        # explanation_method_constructor = _explanation_methods[explanation_method_name]
        # explanation_method = explanation_method_constructor(**explanation_kwargs)
        # explanations = explanation_method.explain_batch(batch)
        table = wandb.Table(columns=["ID", "Image"])
        for idx, (img, seg) in enumerate(
            zip(batch["features"], batch["segmentations"])
        ):
            np_seg = seg.clone().detach().cpu().numpy()
            # parse batch
            mask_img = wandb.Image(
                img,
                masks={
                    "ground_truth": {
                        "mask_data": np_seg,
                        "class_labels": DEEPGLOBE_IDX2NAME,
                    }
                },
            )
            table.add_data(idx, mask_img)
        # log the explanations as wandb images
        wandb.log({"Segmentations": table})

    def _log_rrr_segmentation(self, batch):
        # explanation_method_name = (self.rrr_explanation,)
        # explanation_kwargs = self.config.get("explanation_kwargs", None)
        # from src.xai.explanations.explanation_manager import _explanation_methods
        #
        # explanation_method_constructor = _explanation_methods[explanation_method_name]
        # explanation_method = explanation_method_constructor(**explanation_kwargs)
        # explanations = explanation_method.explain_batch(batch)
        table = wandb.Table(columns=["ID", "Image"])
        for idx, (img, seg) in enumerate(
            zip(batch["features"], batch["segmentations"])
        ):
            np_seg = seg.clone().detach().cpu().numpy()
            # parse batch
            mask_img = wandb.Image(
                img,
                masks={
                    "ground_truth": {
                        "mask_data": np_seg,
                        "class_labels": DEEPGLOBE_IDX2NAME,
                    }
                },
            )
            table.add_data(idx, mask_img)
        # log the explanations as wandb images
        wandb.log({"Segmentations": table})

    def _log_labels(
        self,
        new_labels,
        old_labels,
        new_images,
        old_images,
    ):
        # Initialize a wandb Table with columns for old and new labels
        columns = ["Old Labels", "Cutmix Derived Labels", "Old Image", "Cutmix Image"]
        wandb_table = wandb.Table(columns=columns)
        # Add rows to the wandb Table
        for (
            old_label,
            new_label,
            new_image,
            old_image,
        ) in zip(
            old_labels,
            new_labels,
            new_images,
            old_images,
        ):
            wandb_table.add_data(
                json.dumps(old_label.int().cpu().tolist()),
                json.dumps(new_label.int().cpu().tolist()),
                wandb.Image(old_image.cpu(), caption="Old Image"),
                wandb.Image(new_image.cpu(), caption="Cutmix Image"),
            )
        # Log the table to wandb. Ensure your wandb run is initiated before this step.
        wandb.log({"Batch 1 Label Comparison": wandb_table})

    def log_xai_segmentations(self, batch):
        if self.mode != "cutmix":
            return

        segmentations = batch["xai_segmentations"].clone().detach()
        # squeeze the segmentation masks
        segmentations = segmentations.squeeze().float()
        segmentations = einops.rearrange(segmentations, "b h w c -> b c h w")
        data = []
        for batch_idx in range(segmentations.shape[0]):
            row = []
            for class_idx in range(segmentations.shape[1]):
                caption = f"Class {class_idx}, sum {segmentations[batch_idx, class_idx].sum()}"
                segmentations_mask = (
                    segmentations[batch_idx, class_idx] > self.segmentation_threshold
                ).float()
                row.append(wandb.Image(segmentations_mask, caption=caption))
            data.append(row)

        table = wandb.Table(
            data=data,
            columns=[f"Class {i}" for i in range(segmentations.shape[1])],
        )
        wandb.log({"Segmentation Masks": table})

    def _generate_explanations_as_masks(self, batch):
        # generate explanations
        self.explanation_manager.model.to(self.device)
        batch["features"].to(self.device)
        self.explanation_manager.model.eval()
        explanations = self.explanation_manager.explain_batch(batch, explain_all=True)
        explanation_method_name = list(self.explanation_manager.explanations.keys())[0]
        attr = explanations[f"a_{explanation_method_name}_data"]
        # invert label batcn
        targets = batch["targets"]
        broadcasted_targets = targets.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # null the attributes of the non-target class
        attr = attr * broadcasted_targets.to(attr.device)
        # threshold the attributes
        attr = (attr > self.segmentation_threshold).int()
        # squeeze everything but the batch dimension
        attr = attr.squeeze()
        # move cam dimension to last dimension -> (batch, h, w, class_maps) using torch einsum
        attr = einops.rearrange(attr, "b c h w -> b h w c")
        # replace segmentations in batch with the explanations
        return attr

    ####################################################################################################################
    #                                     END Functions for the augmentation
    ####################################################################################################################


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
