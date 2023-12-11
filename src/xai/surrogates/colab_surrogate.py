import os

from torchgeo.datamodules import EuroSATDataModule

import ssl
import torch.nn as nn
import math
import torch
from torch.utils import model_zoo
import tempfile
import torch.nn.functional as F
import timm

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.functional import accuracy
from torchgeo.models import ResNet18_Weights

ssl._create_default_https_context = ssl._create_unverified_context

dir_path = ""

__all__ = ["bagnet9", "bagnet17", "bagnet33"]

model_urls = {
    "bagnet9": "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar",
    "bagnet17": "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar",
    "bagnet33": "https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar",
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck, self).__init__()
        # print('Creating bottleneck with kernel size {} and stride {} with padding {}'.format(kernel_size, stride, (kernel_size - 1) // 2))
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )  # changed padding from (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]

        out += residual
        out = self.relu(out)

        return out


class BagNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        strides=[1, 2, 2, 2],
        kernel3=[0, 0, 0, 0],
        num_classes=1000,
        avg_pool=True,
    ):
        self.inplanes = 64
        super(BagNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix="layer1"
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=strides[1],
            kernel3=kernel3[1],
            prefix="layer2",
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=strides[2],
            kernel3=kernel3[2],
            prefix="layer3",
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=strides[3],
            kernel3=kernel3[3],
            prefix="layer4",
        )
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=""):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(
            block(self.inplanes, planes, stride, downsample, kernel_size=kernel)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.fc(x)

        return x


def bagnet33(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(
        Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 1, 1], **kwargs
    )

    if torch.cuda.is_available():
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")

    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_urls["bagnet33"], map_location=map_location)
        )
    return model


def bagnet17(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-17 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(
        Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 1, 0], **kwargs
    )
    if torch.cuda.is_available():
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")

    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_urls["bagnet17"], map_location=map_location)
        )
    return model


def bagnet9(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-9 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(
        Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 0, 0], **kwargs
    )
    if torch.cuda.is_available():
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_urls["bagnet9"], map_location=map_location)
        )
    return model


def get_bagnet(bagnet_layers, pretrained=True, num_classes=1000, num_channels=3):
    if bagnet_layers == 9:
        model = bagnet9(pretrained=pretrained)
    elif bagnet_layers == 17:
        model = bagnet17(pretrained=pretrained)
    elif bagnet_layers == 33:
        model = bagnet33(pretrained=pretrained)
    else:
        raise ValueError("Bagnet layers must be 9, 17, or 33")
    # manually replace output classes, to reuse pretrained weights
    if num_classes != 1000:
        model.fc = nn.Linear(512 * model.block.expansion, num_classes)
    if num_channels != 3:
        model.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=1, stride=1, padding=0, bias=False
        )
    return model


class Resnet(LightningModule):
    def __init__(self, resnet_layers=18, input_channels=10, num_classes=10, lr=0.001):
        super(Resnet, self).__init__()
        self.save_hyperparameters()

        weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
        in_chans = weights.meta["in_chans"]
        print(in_chans)
        model = timm.create_model("resnet18", in_chans=in_chans, num_classes=10)
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.hparams.num_classes
        )

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
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


def get_resnet(resnet_layers=18, input_channels=10, num_classes=10, pretrained=True):
    weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
    in_chans = weights.meta["in_chans"]
    model = timm.create_model("resnet18", in_chans=in_chans, num_classes=10)
    model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
    return model


class SurrogateModel(LightningModule):
    def __init__(self, blackbox, surrogate, lr=0.001):
        super(SurrogateModel, self).__init__()
        # self.save_hyperparameters()
        self.lr = lr
        self.blackbox = blackbox
        self.surrogate = surrogate
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        out = self.surrogate(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, _ = batch["image"], batch["label"]
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


if __name__ == "__main__":
    bagnet = get_bagnet(bagnet_layers=9, num_classes=10, num_channels=13)
    resnet = get_resnet()
    model = SurrogateModel(resnet, bagnet)

    in_tests = "PYTEST_CURRENT_TEST" in os.environ
    root = os.path.join(tempfile.gettempdir(), "deepglobe")
    print(root)
    datamodule = EuroSATDataModule(
        root=root, batch_size=64, num_workers=4, download=True
    )

    experiment_dir = os.path.join(tempfile.gettempdir(), "deepglobe_results")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=experiment_dir, save_top_k=1, save_last=True
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10
    )

    csv_logger = CSVLogger(save_dir=experiment_dir, name="pretrained_weights_logs")

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=[csv_logger],
        default_root_dir=experiment_dir,
        min_epochs=1,
        max_epochs=10,
        fast_dev_run=in_tests,
        # accelerator='gpu', devices=1
    )

    trainer.fit(model=model, datamodule=datamodule)

    trainer.save_checkpoint(os.path.join(experiment_dir, "fullnet_eurosat.ckpt"))

    trainer.test(model=model, datamodule=datamodule)
