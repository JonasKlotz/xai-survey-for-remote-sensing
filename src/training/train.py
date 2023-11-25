import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from data.get_data_modules import load_data_module
from models.lightningresnet import LightningResnet
from training.rrr_loss import RightForRightReasonsLoss
from xai.xai_methods.gradcam_impl import GradCamImpl

# Note - you must have torchvision installed for this example

DATA_PATH = os.path.join(os.getcwd(), "..", "..", "data")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = 4 if torch.cuda.is_available() else 0


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model_name="resnet",
    layer_number=18,
    max_epochs=10,
    dataset_name="mnist",
    kwargs=None,
):
    # load datamodule
    data_module = load_data_module(dataset_name)
    rrr_loss = RightForRightReasonsLoss(lambda_=1)
    # load model
    model = LightningResnet(
        num_classes=data_module.num_classes,
        input_channels=data_module.dims[0],
        resnet_layers=layer_number,
    )
    # init trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, data_module)
    model_name = f"{model_name}18_{dataset_name}1.pt"
    model_path = "/home/jonasklotz/Studys/MASTERS/XAI/models/" + model_name
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train(max_epochs=10)
