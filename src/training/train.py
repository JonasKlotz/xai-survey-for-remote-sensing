import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from data.get_data_modules import load_data_module
from models.lightningresnet import LightningResnet

# Note - you must have torchvision installed for this example

DATA_PATH = os.path.join(os.getcwd(), "..", "..", "data")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(training_config: dict):
    # load datamodule
    data_module = load_data_module(training_config["dataset_name"])
    # load model
    model = LightningResnet(num_classes=data_module.num_classes, input_channels=data_module.dims[0])
    # init trainer
    trainer = Trainer(
        max_epochs=training_config["max_epochs"],
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, data_module)
    model_name = f"{training_config['model_name']}18_{training_config['dataset_name']}.pt"
    model_path = "/home/jonasklotz/Studys/MASTERS/XAI/models/"  + model_name
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    # Init DataModule
    # dm = MNISTDataModule()
    # dm = get_cifar_data_module()
    # dm = get_deepglobe_land_cover_dataset()
    channels = 3  # dm.dims[0]
    # Init model from datamodule's attributes
    model = LightningResnet(num_classes=dm.num_classes, input_channels=channels)
    # Init LightningResnet
