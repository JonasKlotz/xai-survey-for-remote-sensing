import os

from models.lightningresnet import LightningResnet
import torch
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pl_bolts.datamodules import CIFAR10DataModule, CityscapesDataModule

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST


DATA_PATH = os.path.join(os.getcwd(), "..", "..", "data")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = 4 if torch.cuda.is_available() else 0


# exemplary data module
class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = DATA_PATH, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)





#
# if __name__ == "__main__":
#     # Init DataModule
#     # dm = MNISTDataModule()
#     dm = get_cifar_data_module()
#     ##dm = get_deepglobe_land_cover_dataset()
#     channels = 3# dm.dims[0]
#     # Init model from datamodule's attributes
#     model = LightningResnet(num_classes=dm.num_classes, input_channels=channels)
#     # Init LightningResnet
#     trainer = Trainer(
#         max_epochs=1,
#         callbacks=[TQDMProgressBar(refresh_rate=20)],
#         accelerator="auto",
#         devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
#     )
#     # Pass the datamodule as arg to trainer.fit to override model hooks :)
#     trainer.fit(model, dm)
#     torch.save(model.state_dict(), "resnet18_cifar.pt")
