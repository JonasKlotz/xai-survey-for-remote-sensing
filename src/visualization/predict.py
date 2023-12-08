import numpy as np
import os
import torch
import torch.nn.functional as F

from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from models.lightningresnet import LightningResnet

from data.data_utils import read_tif
from visualization.plot import quant_norm_data, plot_rgb



def load_image_from_datamodule(datamodule, index=None):
    """Load an image from the datamodule

    :param datamodule: the datamodule
    :param index: the index of the image to load
    :param quant_norm: whether to normalize
    :return: image and label
    """

    datamodule.prepare_data()
    datamodule.setup(stage="validate")
    testloader = datamodule.val_dataloader()

    n_samples = len(testloader)
    if index is None:
        index = int(np.random.random() * n_samples)

    subset_indices = [index]  # select your indices here as a list
    subset = torch.utils.data.Subset(testloader.dataset, subset_indices)
    testloader_subset = torch.utils.data.DataLoader(
        subset, batch_size=1, num_workers=0, shuffle=False
    )

    # get the first batch
    batch = next(iter(testloader_subset))
    batch["image"] = batch["image"].squeeze()

    # get the first image
    img = batch["image"]
    # get the first label
    label = batch["label"]

    img = img.numpy()

    return img, label


def predict_from_datamodule(
    model, datamodule, rgb=(3, 2, 1), index=None, labels=None, plot=True
):
    """Predict the label of an image from the datamodule

    :param plot:
    :param labels:
    :param model: the model
    :param datamodule: the datamodule
    :param index: the index of the image to load
    :return: image, label and prediction
    """

    img, label = load_image_from_datamodule(datamodule, index)

    if plot:
        # plot the image
        plot_rgb(
            img,
            rgb=rgb,
            title=labels[label] if labels is not None else "Label: " + str(label),
        )

    # img to tensor
    img = (torch.from_numpy(img)).float()
    img = img.unsqueeze(0)

    # get the prediction
    pred = model(img)
    predicted_class = torch.argmax(pred, dim=1)

    print(
        "Predicted: ",
        labels[predicted_class]
        if labels is not None
        else "Pred: " + str(predicted_class),
    )

    return img, label, pred, predicted_class


if __name__ == "__main__":
    """parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.h5', help='Path to the model')
    parser.add_argument('--image', type=str, default='test.jpg', help='Path to the image')
    args = parser.parse_args()"""

    model_path = "results/eurosat/resnet18_eurosat.ckpt"
    image_path = "data/eurosat/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/AnnualCrop/AnnualCrop_1.tif"
    LABELS = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ]

    image_path = os.path.join(PROJECT_DIR, image_path)
    model_path = os.path.join(PROJECT_DIR, model_path)

    # show the original image
    img, meta = read_tif(image_path)
    img = quant_norm_data(img)

    from torchgeo.datamodules import EuroSATDataModule, UCMercedDataModule

    # root = '/home/jonasklotz/Studys/MASTERS/XAI_PLAYGROUND/data/eurosat'
    # Load the EuroSAT dataset
    # datamodule = EuroSATDataModule(root=root, batch_size=64, num_workers=4, download=True)

    root = "/home/jonasklotz/Studys/MASTERS/XAI_PLAYGROUND/data/ucmerced"
    # Load the EuroSAT dataset 21 classes
    datamodule = UCMercedDataModule(
        root=root, batch_size=64, num_workers=4, download=True
    )
    input_channels = 3

    # create model

    # resnet = LightningResnet.load_from_checkpoint(model_path,
    #                                               input_channels=input_channels,
    #                                               map_location=torch.device('cpu'))
    resnet = LightningResnet(input_channels=input_channels, num_classes=21)

    # predict
    img, label, pred, predicted_class = predict_from_datamodule(
        resnet, datamodule, labels=LABELS, rgb=(0, 1, 2)
    )
    output_probs = F.softmax(pred, dim=1).squeeze(0)
