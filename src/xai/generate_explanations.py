import yaml
import os

import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from data.data_utils import get_loader_for_datamodule
from models.lightningresnet import LightningResnet
from src.xai.xai_methods.gradcam_impl import GradCamImpl
from src.xai.xai_methods.ig_impl import IntegratedGradientsImpl
from src.xai.xai_methods.lime_impl import LimeImpl
from src.xai.xai_methods.lrp_impl import LRPImpl

# import datamodule
from src.data.get_data_modules import get_mnist_data_module, load_data_module
from src.xai.xai_methods.explanation_manager import ExplanationsManager


def load_test_imagenet_image(idx_to_labels, image_tensor):

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )
    img = Image.open("/home/jonasklotz/Studys/MASTERS/XAI_PLAYGROUND/img/swan.jpeg")
    transformed_img = transform(img)
    image_tensor = transform_normalize(transformed_img)
    image_tensor = image_tensor.unsqueeze(0)


def generate_explanations(explanations_config: dict):
    # load datamodule
    data_module = load_data_module(explanations_config["dataset_name"])
    loaders = get_loader_for_datamodule(data_module)
    test_loader = loaders["test"]

    # load model
    # todo: get model
    model = LightningResnet(
        num_classes=data_module.num_classes, input_channels=data_module.dims[0]
    )
    model_path = f"/home/jonasklotz/Studys/MASTERS/XAI/models/resnet18_{explanations_config['dataset_name']}.pt"
    model.load_state_dict(torch.load(model_path))

    images, labels = next(iter(test_loader))

    output = model(images)
    output_probs = F.softmax(output, dim=1)
    label_idx = output_probs.argmax(dim=1)

    explanation_manager = ExplanationsManager(explanations_config, model)
    explanation_manager.explain_batch(images, label_idx)

    # Lime = LimeImpl(model)
    # attrs_batch = Lime.explain_batch(images, target=label_idx)
    # lime_zarr = ZarrHandler(name='lime_attribution_batch')
    # lime_zarr.append(attrs_batch)
    # print(lime_zarr.shape)
    #
    # print(lime_zarr.shape)
    # # Lime.visualize_batch(attrs_batch, images)
    # Lime.visualize(attrs_batch[0], images[0])
    #
    #
    # layer = model.model.layer4[1].conv2
    # GradCam = GradCamImpl(model, layer)
    # attrs_batch = GradCam.explain(images, target=label_idx)
    # GradCam.visualize(attrs_batch[0], images[0])
    #
    # LRP = LRPImpl(model)
    # attrs_batch = LRP.explain(images, target=label_idx)
    # LRP.visualize(attrs_batch[0], images[0])
    #
    # IG = IntegratedGradientsImpl(model)
    # attrs_batch = IG.explain(images, target=label_idx)
    # IG.visualize(attrs_batch[0], images[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/jonasklotz/Studys/MASTERS/XAI/config/explanations_config.yml",
    )

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generate_explanations(config)
