import json
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
from xai.xai_methods.explanation_manager import ExplanationsManager


def load_test_imagenet_image( idx_to_labels, image_tensor):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    img = Image.open('/home/jonasklotz/Studys/MASTERS/XAI_PLAYGROUND/img/swan.jpeg')
    transformed_img = transform(img)
    image_tensor = transform_normalize(transformed_img)
    image_tensor = image_tensor.unsqueeze(0)


def generate_explanations(explanations_config:dict):
    # load datamodule
    data_module = load_data_module(explanations_config["dataset_name"])
    loaders = get_loader_for_datamodule(data_module)
    test_loader = loaders['test']

    # load model
    # todo: get model
    model = LightningResnet(num_classes=data_module.num_classes, input_channels=data_module.dims[0])
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



if __name__ == '__main__':
    # resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    # resnet = resnet.eval()

    data_module = get_mnist_data_module()
    loaders = get_loader_for_datamodule(data_module)
    test_loader = loaders['test']

    cifar_path = "/home/jonasklotz/Studys/MASTERS/XAI/models/resnet18_cifar.pt"

    resnet = LightningResnet(input_channels=1, num_classes=10)
    resnet.load_state_dict(torch.load(cifar_path))

    images, labels = next(iter(test_loader))

    output = resnet(images)
    output_probs = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output_probs, 1)
    print(f"Prediction: {prediction_score.squeeze()}, Label: {pred_label_idx.squeeze()}")

    pred_label_idx.squeeze_()


    # Lime = LimeImpl(resnet)
    # attrs = Lime.explain(image_tensor, target=pred_label_idx)
    # Lime.visualize(attrs, image_tensor)
    #
    # GradCam = GradCamImpl(resnet)
    # attrs = GradCam.explain(image_tensor, target=pred_label_idx)
    # GradCam.visualize(attrs, image_tensor)

    # LRP = LRPImpl(resnet)
    # attrs = LRP.explain(image_tensor, target=pred_label_idx)
    # LRP.visualize(attrs, image_tensor)

    # IG = IntegratedGradientsImpl(resnet)
    # attrs = IG.explain(image_tensor, target=pred_label_idx)
    # IG.visualize(attrs, image_tensor)



PATH_DATASETS = "/home/jonasklotz/Studys/MASTERS/XAI/data"
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

def create_model():
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet = resnet.eval()
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    resnet.maxpool = nn.Identity()
    return resnet





# if __name__ == '__main__':
#     train_transforms = torchvision.transforms.Compose(
#         [
#             torchvision.transforms.RandomCrop(32, padding=4),
#             torchvision.transforms.RandomHorizontalFlip(),
#             torchvision.transforms.ToTensor(),
#             cifar10_normalization(),
#         ]
#     )
#
#     test_transforms = torchvision.transforms.Compose(
#         [
#             torchvision.transforms.ToTensor(),
#             cifar10_normalization(),
#         ]
#     )
#
#     cifar10_dm = CIFAR10DataModule(
#         data_dir=PATH_DATASETS,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         train_transforms=train_transforms,
#         test_transforms=test_transforms,
#         val_transforms=test_transforms,
#     )
#
#     resnet = create_model()
#
#     cifar10_dm.prepare_data()
#     cifar10_dm.setup()
#     test_loader = cifar10_dm.test_dataloader()
#
#     IG = IntegratedGradientsImpl(resnet)
#
#     for batch in test_loader:
#         for image, label in zip(batch[0], batch[1]):
#             image = image.unsqueeze(0)
#             attrs = IG.explain(image, target=label)
#             IG.visualize(attrs, image)
#             break
#
#

