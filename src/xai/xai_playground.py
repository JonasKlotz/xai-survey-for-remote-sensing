import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import (
    IntegratedGradients,
    DeepLift,
    Saliency,
    NoiseTunnel,
    GradientShap,
    Occlusion,
)
import os

DATA_PATH = os.path.join(os.getcwd(), "..", "..", "data")


def imshow(img, transpose=True, title=""):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def get_cifar_loaders(root=DATA_PATH, batch_size=32, download=False, num_workers=2):
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=download, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=download, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainloader, testloader, classes, transform, transform_normalize


def get_voc_loaders(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.VOCSegmentation(
        root="./data",
        year="2012",
        image_set="train",
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.VOCSegmentation(
        root="./data", year="2012", image_set="val", download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    return trainloader, testloader, classes


def get_sample(test_loader, sample_index=0):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    image = images[sample_index].unsqueeze(0)
    label = labels[sample_index]
    image.requires_grad = True
    return image, label


def visualize_ground_truth_and_predictions(model, test_loader, classes):
    model.eval()
    # get images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))
    # predict
    outputs = model(images)
    predicted = F.softmax(outputs, dim=1)
    prediction_score, pred_label_idx = torch.topk(predicted, 1)
    print("Predicted: ", " ".join("%5s" % classes[pred_label_idx[j]] for j in range(4)))


def attribute_image_features(algorithm, input_image, model, label, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input_image, target=label, **kwargs)

    return tensor_attributions


if __name__ == "__main__":

    # get model
    from src.models.lightningresnet import LightningResnet

    # get cifar10 data
    (
        trainloader,
        testloader,
        classes,
        transform,
        transform_normalize,
    ) = get_cifar_loaders(root=DATA_PATH, batch_size=4, download=True, num_workers=2)

    model = LightningResnet(input_channels=3, num_classes=len(classes))
    model = model.eval()

    # visualize ground truth and predictions
    visualize_ground_truth_and_predictions(model, testloader, classes)

    # get sample
    image, label = get_sample(testloader)

    # predict

    output = model(image)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    label_name = classes[pred_label_idx]

    # plot image with label
    imshow(image.squeeze(0).detach(), transpose=False, title=classes[label])

    integrated_gradients = IntegratedGradients(model)
    attribution = integrated_gradients.attribute(
        image, target=pred_label_idx, n_steps=2
    )

    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )

    _ = viz.visualize_image_attr_multiple(
        np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "absolute_value"],
        cmap=default_cmap,
        show_colorbar=True,
    )

    # not enough memory
    # noise_tunnel = NoiseTunnel(integrated_gradients)
    #
    # attributions_ig_nt = noise_tunnel.attribute(image, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
    # _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    #                                       np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    #                                       ["original_image", "heat_map"],
    #                                       ["all", "positive"],
    #                                       cmap=default_cmap,
    #                                       show_colorbar=True)

    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([image * 0, image * 1])

    attributions_gs = gradient_shap.attribute(
        image,
        n_samples=50,
        stdevs=0.0001,
        baselines=rand_img_dist,
        target=pred_label_idx,
    )
    _ = viz.visualize_image_attr_multiple(
        np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "absolute_value"],
        cmap=default_cmap,
        show_colorbar=True,
    )

    saliency = Saliency(model)
    grads = saliency.attribute(image, target=label)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    _ = viz.visualize_image_attr_multiple(
        grads,
        np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        show_colorbar=True,
        outlier_perc=2,
    )

    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(
        image,
        strides=(3, 8, 8),
        target=pred_label_idx,
        sliding_window_shapes=(3, 15, 15),
        baselines=0,
    )

    _ = viz.visualize_image_attr_multiple(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        show_colorbar=True,
        outlier_perc=2,
    )

    print("Others")
    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(
        ig,
        image,
        model,
        label=label,
        baselines=image * 0,
        return_convergence_delta=True,
    )
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print("Approximation delta: ", abs(delta))

    dl = DeepLift(model)
    attr_dl, dl_delta = attribute_image_features(
        dl,
        image,
        model,
        label=label,
        baselines=image * 0,
        return_convergence_delta=True,
    )
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    _ = viz.visualize_image_attr_multiple(
        attr_dl,
        np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        show_colorbar=True,
        outlier_perc=2,
    )
