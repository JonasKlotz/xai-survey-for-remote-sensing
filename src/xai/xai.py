from captum.attr import IntegratedGradients, Saliency, DeepLift, NoiseTunnel
from captum.attr import visualization as viz
import numpy as np
import torch


class captum_explainer:
    def __init__(self, algorithm, **kwargs):
        self.algorithm = algorithm
        self.algorithm_name = algorithm.__name__
        self.kwargs = kwargs


def attribute_image_features(algorithm, input_tensor, model, labels, ind, **kwargs):
    """Generates attributions for the given image and model.

    Args:
        algorithm: The algorithm to be used for attribution.
        input_tensor (torch.Tensor): The input tensor for which the attributions
            are to be computed.
        model (torch.nn.Module): The model for which the attributions are to be
            computed.
        labels (list): The list of labels for the model.
        ind (int): The index of the image to be explained.

    Returns:
        (torch.Tensor, torch.Tensor): The attribution map and the corresponding
            output from the model.
    """
    model.zero_grad()
    tensor_attributions = algorithm.attribute(
        input_tensor, target=labels[ind], **kwargs
    )
    return tensor_attributions


def calculate_saliency(model, input_tensor, target):
    saliency = Saliency(model)
    grads = saliency.attribute(input_tensor, target=target)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return grads


def calculate_integrated_gradients(model, input_tensor):
    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(
        ig, input_tensor, baselines=input_tensor * 0, return_convergence_delta=True
    )
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print("Approximation delta: ", abs(delta))
    return attr_ig


def calculate_integrated_gradients_with_noise_tunnel(model, input_tensor):
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(
        nt,
        input_tensor,
        baselines=input_tensor * 0,
        nt_type="smoothgrad_sq",
        nt_samples=100,
        stdevs=0.2,
    )
    # if cuda is not available, move the tensor to cpu
    if torch.cuda.is_available():
        attr_ig_nt = np.transpose(
            attr_ig_nt.squeeze(0).gpu().detach().numpy(), (1, 2, 0)
        )
    else:
        attr_ig_nt = np.transpose(
            attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0)
        )
    return attr_ig_nt


def calculate_deeplift(model, input_tensor):
    dl = DeepLift(model)
    attr_dl = attribute_image_features(dl, input_tensor, model, baselines=input * 0)
    if torch.cuda.is_available():
        attr_ig_nt = np.transpose(attr_dl.squeeze(0).gpu().detach().numpy(), (1, 2, 0))
    else:
        attr_ig_nt = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    return attr_ig_nt


def visualize_image_attributions(attr_dl, original_image, title=""):
    """Visualizes the image attributions.

    Args:
        attr_dl (torch.Tensor): The attribution map.
        original_image (torch.Tensor): The original image.
        title (str): The title of the visualization.
    """
    _ = viz.visualize_image_attr(
        attr_dl,
        original_image,
        method="blended_heat_map",
        sign="all",
        show_colorbar=True,
        title=title,
    )
