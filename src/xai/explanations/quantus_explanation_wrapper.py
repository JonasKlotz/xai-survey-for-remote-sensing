import numpy as np
import torch

from utility.cluster_logging import logger
from xai.explanations.explanation_manager import _explanation_methods


def explanation_wrapper(model, inputs, targets, **explain_func_kwargs):
    """
    Wrapper for explanation methods.

    The main purpose of this wrapper is to adapt the explanation to the interface of the quantus framework.
    Also this handles some of the functionality for the interaction of quantus

    Parameters
    ----------
    model: torch.nn.Module
        The model to explain.
    inputs: torch.Tensor
        Batch of images to explain.
    targets: torch.Tensor
        Targets of the batch.
    explain_func_kwargs: dict
        Keyword arguments for the explanation method.

    Returns
    -------
    attrs: torch.Tensor
        The attributions of the explanation method.
    """
    """
    Batch 2,3,h,w
        - RIS, ROS
    """
    # if numpy array convert to tensor
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(targets, list):
        # in case of MLC we have a list of tensors, as the labels are not one hot encoded
        targets = [torch.tensor(target) for target in targets]

    model = model.float()

    # mandatory kwargs
    explanation_method_name = explain_func_kwargs.pop("explanation_method_name")
    device = explain_func_kwargs.pop("device")

    explanation_method = _explanation_methods[explanation_method_name]
    explanation = explanation_method(model, device=device, **explain_func_kwargs)

    attributions = explanation.explain_batch(
        tensor_batch=inputs, target_batch=targets
    ).numpy(force=True)
    if np.all((attributions == 0)):
        logger.warning(f"All zero for method {explanation_method_name}")

    return attributions
