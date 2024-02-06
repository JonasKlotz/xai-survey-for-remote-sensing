import numpy as np
import torch

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
    model = model.float()

    # mandatory kwargs
    explanation_method_name = explain_func_kwargs.pop("explanation_method_name")
    device = explain_func_kwargs.pop("device")
    # multi_label = explain_func_kwargs.pop("multi_label")

    explanation_method = _explanation_methods[explanation_method_name]
    explanation = explanation_method(model, device=device, **explain_func_kwargs)

    attributions = explanation.explain_batch(
        tensor_batch=inputs, target_batch=targets
    ).numpy(force=True)
    return attributions