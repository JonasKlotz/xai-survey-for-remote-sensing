from typing import Union

import torch
from captum.attr import IntegratedGradients

from src.xai.explanations.explanation_methods.explanation import Explanation


class IntegratedGradientsImpl(Explanation):
    attribution_name = "integrated_gradients"

    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)

        self.attributor = IntegratedGradients(model)

    def explain(
        self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None
    ):
        attrs = self.attributor.attribute(image_tensor, target=target)
        return attrs
