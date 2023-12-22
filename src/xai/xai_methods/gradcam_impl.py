from typing import Union

import torch
from captum.attr import GuidedGradCam

from src.xai.xai_methods.explanation import Explanation


class GradCamImpl(Explanation):
    attribution_name = "GradCam"

    def __init__(self, model, layer=None, **kwargs):
        super().__init__(model, **kwargs)

        if not layer:
            layer = model.model.layer4[1].conv2

        self.attributor = GuidedGradCam(model, layer)

    def explain(
        self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None
    ):
        attrs = self.attributor.attribute(image_tensor, target=target)
        return attrs
