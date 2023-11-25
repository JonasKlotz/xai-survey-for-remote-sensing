from typing import Union

import torch
from captum.attr import GuidedGradCam

from src.xai.xai_methods.explanation import Explanation


class GradCamImpl(Explanation):
    attribution_name = "GradCam"

    def __init__(self, model, layer=None):
        super().__init__(model)
        if not layer:
            layer = model.model.layer4[1].conv2
        # todo layer customizen
        self.attributor = GuidedGradCam(model, layer)

    def explain(
        self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None
    ):
        attrs = self.attributor.attribute(image_tensor, target=target)
        return attrs
