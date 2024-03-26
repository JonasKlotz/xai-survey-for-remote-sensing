from typing import Union

import torch
from captum.attr import DeepLift

from xai.explanations.explanation_methods.explanation import Explanation


class DeepLiftImpl(Explanation):
    attribution_name = "deeplift"

    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)

        self.attributor = DeepLift(model, multiply_by_inputs=False)

    def explain(
        self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None
    ):
        attrs = self.attributor.attribute(image_tensor, target=target)
        return attrs
