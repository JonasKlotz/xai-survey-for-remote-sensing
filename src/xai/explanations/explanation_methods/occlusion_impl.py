from typing import Union

import torch
from captum.attr import Occlusion

from xai.explanations.explanation_methods.explanation import Explanation


class OcclusionImpl(Explanation):
    attribution_name = "occlusion"

    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)

        self.attributor = Occlusion(model)
        self.sliding_window_shapes = (3, 50, 50)
        self.strides = (3, 10, 10)
        self.baselines = 0

    def explain(
        self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None
    ):
        attrs = self.attributor.attribute(
            image_tensor,
            target=target,
            sliding_window_shapes=self.sliding_window_shapes,
            strides=self.strides,
            baselines=self.baselines,
        )
        return attrs
