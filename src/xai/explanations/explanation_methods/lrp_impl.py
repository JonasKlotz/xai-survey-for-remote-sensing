import copy
from typing import Union

import torch
from captum.attr import LRP
from captum.attr._utils.lrp_rules import (
    EpsilonRule,
    GammaRule,
)

from src.xai.explanations.explanation_methods.explanation import Explanation


class LRPImpl(Explanation):
    attribution_name = "lrp"

    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)
        self.layers = []
        model_copy = copy.deepcopy(model)

        self._get_layers(model_copy)
        self._set_model_rules(model_copy)

        self.attributor = LRP(model_copy)

    def explain(
        self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None
    ):
        attrs = self.attributor.attribute(image_tensor, target=target)
        return attrs

    def _set_model_rules(self, model):
        num_layers = len(model._modules.keys())

        for i, layer in enumerate(model._modules.keys()):
            if i < num_layers / 3:
                model._modules[layer].rule = GammaRule()
            elif i < 2 * num_layers / 3:
                model._modules[layer].rule = EpsilonRule()
            else:
                model._modules[layer].rule = EpsilonRule(0)

    def _get_layers(self, model) -> None:
        for layer in model.children():
            if len(list(layer.children())) == 0:
                self.layers.append(layer)
            else:
                self._get_layers(layer)
