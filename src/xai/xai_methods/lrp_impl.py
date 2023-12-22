from typing import Union

import torch
from captum.attr import LRP
from captum.attr._utils.lrp_rules import (
    EpsilonRule,
    GammaRule,
    IdentityRule,
)

from src.xai.xai_methods.explanation import Explanation


class LRPImpl(Explanation):
    attribution_name = "LRP"

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self._rule_resnet(model)

        self.attributor = LRP(model)
        self.layers = []

    def explain(
        self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None
    ):
        attrs = self.attributor.attribute(image_tensor, target=target)
        return attrs

    def _rule_resnet(self, model):
        num_layers = len(model._modules.keys())

        for i, layer in enumerate(model._modules.keys()):
            if i < num_layers / 3:
                model._modules[layer].lrprule = GammaRule()
            elif i < 2 * num_layers / 3:
                model._modules[layer].rule = EpsilonRule()
            else:
                model._modules[layer].rule = EpsilonRule(0)

    def _set_layers_rules(self, model) -> None:
        for layer in model.children():
            if len(list(layer.children())) == 0:
                layer.rule = IdentityRule()
            else:
                self._set_layers_rules(layer)
