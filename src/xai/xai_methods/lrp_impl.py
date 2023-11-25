from typing import Union

import torch
from captum.attr import LRP
from captum.attr._utils.lrp_rules import (
    EpsilonRule,
    GammaRule,
    IdentityRule,
    Alpha1_Beta0_Rule,
)

from src.xai.xai_methods.explanation import Explanation


class LRPImpl(Explanation):
    attribution_name = "LRP"

    def __init__(self, model):
        super().__init__(model)
        self._rule_resnet(model)

        self.attributor = LRP(model)

    def explain(
        self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None
    ):
        attrs = self.attributor.attribute(image_tensor, target=target)
        return attrs

    def _rule_resnet(self, model):
        num_layers = len(model._modules.keys())

        for i, layer in enumerate(model._modules.keys()):
            if i < num_layers / 3:
                model._modules[layer].lrp_rule = GammaRule()
            elif i < 2 * num_layers / 3:
                model._modules[layer].lrp_rule = EpsilonRule()
            else:
                model._modules[layer].lrp_rule = EpsilonRule(0)
