from typing import Union

import torch
from captum.attr import GuidedGradCam

from src.xai.explanations.explanation_methods.explanation import Explanation


class GradCamImpl(Explanation):
    attribution_name = "gradcam"

    def __init__(self, model, device, layer=None, **kwargs):
        super().__init__(model, device, **kwargs)
        from models.lightning_models import LightningResnet, LightningVGG

        if isinstance(model, LightningResnet):
            if model.hparams.resnet_layers == 18:
                layer = model.backbone.layer4[1].conv2
            elif model.hparams.resnet_layers == 34:
                layer = model.backbone.layer4[2].conv2
            elif model.hparams.resnet_layers == 50:
                layer = model.backbone.layer4[2].conv3
        elif isinstance(model, LightningVGG):
            layer = model.backbone.features[28]

        self.attributor = GuidedGradCam(model, layer)

    def explain(
        self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None
    ):
        attrs = self.attributor.attribute(image_tensor, target=target)
        return attrs
