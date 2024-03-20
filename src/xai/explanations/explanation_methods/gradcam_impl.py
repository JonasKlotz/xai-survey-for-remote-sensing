from typing import Union

import torch
from captum.attr import GuidedGradCam, LayerGradCam

from src.xai.explanations.explanation_methods.explanation import Explanation


class GradCamImpl(Explanation):
    attribution_name = "gradcam"

    def __init__(
        self,
        model,
        device,
        layer=None,
        guided=True,
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        if layer is None:
            layer = self.get_layer(layer, model)
        if guided:
            self.attributor = GuidedGradCam(model, layer)
        else:
            self.attributor = LayerGradCam(model, layer)

    def get_layer(self, layer, model):
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
        return layer

    def explain(
        self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None
    ):
        attrs = self.attributor.attribute(image_tensor, target=target)
        return attrs
