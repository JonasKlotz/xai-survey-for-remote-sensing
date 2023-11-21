from datetime import datetime

import torch

from data.zarr_handler import ZarrHandler
from src.xai.xai_methods.deeplift_impl import DeepLiftImpl
from src.xai.xai_methods.gradcam_impl import GradCamImpl
from src.xai.xai_methods.ig_impl import IntegratedGradientsImpl
from src.xai.xai_methods.lime_impl import LimeImpl
from src.xai.xai_methods.lrp_impl import LRPImpl

_explanation_methods = {
    "gradcam": GradCamImpl,
    "integrated_gradients": IntegratedGradientsImpl,
    "lime": LimeImpl,
    "lrp": LRPImpl,
    "deeplift": DeepLiftImpl
}


class ExplanationsManager:

    def __init__(self, explanations_config: dict,
                 model: torch.nn.Module,
                 datetime_appendix: bool = True):
        self.explanations_config = explanations_config
        self.explanations = {}
        self.explanations_zarr_handler = {}
        self.model = model
        self.visualize = explanations_config['visualize']

        if datetime_appendix:
            self.datetime_appendix = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._init_explanations()

    def _init_explanations(self):
        for explanation_name in self.explanations_config['explanation_methods']:
            self._init_explanation(explanation_name)

    def _init_explanation(self, explanation_name: str):
        self.explanations[explanation_name] = _explanation_methods[explanation_name](model=self.model)
        self.explanations_zarr_handler[explanation_name] = (
            ZarrHandler(name=f"a_batch_{explanation_name}",
                        folder_name=f"batches_{self.datetime_appendix}")
        )
        self.x_batch_handler = ZarrHandler(name="x_batch",
                                           folder_name=f"batches_{self.datetime_appendix}")
        self.y_batch_handler = ZarrHandler(name="y_batch",
                                             folder_name=f"batches_{self.datetime_appendix}")

    def explain_batch(self, image_batch: torch.Tensor, target_batch: torch.Tensor):
        """Explain a batch of images and save it to zarr

        :param image_batch: batch of images
        :param target_batch: batch of targets
        :return: None
        """
        self.x_batch_handler.append(image_batch)
        self.y_batch_handler.append(target_batch)
        for explanation_name, explanation in self.explanations.items():
            batch_attrs = explanation.explain_batch(image_batch, target_batch)
            if self.visualize:
                explanation.visualize(batch_attrs[0], image_batch[0])
            # save it to zarr
            self.explanations_zarr_handler[explanation_name].append(batch_attrs)
