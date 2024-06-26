import numpy as np
import torch

from data.data_utils import parse_batch
from data.zarr_handler import ZarrGroupHandler

from src.xai.explanations.explanation_methods.gradcam_impl import GradCamImpl
from src.xai.explanations.explanation_methods.ig_impl import IntegratedGradientsImpl
from src.xai.explanations.explanation_methods.lime_impl import LimeImpl
from src.xai.explanations.explanation_methods.lrp_impl import LRPImpl
from xai.explanations.explanation_methods.deeplift_impl import DeepLiftImpl
from xai.explanations.explanation_methods.guided_gradcam_impl import GuidedGradCamImpl
from xai.explanations.explanation_methods.occlusion_impl import OcclusionImpl

_explanation_methods = {
    "gradcam": GradCamImpl,
    "integrated_gradients": IntegratedGradientsImpl,
    "lime": LimeImpl,
    "lrp": LRPImpl,
    "deeplift": DeepLiftImpl,
    "guided_gradcam": GuidedGradCamImpl,
    "occlusion": OcclusionImpl,
}


class ExplanationsManager:
    def __init__(
        self,
        cfg: dict,
        model: torch.nn.Module,
        save=True,
        dtype=torch.float32,
    ):
        self.cfg = cfg
        self.save = save
        self.output_path = f"{self.cfg['results_path']}/{cfg["dataset_name"]}_metrics/{self.cfg['experiment_name']}"
        self.explanations = {}
        self.explanations_zarr_handler = {}

        self.device = cfg["device"]
        self.dtype = dtype

        self.model = model
        self.model.eval()
        self.model.to(self.device, dtype=dtype)

        self.threshold: float = cfg["threshold"]
        self.device = cfg["device"]
        self.dtype = dtype
        self.task = cfg["task"]
        self.has_segmentation = cfg["dataset_name"] != "ben"

        self._init_explanations()

    def _init_explanations(self):
        """
        Initialize the explanation methods.
        """

        for explanation_name in self.cfg["explanation_methods"]:
            self.explanations[explanation_name] = _explanation_methods[
                explanation_name
            ](
                model=self.model,
                device=self.device,
                num_classes=self.cfg["num_classes"],
                multi_label=self.task == "multilabel",
            )

        if not self.save:
            self.storage_handler = None
            return

        explanation_keys = [name + "_data" for name in list(self.explanations.keys())]
        storage_keys = [
            "x_data",
            "y_data",
            "y_pred_data",
            "index_data",
        ] + explanation_keys
        if self.has_segmentation:
            storage_keys.append("s_data")

        zarr_storage_path = f"{self.output_path}/{self.cfg['experiment_name']}.zarr"
        self.storage_handler = ZarrGroupHandler(
            path=zarr_storage_path,
            keys=storage_keys,
        )

    def explain_batch(
        self, batch: dict, explain_all: bool = False, explain_true_labels: bool = False
    ):
        """
        Explain a batch of images.
        Parameters
        ----------
        explain_true_labels: bool
            If True, the explanation will be generated for the true labels.
        explain_all: bool
            If True, the explanation will be generated for all labels.

        batch: torch.Tensor
            The batch to explain.

        Returns
        -------

        """
        (
            features,
            target,
            _,
            segments,
            idx,
            _,
        ) = parse_batch(batch)
        features, target = self._to_tensor(features), self._to_tensor(target)

        features = features.to(self.device, dtype=self.dtype)

        # ensure that the model and the input are on the same device
        predictions, logits = self.model.prediction_step(features)

        # if we want to generate an explanation for every label
        if explain_true_labels:
            explanation_targets = target
        elif explain_all:
            explanation_targets = torch.ones(target.shape, device=target.device)
        else:
            explanation_targets = predictions

        tmp_storage_dict = {
            "x_data": features,
            "y_data": target,
            "y_pred_data": predictions,
            "index_data": idx,
        }
        if segments is not None:
            tmp_storage_dict["s_data"] = segments

        # Explain batch for each explanation method
        for explanation_name, explanation in self.explanations.items():
            batch_attrs = explanation.explain_batch(features, explanation_targets)

            # save it to zarr
            tmp_storage_dict["a_" + explanation_name + "_data"] = batch_attrs

        if self.storage_handler is not None:
            self.storage_handler.append(tmp_storage_dict)

        return tmp_storage_dict

    def _to_tensor(self, input_tensor):
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.tensor(input_tensor)
        input_tensor = input_tensor.to(self.device)
        return input_tensor
