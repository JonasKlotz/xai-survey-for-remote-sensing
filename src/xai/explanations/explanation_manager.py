import numpy as np
import torch

from data.zarr_handler import ZarrGroupHandler
from src.xai.explanations.explanation_methods.gradcam_impl import GradCamImpl
from src.xai.explanations.explanation_methods.ig_impl import IntegratedGradientsImpl
from src.xai.explanations.explanation_methods.lime_impl import LimeImpl
from src.xai.explanations.explanation_methods.lrp_impl import LRPImpl
from xai.explanations.explanation_methods.deeplift_impl import DeepLiftImpl

_explanation_methods = {
    "gradcam": GradCamImpl,
    "integrated_gradients": IntegratedGradientsImpl,
    "lime": LimeImpl,
    "lrp": LRPImpl,
    "deeplift": DeepLiftImpl,
}


class ExplanationsManager:
    def __init__(
        self,
        cfg: dict,
        model: torch.nn.Module,
    ):
        self.explanations_config = cfg
        self.explanations = {}
        self.explanations_zarr_handler = {}
        self.model = model
        self.model.eval()

        self.threshold: float = cfg["threshold"]
        self.device = cfg["device"]
        self.task = cfg["task"]

        self._init_explanations()

    def _init_explanations(self):
        """
        Initialize the explanation methods.
        """

        for explanation_name in self.explanations_config["explanation_methods"]:
            self.explanations[explanation_name] = _explanation_methods[
                explanation_name
            ](
                model=self.model,
                vectorize=self.explanations_config["vectorize"],
                device=self.device,
                num_classes=self.explanations_config["num_classes"],
                multi_label=self.task == "multilabel",
            )

        explanation_keys = [name + "_data" for name in list(self.explanations.keys())]
        storage_keys = [
            "x_data",
            "y_data",
            "y_pred_data",
            "s_data",
            "index_data",
        ] + explanation_keys
        zarr_storage_path = f"{self.explanations_config['results_path']}/{self.explanations_config['experiment_name']}.zarr"
        self.storage_handler = ZarrGroupHandler(
            path=zarr_storage_path,
            keys=storage_keys,
        )

    def explain_batch(self, batch: torch.Tensor):
        """
        Explain a batch of images.
        Parameters
        ----------
        batch: torch.Tensor
            The batch to explain.

        Returns
        -------

        """
        images = batch["features"].to(self.device)
        target = batch["targets"].to(self.device)
        idx = batch["index"].to(self.device)
        segments = batch.get("segments", torch.tensor([])).to(self.device)

        prediction_batch = self.model.prediction_step(images)

        tmp_storage_dict = {
            "x_data": images,
            "y_data": target,
            "y_pred_data": prediction_batch,
            "s_data": segments,
            "index_data": idx,
        }

        # Explain batch for each explanation method
        for explanation_name, explanation in self.explanations.items():
            batch_attrs = explanation.explain_batch(images, prediction_batch, target)

            # save it to zarr
            tmp_storage_dict["a_" + explanation_name + "_data"] = batch_attrs

        self.storage_handler.append(tmp_storage_dict)

        return tmp_storage_dict


def explanation_wrapper(model, inputs, targets, **explain_func_kwargs):
    """
    Wrapper for explanation methods.

    Parameters
    ----------
    model: torch.nn.Module
        The model to explain.
    inputs: torch.Tensor
        The input to explain.
    targets: torch.Tensor
        The target to explain.
    explain_func_kwargs: dict
        Keyword arguments for the explanation method.

    Returns
    -------
    attrs: torch.Tensor
        The attributions of the explanation method.
    """
    # if numpy array convert to tensor
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    model = model.float()

    explanation_method_name = explain_func_kwargs.pop("explanation_method_name")
    explanation_method = _explanation_methods[explanation_method_name]
    explanation = explanation_method(model)
    attrs = (
        explanation.explain_batch(tensor_batch=inputs, prediction_batch=targets)
        .detach()
        .numpy()
    )
    return attrs
