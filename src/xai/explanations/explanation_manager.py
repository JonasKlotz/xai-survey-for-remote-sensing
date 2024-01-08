import numpy as np
import torch

from data.zarr_handler import ZarrGroupHandler
from xai.explanations.explanation_methods.deeplift_impl import DeepLiftImpl
from src.xai.explanations.explanation_methods.gradcam_impl import GradCamImpl
from src.xai.explanations.explanation_methods.ig_impl import IntegratedGradientsImpl
from src.xai.explanations.explanation_methods.lime_impl import LimeImpl
from src.xai.explanations.explanation_methods.lrp_impl import LRPImpl
from utility.cluster_logging import logger

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

        self.visualize = cfg["visualize"]
        self.threshold: float = cfg["threshold"]
        self.vis_path = f"{self.explanations_config['results_path']}/visualizations_{self.explanations_config['timestamp']}"

        self._init_explanations()

    def _init_explanations(self):
        """
        Initialize the explanation methods.
        """

        folder_name = f"batches_{self.explanations_config['timestamp']}"

        for explanation_name in self.explanations_config["explanation_methods"]:
            self._init_explanation(explanation_name, folder_name)

        explanation_keys = [name + "_data" for name in list(self.explanations.keys())]
        storage_keys = [
            "x_data",
            "y_data",
            "y_pred_data",
            "s_data",
            "index_data",
        ] + explanation_keys
        zarr_storage_path = f"{self.explanations_config['results_path']}/explanations_{self.explanations_config['timestamp']}.zarr"
        self.storage_handler = ZarrGroupHandler(
            path=zarr_storage_path,
            keys=storage_keys,
        )

    def _init_explanation(self, explanation_name: str, folder_name: str):
        self.explanations[explanation_name] = _explanation_methods[explanation_name](
            model=self.model,
            vectorize=self.explanations_config["vectorize"],
        )
        # self.explanations_zarr_handler[explanation_name] = ZarrHandler(
        #     results_dir=self.explanations_config["results_path"],
        #     name=f"a_batch_{explanation_name}",
        #     folder_name=folder_name,
        # )

    def explain_batch(
        self,
        batch: torch.Tensor,
    ):
        """
        Explain a batch of images.
        Parameters
        ----------
        batch: torch.Tensor
            The batch to explain.

        Returns
        -------

        """
        logits, _ = self.model.predict(batch)
        prediction_batch = (logits > self.threshold).long()

        image_batch, target_batch, idx, segments_batch = batch

        tmp_storage_dict = {
            "x_data": image_batch,
            "y_data": target_batch,
            "y_pred_data": prediction_batch,
            "s_data": segments_batch,
            "index_data": idx,
        }

        # Explain batch for each explanation method
        for explanation_name, explanation in self.explanations.items():
            batch_attrs = explanation.explain_batch(
                image_batch, prediction_batch, target_batch
            )
            if self.visualize:
                self._visualize(
                    batch_attrs, explanation, explanation_name, idx, image_batch
                )
            # save it to zarr
            tmp_storage_dict["a_" + explanation_name + "_data"] = batch_attrs

        self.storage_handler.append(tmp_storage_dict)

        return tmp_storage_dict

    def _visualize(self, batch_attrs, explanation, explanation_name, idx, image_batch):
        # todo this is only for mlc and only prints the first attr map regarding whether its all 0
        fig = explanation.visualize(batch_attrs[0][0], image_batch[0])
        save_path = f"{self.vis_path}/{idx}_{explanation_name}.png"
        logger.info(f"Saving visualization to {save_path}")
        fig.savefig(save_path)


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
    # todo differentiate between batch and single
    attrs = (
        explanation.explain_batch(tensor_batch=inputs, prediction_batch=targets)
        .detach()
        .numpy()
    )
    return attrs
