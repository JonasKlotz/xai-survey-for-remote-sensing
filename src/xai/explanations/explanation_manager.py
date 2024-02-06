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
        save=True,
    ):
        self.explanations_config = cfg
        self.save = save
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
                device=self.device,
                num_classes=self.explanations_config["num_classes"],
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
        features = batch["features"].to(self.device)
        target = batch["targets"].to(self.device)
        idx = batch["index"].to(self.device)
        segments = batch.get("segmentations", torch.tensor([])).to(self.device)

        predictions = self.model.prediction_step(features)

        tmp_storage_dict = {
            "x_data": features,
            "y_data": target,
            "y_pred_data": predictions,
            "s_data": segments,
            "index_data": idx,
        }

        # Explain batch for each explanation method
        for explanation_name, explanation in self.explanations.items():
            # todo dimensions wrong????
            batch_attrs = explanation.explain_batch(features, predictions)

            # save it to zarr
            tmp_storage_dict["a_" + explanation_name + "_data"] = batch_attrs

        if self.storage_handler is not None:
            self.storage_handler.append(tmp_storage_dict)

        return tmp_storage_dict
