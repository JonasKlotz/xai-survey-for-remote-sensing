import torch

from data.data_utils import _parse_segments
from src.xai.explanations.explanation_manager import _explanation_methods


# todo a problem here is how to get th explanation while training?
# the problem is that generating the explanation might mess up the regula loss calulation,e.g. forwards pass in the explanation
# messing up gradients


class RightForRightReasonsLoss(torch.nn.Module):
    def __init__(
        self,
        task: str = "multiclass",
        num_classes: int = None,
        lambda_=1,
        dataset_name: str = "unknown",
        explanation_method_name: str = "deeplift",
        explanation_kwargs: dict = None,
    ):
        """

        The right for the right reasons loss is defined as:

        ::math::
        \mathcal{L}_{rrr} (f_{\theta^t}(x_i), y_i) = \mathcal{L}_{pred}(f_{\theta^t}(x_i), y_i) + \lambda \mathcal{L}_{reason} (r_i^{l,t}, a_i^l)

        Parameters
        ----------
        loss: torch.nn.Module
            The other loss function to use
        model: torch.nn.Module
            The model to explain
        lambda_: float
            The weight of the explanation loss
        """
        super().__init__()
        self.explanation_method_name = explanation_method_name
        self.explanation_method_constructor = _explanation_methods[
            explanation_method_name
        ]
        if explanation_kwargs is None:
            explanation_kwargs = {}
        self.explanation_kwargs = (
            explanation_kwargs if explanation_kwargs is not None else {}
        )

        #
        self.lambda_ = lambda_

        self.task = task
        self.num_classes = num_classes
        self.dataset_name = dataset_name

    def forward(
        self,
        model: torch.nn.Module,
        x_batch: torch.Tensor,
        y_pred_batch: torch.Tensor,
        s_batch: torch.Tensor,
    ):
        """

        Parameters
        ----------
        x_batch : torch.Tensor
            The input batch
        y_batch: torch.Tensor
            The target batch
        y_pred_batch: torch.Tensor
            The prediction batch, values of the labels, e.g. (2, 5, 13) for MLC
        s_batch: torch.Tensor
            The relevancy map, this map is 1 where the explanation should be 0
        Returns
        -------

        """
        # initialize the explanation method
        explanation_method = self.explanation_method_constructor(
            model=model,
            device=x_batch.device,
            multi_label=self.task == "multilabel",
            **self.explanation_kwargs,
        )

        attrs = explanation_method.explain_batch(
            tensor_batch=x_batch, target_batch=y_pred_batch
        )

        s_batch = segmentations_to_relevancy_map(
            s_batch, num_classes=self.num_classes, dataset_name=self.dataset_name
        )
        # push to device
        s_batch = s_batch.to(x_batch.device)
        rrr_loss = torch.linalg.norm(attrs * s_batch)
        return self.lambda_ * rrr_loss


def segmentations_to_relevancy_map(
    s_batch, num_classes: int = None, dataset_name="unknown"
):
    """
    Convert a batch of segmentations to a relevancy map

    Parameters
    ----------
    s_batch: torch.Tensor
        The batch of segmentations

    Returns
    -------
    relevancy_map: torch.Tensor
        The relevancy map
    """

    s_batch = _parse_segments(s_batch, dataset_name, num_classes)

    """
        flip the segmentation batch, every element that is 1 should be 0 and vice versa
        
        Segmentation  Relevancy Map   Attribution   
        0 0 0 1 1      1 1 1 0 0      1 1 1 0 0  
        0 0 0 1 1  ->  1 1 1 0 0  +   1 1 1 0 0  -> Is the worst possible explanation and should result in a high loss
        0 0 0 0 0      1 1 1 1 1      1 1 1 1 1
    """
    s_batch = 1 - s_batch
    return s_batch
