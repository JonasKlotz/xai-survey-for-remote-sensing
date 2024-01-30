from typing import Union

import matplotlib.pyplot as plt
import torch
from captum._utils.models import SkLearnLinearRegression
from captum.attr._core.lime import get_exp_kernel_similarity_function, Lime
from skimage.segmentation import mark_boundaries, slic

from src.xai.explanations.explanation_methods.explanation import Explanation


class LimeImpl(Explanation):
    attribution_name = "LIME"

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

        self.similarity_func = get_exp_kernel_similarity_function(
            "euclidean", kernel_width=500
        )
        self.interpretable_model = (
            SkLearnLinearRegression()
        )  # SkLearnLasso(alpha=0.0)  # SkLearnLinearRegression()

        self.attributor = Lime(
            model,
            interpretable_model=self.interpretable_model,  # build-in wrapped sklearn Linear Regression
            similarity_func=self.similarity_func,
        )

    def explain(
        self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None
    ):
        """Explain a single image

        Parameters
        ----------
        image_tensor: torch.Tensor
            The image to explain as tensor of shape (1, channels, height, width)
        target: Union[int, torch.Tensor]
            The target to explain

        Returns
        -------
        attrs: torch.Tensor
            The attributions of the explanation method.

        """

        segments = slic_from_tensor(
            image_tensor,
            plot=False,
            n_segments=15,
            sigma=5,
            multi_label=self.multi_label,
        ).unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        target = target.to(self.device)
        segments = segments.to(self.device)

        attrs = self.attributor.attribute(
            image_tensor,
            target=target,
            feature_mask=segments,
        )
        return attrs


def _batchwise_slic(img_tensor_batch: torch.Tensor, plot=False, n_segments=50, sigma=5):
    """
    Apply SLIC algorithm to a batch of images.

    Parameters
    ----------
    img_tensor_batch : torch.Tensor
        A batch of images to segment as tensor of shape (batchsize, channels, height, width)
    plot : bool, optional
        Whether to plot the segmented images. Default is False.
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.
    sigma : float, optional
        Width of Gaussian smoothing kernel for pre-processing for segmentation.

    Returns
    -------
    segments_batch : torch.Tensor
        A batch of segmented images.
    """
    segments_batch = []
    for img_tensor in img_tensor_batch:
        segments = slic_from_tensor(
            img_tensor, plot=plot, n_segments=n_segments, sigma=sigma
        )
        segments_batch.append(segments)
    return torch.stack(segments_batch)


def slic_from_tensor(
    img_tensor: torch.Tensor,
    plot=True,
    save_path=None,
    title="",
    n_segments=50,
    sigma=5,
    multi_label=True,
):
    """Superpixel segmentation using SLIC algorithm"""

    if len(img_tensor.shape) == 4:
        img_tensor = img_tensor.squeeze(0)

    img = img_tensor.permute(1, 2, 0).cpu().to(torch.double).numpy()

    segments = slic(img, start_label=0, n_segments=n_segments, sigma=sigma)

    segments = torch.from_numpy(segments)

    return segments


def _plot_slic(img, n_segments, save_path, segments, title):
    # show the output of SLIC
    fig = plt.figure(f"{title} Superpixels -- {n_segments} segments")
    ax = fig.add_subplot(1, 1, 1)
    # convert tensor to numpy
    ax.imshow(mark_boundaries(img, segments))
    plt.axis("off")
    # show the plots
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
