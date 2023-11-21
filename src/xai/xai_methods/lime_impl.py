from typing import Union

import matplotlib.pyplot as plt
import torch
from captum._utils.models import SkLearnLinearRegression
from captum.attr._core.lime import get_exp_kernel_similarity_function, Lime
from skimage.segmentation import mark_boundaries, slic

from src.xai.xai_methods.explanation import Explanation


class LimeImpl(Explanation):
    attribution_name = "LIME"

    def __init__(self, model):
        super().__init__(model)

        self.similarity_func = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
        self.interpretabel_model = SkLearnLinearRegression()

        self.attributor = Lime(
            model,
            interpretable_model=self.interpretabel_model,  # build-in wrapped sklearn Linear Regression
            similarity_func=self.similarity_func
        )

    def explain(self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor] = None):
        segments = slic_from_tensor(image_tensor, plot=False, save_path=None, title="", n_segments=200,
                                    sigma=5)

        attrs = self.attributor.attribute(image_tensor,
                                          target=target,
                                          feature_mask=segments)
        return attrs


def slic_from_tensor(img_tensor, plot=True, save_path=None, title="", n_segments=200, sigma=5):
    """ Apply SLIC to a tensor

    """
    if len(img_tensor.shape) == 4:
        img_tensor = img_tensor.squeeze(0)
    img = img_tensor.permute(1, 2, 0).to(torch.double).numpy()

    segments = slic(img, start_label=0, n_segments=n_segments, sigma=sigma)

    if plot:
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

    segments = torch.from_numpy(segments)

    return segments
