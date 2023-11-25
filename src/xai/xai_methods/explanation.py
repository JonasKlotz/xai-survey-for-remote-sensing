from abc import abstractmethod
from typing import Union

import numpy as np
import torch
from captum.attr import visualization as viz


class Explanation:
    attribution_name = "Explanation"

    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.explain(*args, **kwargs)

    @abstractmethod
    def explain(self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor]):
        """ Explain a single image

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
        pass

    def explain_batch(self, tensor_batch: torch.Tensor, target_batch: Union[int, torch.Tensor] = None):
        #todo vectorize
        # create output tensor without channels batchsize x 1 x height x width
        all_attrs = torch.zeros_like(tensor_batch[:, 0:1, :, :])
        for i in range(tensor_batch.shape[0]):
            image_tensor = tensor_batch[i:i+1]
            target = target_batch[i]
            attrs = self.explain(image_tensor, target)
            all_attrs[i] = attrs

        return all_attrs

    def visualize(self, attrs: torch.Tensor, image_tensor: torch.Tensor):
        if len(attrs.shape) == 4:
            attrs = attrs.squeeze(0)
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.squeeze(0)
        heatmap = np.transpose(attrs.cpu().detach().numpy(), (1, 2, 0))
        image = np.transpose(image_tensor.cpu().detach().numpy(), (1, 2, 0))

        # min max normalization
        #heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        image = (image - image.min()) / (image.max() - image.min())

        _ = viz.visualize_image_attr_multiple(
            attr=heatmap,
            original_image=image,
            methods = ["original_image", "heat_map"],
            signs = ["all", "positive"],
            titles=[f"Original Image for {self.attribution_name} on {self.model.__class__.__name__}",
                    f"Heat Map for {self.attribution_name} on {self.model.__class__.__name__}"],
            show_colorbar=True,
            outlier_perc=2,
        )


    def visualize_batch(self, attrs_batch, image_batch):
        for i in range(attrs_batch.shape[0]):
            attrs = attrs_batch[i]
            image = image_batch[i]
            self.visualize(attrs, image)


