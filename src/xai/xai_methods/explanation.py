from abc import abstractmethod
from typing import Union

import numpy as np
import torch
from captum.attr import visualization as viz


class Explanation:
    attribution_name = "Explanation"

    def __init__(self, model, multi_label: bool = True, num_classes: int = 6):
        self.model = model
        self.multi_label = multi_label
        self.num_classes = num_classes

    def __call__(self, *args, **kwargs):
        return self.explain(*args, **kwargs)

    @abstractmethod
    def explain(self, image_tensor: torch.Tensor, target: Union[int, torch.Tensor]):
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
        pass

    def explain_batch(
        self, tensor_batch: torch.Tensor, target_batch: torch.Tensor = None
    ):
        """Explain a batch of images

        Parameters
        ----------
        tensor_batch: torch.Tensor
            The batch of images to explain as tensor of shape (batchsize, channels, height, width)
        target_batch: torch.Tensor
            The batch of targets to explain

        Returns
        -------
        all_attrs: torch.Tensor
            The attributions of the explanation method for the whole batch.
        """

        # todo vectorize
        # create output tensor without channels batchsize x 1 x height x width
        if self.multi_label:
            return self._handle_mlc_explanation(tensor_batch, target_batch)

        return self._handle_slc_explanation(tensor_batch, target_batch)

    def visualize(self, attrs: torch.Tensor, image_tensor: torch.Tensor):
        if len(attrs.shape) == 4:
            attrs = attrs.squeeze(0)
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.squeeze(0)

        heatmap = np.transpose(attrs.cpu().detach().numpy(), (1, 2, 0))
        image = np.transpose(image_tensor.cpu().detach().numpy(), (1, 2, 0))

        # min max normalization
        # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        image = (image - image.min()) / (image.max() - image.min())

        fig = viz.visualize_image_attr_multiple(
            attr=heatmap,
            original_image=image,
            methods=["original_image", "heat_map"],
            signs=["all", "positive"],
            titles=[
                f"Original Image for {self.attribution_name} on {self.model.__class__.__name__}",
                f"Heat Map for {self.attribution_name} on {self.model.__class__.__name__}",
            ],
            show_colorbar=True,
            outlier_perc=2,
            use_pyplot=False,
        )
        return fig

    def visualize_batch(self, attrs_batch, image_batch):
        for i in range(attrs_batch.shape[0]):
            attrs = attrs_batch[i]
            image = image_batch[i]
            self.visualize(attrs, image)

    def _handle_mlc_explanation(
        self, tensor_batch: torch.Tensor, target_batch: Union[int, torch.Tensor] = None
    ):
        batchsize, _, height, width = tensor_batch.shape

        # create output tensor without channels (classes * batchsize) x 1 x height x width
        all_attrs = torch.zeros((batchsize, self.num_classes, 1, height, width))
        for batch_index in range(batchsize):
            image_tensor = tensor_batch[batch_index : batch_index + 1]
            # print(f"image_tensor shape: {image_tensor.shape}")
            # print(f"target_batch shape: {target_batch.shape}")
            for label_index in range(len(target_batch[batch_index])):
                target = target_batch[batch_index][label_index]
                # print(f"target: {target}")
                # we only want to explain the labels that are present in the image
                if target == 0:
                    continue

                attrs = self.explain(
                    image_tensor, torch.tensor(label_index).unsqueeze(0)
                )
                print(f"attrs shape: {attrs.shape}")
                # if attrs is 3 channel sum over channels
                # todo: be careful with this, channels might differ use asserts
                if len(attrs.shape) == 4 and attrs.shape[1] == 3:
                    attrs = attrs.sum(dim=1, keepdim=True)
                all_attrs[batch_index, label_index] = attrs
        return all_attrs

    def _handle_slc_explanation(
        self, tensor_batch: torch.Tensor, target_batch: Union[int, torch.Tensor] = None
    ):
        batchsize, channels, height, width = tensor_batch.shape

        all_attrs = torch.zeros((batchsize, 1, height, width))
        for i in range(batchsize):
            image_tensor = tensor_batch[i : i + 1]
            target = target_batch[i]
            attrs = self.explain(image_tensor, target)
            all_attrs[i] = attrs

        return all_attrs

    def _assert_shapes(self):
        pass

    def visualize_numpy(self, attrs: np.ndarray, image_array: np.ndarray):
        if len(attrs.shape) == 4:
            attrs = attrs.squeeze(0)
        if len(image_array.shape) == 4:
            image_array = image_array.squeeze(0)

        heatmap = np.transpose(attrs, (1, 2, 0))
        image = np.transpose(image_array, (1, 2, 0))

        # min max normalization
        # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        image = (image - image.min()) / (image.max() - image.min())

        _ = viz.visualize_image_attr_multiple(
            attr=heatmap,
            original_image=image,
            methods=["original_image", "heat_map"],
            signs=["all", "positive"],
            titles=[
                f"Original Image for {self.attribution_name} on {self.model.__class__.__name__}",
                f"Heat Map for {self.attribution_name} on {self.model.__class__.__name__}",
            ],
            show_colorbar=True,
            outlier_perc=2,
        )


""" refactored
def _handle_mlc_explanation(
    self, tensor_batch: torch.Tensor, target_batch: Union[int, torch.Tensor] = None
):
    batchsize, _, height, width = tensor_batch.shape

    # create output tensor without channels (classes * batchsize) x 1 x height x width
    all_attrs = torch.zeros((batchsize, self.num_classes, 1, height, width))

    # Get the indices of the non-zero targets
    non_zero_targets = target_batch.nonzero(as_tuple=True)

    # Select the images and targets that are non-zero
    image_tensors = tensor_batch[non_zero_targets]
    targets = target_batch[non_zero_targets]

    # Explain the selected images
    attrs = self.explain(image_tensors, targets)

    # If attrs is 3 channel sum over channels
    if len(attrs.shape) == 4 and attrs.shape[1] == 3:
        attrs = attrs.sum(dim=1, keepdim=True)

    # Assign the attrs to the corresponding positions in all_attrs
    all_attrs[non_zero_targets] = attrs

    return all_attrs
"""
