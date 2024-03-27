import time
from abc import abstractmethod
from functools import wraps
from typing import Union

import torch


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def _normalize(attrs):
    return (attrs - attrs.min()) / (attrs.max() - attrs.min() + 0.00000001)


class Explanation:
    attribution_name = "Explanation"

    def __init__(
        self,
        model,
        device: torch.device,
        multi_label: bool = False,
        num_classes: int = 6,
        normalize: bool = False,
        dtype=torch.float32,
    ):
        self.model = model
        self.multilabel = multi_label
        self.num_classes = num_classes

        # todo: This is crap and should be removed
        self.only_explain_true_labels = False
        self.only_explain_predictions = True
        self.explain_true_label_and_preds = False

        self.normalize = normalize

        self.device = device
        self.dtype = dtype

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
        self,
        tensor_batch: torch.Tensor,
        target_batch: Union[list, torch.Tensor],
    ):
        """
        Explain a batch of images.

        This method takes a batch of image tensors and their corresponding targets,
        and computes the attributions for each image and target pair.
        The attributions are then stored in a tensor that is returned.

        If the model is a multi-label classification model, it will handle the explanation
        in a vectorized manner if vectorize is True, otherwise it will handle it in a non-vectorized manner.
        If the model is a single-label classification model, it will handle the explanation accordingly.

        Parameters
        ----------
        tensor_batch : torch.Tensor
            A batch of image tensors of shape (batchsize, channels, height, width).
        target_batch : Union[list, torch.Tensor]
            The corresponding targets for each image in the batch.
            ** Not one hot encoded, if multilabel classification a list **

        Returns
        -------
        all_attrs : torch.Tensor
            Single Label:
                A tensor of shape (batchsize, 1, height, width) containing the
                computed attributions for each image and target pair in the batch.
            Multilabel:
                A tensor of shape (batchsize, num_classes,  1, height, width) containing the
                computed attributions for each image and target pair in the batch.
        """
        tensor_batch = tensor_batch.to(self.device)

        if isinstance(target_batch, list):
            target_batch = [t.to(self.device) for t in target_batch]

        elif isinstance(target_batch, torch.Tensor):
            target_batch = target_batch.to(self.device)

        if self.multilabel:
            return self._handle_mlc_explanation(tensor_batch, target_batch)

        return self._handle_slc_explanation(tensor_batch, target_batch)

    def _handle_mlc_explanation(
        self,
        tensor_batch: torch.Tensor,
        target_batch: Union[int, torch.Tensor, list] = None,
    ):
        """
        Handle multi-label classification explanation.

        This method takes a batch of image tensors and their corresponding targets,
        and computes the attributions for each image and target pair.
        The attributions are then stored in a tensor that is returned.

        Parameters
        ----------
        tensor_batch : torch.Tensor
            A batch of image tensors of shape (batchsize, channels, height, width).
        target_batch : Union[int, torch.Tensor], optional
            The corresponding targets for each image in the batch. If not provided,
            the method will compute attributions for all classes. Not one hot encoded!

        Returns
        -------
        all_attrs : torch.Tensor
            A tensor of shape (batchsize, num_classes, 1, height, width) containing the
            computed attributions for each image and target pair in the batch.
        """
        batchsize, _, height, width = tensor_batch.shape
        # create output tensor without channels (classes * batchsize) x 1 x height x width
        all_attrs = torch.zeros((batchsize, self.num_classes, 1, height, width)).to(
            self.device, dtype=self.dtype
        )

        for batch_index in range(batchsize):
            image_tensor = tensor_batch[batch_index : batch_index + 1]
            image_tensor = image_tensor.to(self.device)
            tmp_target = target_batch[batch_index]
            # if tmp_target is  0 dim tensor expand to 1 dim tensor
            if len(tmp_target.shape) == 0:
                tmp_target = tmp_target.unsqueeze(0)

            # if the labels are one-hot-encoded, convert them to indices
            if len(tmp_target) != 1:
                # The explanation expects the targets not to be one-hot-encoded.
                tmp_target = torch.nonzero(tmp_target, as_tuple=False)

            for target in tmp_target:
                target = target.to(self.device)
                image_tensor = image_tensor.to(self.device, dtype=self.dtype)

                attrs = self.explain(
                    image_tensor,
                    target,
                )

                attrs = self._post_process_attribution(attrs)
                if attrs.shape != all_attrs[batch_index, target]:
                    # Shape mismatch e.g. from GradCAM without interpolation
                    attrs = torch.nn.functional.interpolate(
                        attrs,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                all_attrs[batch_index, target] = attrs.to(dtype=self.dtype)
        return all_attrs

    def _handle_slc_explanation(
        self, tensor_batch: torch.Tensor, target_batch: Union[int, torch.Tensor] = None
    ):
        attrs = self.explain(tensor_batch, target_batch)

        attrs = self._post_process_attribution(attrs)
        return attrs

    def _post_process_attribution(self, attribution):
        # If attrs is 3 channel sum over channels
        if len(attribution.shape) == 4 and attribution.shape[1] == 3:
            attribution = attribution.sum(dim=1, keepdim=True)

        # min max normalization
        if self.normalize:
            attribution = _normalize(attribution)

        # relu to remove negative attributions
        # attribution = torch.relu(attribution)
        return attribution
