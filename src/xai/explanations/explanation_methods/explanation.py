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


class Explanation:
    attribution_name = "Explanation"

    def __init__(
        self,
        model,
        device: torch.device,
        multi_label: bool = True,
        num_classes: int = 6,
        vectorize: bool = False,
        model_name: str = None,
    ):
        self.model = model
        self.multi_label = multi_label
        self.num_classes = num_classes
        self.vectorize = vectorize
        self.only_explain_true_labels = False
        self.only_explain_predictions = False
        self.explain_true_label_and_preds = True

        self.device = device

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
        prediction_batch: torch.Tensor,
        labels_batch: torch.Tensor,
    ):
        """
        Explain a batch of images.

        This method takes a batch of image tensors and their corresponding targets,
        and computes the attributions for each image and target pair.
        The attribitions are then stored in a tensor that is returned.

        If the model is a multi-label classification model, it will handle the explanation
        in a vectorized manner if vectorize is True, otherwise it will handle it in a non-vectorized manner.
        If the model is a single-label classification model, it will handle the explanation accordingly.

        Parameters
        ----------
        tensor_batch : torch.Tensor
            A batch of image tensors of shape (batchsize, channels, height, width).
        prediction_batch : torch.Tensor, optional
            The corresponding targets for each image in the batch. If not provided,
            the method will compute attributions for all classes.

        Returns
        -------
        all_attrs : torch.Tensor
            A tensor of shape (batchsize, num_classes, 1, height, width) containing the
            computed attributions for each image and target pair in the batch.
        """
        if self.multi_label:
            return self._handle_mlc_explanation(
                tensor_batch, prediction_batch, labels_batch
            )

        return self._handle_slc_explanation(tensor_batch, prediction_batch)

    def _handle_mlc_explanation_vectorized(
        self,
        image_tensors: torch.Tensor,
        target_batch: torch.Tensor = None,
        labels_batch: torch.Tensor = None,
    ):
        """
        Handle multi-label classification explanation in a vectorized manner.

        This method takes a batch of image tensors and their corresponding targets,
        reshapes the image tensors, and computes the attributions for each image and target pair.
        The attributions are then stored in a tensor that is returned.

        Parameters
        ----------
        image_tensors : torch.Tensor
            A batch of image tensors of shape (batchsize, channels, height, width).
        target_batch : torch.Tensor, optional
            The corresponding targets for each image in the batch. If not provided,
            the method will compute attributions for all classes.

        Returns
        -------
        all_attrs : torch.Tensor
            A tensor of shape (batchsize, num_classes, 1, height, width) containing the
            computed attributions for each image and target pair in the batch.
        """
        batchsize, _, height, width = image_tensors.shape

        # Getting the indices of zeros and ones
        # todo this has changed check if correct
        ones_indices = (labels_batch == 1).nonzero(as_tuple=False)

        target_indices = ones_indices[:, 0] * target_batch.size(1) + ones_indices[:, 1]
        targets = ones_indices[:, 1]

        image_tensors = (
            image_tensors.unsqueeze(1)
            .expand(-1, 6, -1, -1, -1)
            .reshape(-1, 3, 120, 120)
        )
        image_tensors = image_tensors[target_indices]

        attrs = self.explain(image_tensors, targets)
        # If attrs is 3 channel sum over channels
        if len(attrs.shape) == 4 and attrs.shape[1] == 3:
            attrs = attrs.sum(dim=1, keepdim=True)

        all_attrs = torch.zeros(
            (batchsize * self.num_classes, 1, height, width), dtype=attrs.dtype
        )
        all_attrs[target_indices] = attrs
        all_attrs = all_attrs.reshape(batchsize, self.num_classes, 1, height, width)
        return all_attrs

    def _handle_mlc_explanation(
        self,
        tensor_batch: torch.Tensor,
        target_batch: Union[int, torch.Tensor] = None,
        labels_batch: torch.Tensor = None,
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
            the method will compute attributions for all classes.

        Returns
        -------
        all_attrs : torch.Tensor
            A tensor of shape (batchsize, num_classes, 1, height, width) containing the
            computed attributions for each image and target pair in the batch.
        """
        batchsize, _, height, width = tensor_batch.shape
        # create output tensor without channels (classes * batchsize) x 1 x height x width
        all_attrs = (
            torch.zeros((batchsize, self.num_classes, 1, height, width))
            .to(self.device)
            .float()
        )
        self.model.to(self.device)
        for batch_index in range(batchsize):
            image_tensor = tensor_batch[batch_index : batch_index + 1]

            for label_index in range(self.num_classes):
                # we only want to explain the labels that are present in the image or the predictions
                if self._skip_explanation(
                    batch_index, label_index, labels_batch, target_batch
                ):
                    continue

                image_tensor = image_tensor.to(self.device)
                label_index = torch.tensor(label_index).unsqueeze(0).to(self.device)

                attrs = self.explain(
                    image_tensor,
                    label_index,
                )
                # If attrs is 3 channel sum over channels
                if len(attrs.shape) == 4 and attrs.shape[1] == 3:
                    attrs = attrs.sum(dim=1, keepdim=True)

                all_attrs[batch_index, label_index] = attrs.float()
        return all_attrs

    def _skip_explanation(self, batch_index, label_index, labels_batch, target_batch):
        # we only want to explain the labels that are present in the image
        positive_label = labels_batch[batch_index][label_index]
        prediction = target_batch[batch_index][label_index]
        if self.only_explain_true_labels:
            return positive_label == 0
        # we only want to explain the predictions that are present in the image
        elif self.only_explain_predictions:
            return prediction == 0
        elif self.explain_true_label_and_preds:
            return positive_label == 0 and prediction == 0
        return False

    def _handle_slc_explanation(
        self, tensor_batch: torch.Tensor, target_batch: Union[int, torch.Tensor] = None
    ):
        # iterate over batch and explain each image
        all_attrs = []
        for batch_index in range(tensor_batch.shape[0]):
            image_tensor = tensor_batch[batch_index : batch_index + 1]
            target = target_batch[batch_index : batch_index + 1]
            attrs = self.explain(image_tensor, target)
            all_attrs.append(attrs)
        all_attrs = torch.stack(all_attrs)
        return all_attrs
