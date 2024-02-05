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
        multi_label: bool = False,
        num_classes: int = 6,
        vectorize: bool = False,
        normalize: bool = False,
    ):
        self.model = model
        self.multilabel = multi_label
        self.num_classes = num_classes
        self.vectorize = vectorize

        # todo: This is crap and should be removed
        self.only_explain_true_labels = False
        self.only_explain_predictions = True
        self.explain_true_label_and_preds = False

        self.normalize = normalize

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
        target_batch: torch.Tensor,
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
        target_batch : torch.Tensor, optional
            The corresponding targets for each image in the batch. If not provided,
            the method will compute attributions for all classes.

        Returns
        -------
        all_attrs : torch.Tensor
            A tensor of shape (batchsize, 1, height, width) containing the
            computed attributions for each image and target pair in the batch.
        """
        tensor_batch = tensor_batch.to(self.device)
        target_batch = [t.to(self.device) for t in target_batch]
        self.model.to(self.device)

        if self.multilabel:
            # todo: to fix the MLC explanation for the metrics we need to change the shape of the output?
            # currently it is (batchsize, num_classes, 1, height, width) but it should be (batchsize, 1, height, width)
            # this is because the metrics expect the shape to be (batchsize, 1, height, width)
            # thus we need to expand the input beforehand.
            # before we we use this method we expand the x_data to (batchsize * num_classes, channels, height, width)
            # and the y_data to (batchsize * num_classes, 1) and then we can use this method
            # then we have the same problem, how do we now which attributions belong to which input

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
        all_attrs = (
            torch.zeros((batchsize, self.num_classes, 1, height, width))
            .to(self.device)
            .float()
        )
        self.model.to(self.device)
        for batch_index in range(batchsize):
            image_tensor = tensor_batch[batch_index : batch_index + 1]
            # todo: somepoint expansion was necessary
            tmp_target = target_batch[batch_index]
            # if tmp_target is  0 dim tensor expand to 1 dim tensor
            if len(tmp_target.shape) == 0:
                tmp_target = tmp_target.unsqueeze(0)

            for target in tmp_target:
                image_tensor = image_tensor.to(self.device)
                target = target.to(self.device)

                attrs = self.explain(
                    image_tensor,
                    target,
                )
                # If attrs is 3 channel sum over channels
                if len(attrs.shape) == 4 and attrs.shape[1] == 3:
                    attrs = attrs.sum(dim=1, keepdim=True)

                all_attrs[batch_index, target] = attrs.float()
        return all_attrs

    def _skip_explanation(
        self, batch_index, label_index, labels_batch=None, target_batch=None
    ):
        """
        Check if we should skip the explanation for a specific label.

        Depending on what we want to explain: only true labels, only predictions, or both,
        we decide if we should skip the explanation for a specific label.
        Parameters
        ----------
        batch_index : int
            The index of the batch.
        label_index:
            The index of the label.
        labels_batch:
            The labels_batch, thea actual true labels in one hot encoding.
        target_batch
            The target_batch, the actual predictions in one hot encoding.

        Returns
        -------
        bool
            Whether to skip the explanation for the label.

        """
        # we only want to explain the labels that are present in the image

        if self.only_explain_true_labels:
            positive_label = labels_batch[batch_index][label_index]
            return positive_label == 0

        # we only want to explain the predictions that are present in the image
        elif self.only_explain_predictions:
            prediction = target_batch[batch_index][label_index]

            return prediction == 0
        elif self.explain_true_label_and_preds:
            positive_label = labels_batch[batch_index][label_index]
            prediction = target_batch[batch_index][label_index]
            return positive_label == 0 and prediction == 0

        return False

    def _handle_slc_explanation(
        self, tensor_batch: torch.Tensor, target_batch: Union[int, torch.Tensor] = None
    ):
        attrs = self.explain(tensor_batch, target_batch)

        # If attrs is 3 channel sum over channels
        if len(attrs.shape) == 4 and attrs.shape[1] == 3:
            attrs = attrs.sum(dim=1, keepdim=True)

        # min max normalization
        if self.normalize:
            attrs = self._normalize(attrs)

        # relu to remove negative attributions
        attrs = torch.relu(attrs)
        return attrs

    def _normalize(self, attrs):
        return (attrs - attrs.min()) / (attrs.max() - attrs.min())
