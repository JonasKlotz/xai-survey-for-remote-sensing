from typing import Dict
from typing import Iterable
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from data.ben.BENv2Utils import ben_19_labels_to_multi_hot as ben_19_labels_to_multi_hot_np


def stack_and_interpolate(
    bands: Dict[str, np.ndarray],
    order: Optional[Iterable[str]] = None,
    img_size: int = 120,
    upsample_mode: str = "nearest",
) -> np.array:
    """
    Stacks the bands in the order given by order and interpolates them to the given img_size.

    :param bands: dict of the form {bandname: np.ndarray} where the np.ndarray is of shape (height, width)
    :param order: order of the bands, defaults to alphabetical order of the keys of :param:`bands`
    :param img_size: size of the output image
    :param upsample_mode: interpolation mode, defaults to "nearest". Supports "nearest", "bilinear" and "bicubic"
        interpolation
    :return: torch.Tensor of shape (len(bands), img_size, img_size)
    """

    def _interpolate(img_data):
        if not img_data.shape[-2:] == (img_size, img_size):
            return F.interpolate(
                torch.Tensor(np.float32(img_data)).unsqueeze(0).unsqueeze(0),
                (img_size, img_size),
                mode=upsample_mode,
                align_corners=True
                if upsample_mode in ["bilinear", "bicubic"]
                else None,
            ).squeeze()
        else:
            return torch.Tensor(np.float32(img_data))

    # if order is None, order is alphabetical
    if order is None:
        order = sorted(bands.keys())
    return torch.stack([_interpolate(bands[x]) for x in order])


def ben_19_labels_to_multi_hot(
    labels: Iterable[str], lex_sorted: bool = True
) -> torch.Tensor:
    """
    Convenience function that converts an input iterable of labels into a multi-hot encoded vector.
    If `lex_sorted` is True (default) the classes are lexigraphically ordered, as they are in `constants.NEW_LABELS`.

    If an unknown label is given, a `KeyError` is raised.

    Be aware that this approach assumes that **all** labels are actually used in the dataset!
    This is not necessarily the case if you are using a subset!

    :param labels: iterable of labels
    :param lex_sorted: whether to lexigraphically sort the labels, defaults to True
    :return: multi-hot encoded vector as torch.Tensor of shape (len(constants.NEW_LABELS),) where the i-th entry is 1 if
        the i-th label is present in `labels` and 0 otherwise
    """
    return torch.from_numpy(ben_19_labels_to_multi_hot_np(labels, lex_sorted))
