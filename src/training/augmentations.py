import random
from enum import IntEnum

import numpy as np
import torch
from einops import rearrange
from typing import Tuple


def CutMix_Xai(
    batch,
    threshold: float,
    get_box_version=1,
    max_aug_area=0.5,
    min_aug_area=0.1,
    aug_p=0.5,
    overhead=10,
):
    """
    The function take a batch = [img_tensors, segmentation_masks, labels] and returns
    a batch with the same shape but the content differs corresponding to the mask.

    It expects the XAI masks tobe parsed as segmentation masks.


    Parameters
    ----------
    batch
    threshold
    get_box_version
    max_aug_area
    min_aug_area
    aug_p
    overhead

    Returns
    -------

    """

    if not torch.is_tensor(batch[0]):
        raise ValueError(
            f"Expecting batch to be a torch.tensor but its {type(batch[0])}!"
        )
    batch[0] = rearrange(batch[0], "b c h w -> b h w c")

    # todo: Parse batch into dict format
    imgs = batch[0].detach().numpy()
    seg_masks = batch[1].detach().numpy()
    labels = batch[2]

    x1, x2, y1, y2 = generate_mask(
        batch=imgs,
        get_box_version=get_box_version,
        max_aug_area=max_aug_area,
        min_aug_area=min_aug_area,
        overhead=overhead,
    )

    new_order = np.random.permutation(imgs.shape[0])
    # its save because of advanced indexing https://numpy.org/doc/stable/user/basics.indexing.html

    new_imgs = imgs[new_order]

    new_seg_masks = seg_masks[new_order]
    # if labels are not changed we exlcude the multi-labels from derivation function
    not_changed = []
    for i, (xx1, xx2, yy1, yy2) in enumerate(zip(x1, x2, y1, y2)):
        if random.random() < aug_p:
            new_xx1, new_xx2, new_yy1, new_yy2 = get_random_box_location(
                xx2 - xx1, yy2 - yy1, seg_masks[i].shape[:2]
            )
            imgs[i][new_xx1:new_xx2, new_yy1:new_yy2] = new_imgs[i][xx1:xx2, yy1:yy2]
            seg_masks[i][new_xx1:new_xx2, new_yy1:new_yy2] = new_seg_masks[i][
                xx1:xx2, yy1:yy2
            ]
        else:
            not_changed.append(i)
    labels = derive_labels(
        old_labels=labels,
        seg_masks=seg_masks,
        not_changed=not_changed,
        threshold=threshold,
    )
    imgs = torch.from_numpy(imgs)
    imgs = rearrange(imgs, "b h w c -> b c h w")
    return [imgs, labels]


def generate_mask(
    batch, get_box_version=1, max_aug_area=0.5, min_aug_area=0.1, overhead=10
):
    # we want to select only the masked values from the batch -> every non selected is 0
    #    mask = np.zeros_like(batch[:, :, :, 0], dtype=np.bool8)

    if get_box_version == GetBoxVersion.Version1:
        x1, x2, y1, y2 = get_box_version1(
            batch, max_aug_area, min_aug_area, overhead=overhead
        )
    else:
        raise NotImplementedError(f"version: {get_box_version} is not implemented yet")

    """
    # place ones at each selected position (selected by the box)
    for i, (xx1, xx2, yy1, yy2) in enumerate(zip(x1, x2, y1, y2)):
        if np.random.random() < aug_p:
            mask[i, xx1:xx2, yy1:yy2] = 1

    return mask
    """
    return x1, x2, y1, y2


class GetBoxVersion(IntEnum):
    """
    The purpose of this class is to avoid magic numbers in the code
    and collect the different get_box_versions.
    """

    Version1 = 1
    Version2 = 2
    Version3 = 3


def get_box_version1(batch, max_aug_area, min_aug_area, overhead):
    """
    This function creates coordinates for the box.
    The box will have an area in the open interval (min_aug_area, aug_area).

    The coordinates are computed by generating random start and end values in the range of the maximal side length (given by img dimensions).
    Then it is checked whether the points are providing an area in the right intervall and if not we select new points.
    To speed up the process one can use an overhead. The overhead specifies the number of start and endpoints that should be generate at each point.

    """

    # x dim
    a = batch.shape[1]
    # y dim
    b = batch.shape[2]
    batch_size = batch.shape[0]

    x1 = np.random.randint(0, a, overhead * batch_size)
    x2 = np.random.randint(0, a, overhead * batch_size)

    y1 = np.random.randint(0, b, overhead * batch_size)
    y2 = np.random.randint(0, b, overhead * batch_size)

    # get the indices of of the entries that are bigger than the aug_area
    bigger_not_ready = abs((x2 - x1) * (y2 - y1) / (a * b)) > max_aug_area

    # get the indices of the entries that are less than the min_aug_area
    less_not_ready = abs((x2 - x1) * (y2 - y1)) / (a * b) < min_aug_area

    # collect the indices of the entries that don't fulfill the area condition
    not_ready = bigger_not_ready | less_not_ready

    # get the ones that are in the intervall
    ready = np.invert(not_ready)

    while ready.sum() < batch_size:
        x1[not_ready] = np.random.randint(0, a, not_ready.sum())
        x2[not_ready] = np.random.randint(0, a, not_ready.sum())

        y1[not_ready] = np.random.randint(0, b, not_ready.sum())
        y2[not_ready] = np.random.randint(0, b, not_ready.sum())

        bigger_not_ready = abs((x2 - x1) * (y2 - y1) / (a * b)) > max_aug_area
        less_not_ready = abs((x2 - x1) * (y2 - y1) / (a * b)) < min_aug_area

        not_ready = bigger_not_ready | less_not_ready

        ready = np.invert(not_ready)

    # select the first #batch_size start and endpoints that are fulfilling the condition
    x1 = x1[ready][:batch_size]
    x2 = x2[ready][:batch_size]

    y1 = y1[ready][:batch_size]
    y2 = y2[ready][:batch_size]

    # change the entries when startpoint is bigger than endpoint
    swap(x1, x2)
    swap(y1, y2)
    return x1, x2, y1, y2


def derive_labels(old_labels, seg_masks, not_changed, threshold):
    new_labels = torch.zeros_like(old_labels)
    new_labels[np.sum(seg_masks, axis=(1, 2)) > threshold] = 1.0
    new_labels[not_changed] = old_labels[not_changed]
    return new_labels


def swap(a1, a2):
    """
    Swap entries of a1 and a2 if the entry of a1 is greater then the entry of a2

    no return values -> call by reference for np.arrays
    Interesting: https://stackoverflow.com/questions/14933577/swap-slices-of-numpy-arrays/14933939#14933939
    """
    swap_idx = a1 > a2

    # works because we are using fixed index arrays
    a1[swap_idx], a2[swap_idx] = a2[swap_idx], a1[swap_idx]


def get_random_box_location(
    x_length: int, y_length: int, ground_shape: Tuple[int]
) -> Tuple[int]:
    assert ground_shape[0] > 1, "Please unsqueeze the first dim of the ground shape!"
    x1, x2 = random_coords(ground_shape[0], x_length)
    y1, y2 = random_coords(ground_shape[1], y_length)
    return x1, x2, y1, y2


def random_coords(max_length: int, should_length: int, start: int = 0) -> Tuple[int]:
    a1 = random.randrange(start, max_length - should_length)
    a2 = a1 + should_length
    return a1, a2
