import random
from typing import Tuple

import numpy as np
import torch
from einops import rearrange

from data.data_utils import segmask_to_multilabel_torch
from training.dataset_sanity_checker import calculate_segmentation_label_mse_loss


def CutMix_segmentations(
    batch,
    max_aug_area=0.5,
    min_aug_area=0.1,
    aug_p=0.5,
    overhead=10,
):
    """Calculates CutMix using the label information from the segmentation masks."""

    features = batch["features"]
    features = rearrange(features, "b c h w -> b h w c")

    segmentations = batch["segmentations"]
    segmentations = segmentations.squeeze()

    x1, x2, y1, y2 = generate_mask(
        batch=features,
        max_aug_area=max_aug_area,
        min_aug_area=min_aug_area,
        overhead=overhead,
    )
    # we generate a permutation
    new_order = np.random.permutation(features.shape[0])
    # its save because of advanced indexing https://numpy.org/doc/stable/user/basics.indexing.html

    # these are the images which we insert into the old images
    new_imgs = features[new_order]
    new_seg_masks = segmentations[new_order]

    # if labels are not changed we exclude the multi-labels from derivation function
    not_changed = []

    for i, (xx1, xx2, yy1, yy2) in enumerate(zip(x1, x2, y1, y2)):
        if random.random() < aug_p:
            new_xx1, new_xx2, new_yy1, new_yy2 = get_random_box_location(
                xx2 - xx1, yy2 - yy1, segmentations[i].shape[:2]
            )

            features[i][new_xx1:new_xx2, new_yy1:new_yy2] = new_imgs[i][
                xx1:xx2, yy1:yy2
            ]

            segmentations[i][new_xx1:new_xx2, new_yy1:new_yy2] = new_seg_masks[i][
                xx1:xx2, yy1:yy2
            ]
        else:
            not_changed.append(i)

    targets = segmask_to_multilabel_torch(segmentations, num_classes=6).to(
        dtype=batch["targets"].dtype
    )
    # overwrite the not changed labels
    targets[not_changed] = batch["targets"][not_changed]

    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    features = rearrange(features, "b h w c -> b c h w")

    assert (
        calculate_segmentation_label_mse_loss(segmentations, targets) == 0.0
    ), f"Error for labels and segmentation: {calculate_segmentation_label_mse_loss(segmentations, targets)}"

    # overwrite the old values
    batch["features"] = features
    batch["targets"] = targets
    batch["segmentations"] = segmentations

    return batch


def CutMix_Xai(
    batch,
    threshold: float,
    max_aug_area=0.5,
    min_aug_area=0.1,
    aug_p=0.5,
    overhead=10,
    segmentation_threshold=0.1,
):
    """
    The function take a batch = [img_tensors, segmentation_masks, labels] and returns
    a batch with the same shape but the content differs corresponding to the mask.

    It expects the XAI masks to be parsed as segmentation masks.

    Assert the features to be in the shape [batch_size, channels, height, width]
    the segmentation masks to be in the shape [batch_size,channel,  height, width, classes] b h w c
    and the labels to be in the shape [batch_size, classes]

    """
    features = batch["features"]

    features = rearrange(features, "b c h w -> b h w c")

    targets = batch["targets"]
    segmentations = batch["xai_segmentations"]
    # squeeze the segmentation masks
    segmentations = segmentations.squeeze()

    # get the mask using the segmentation threshold
    segmentations_mask = (segmentations > segmentation_threshold).bool()

    x1, x2, y1, y2 = generate_mask(
        batch=features,
        max_aug_area=max_aug_area,
        min_aug_area=min_aug_area,
        overhead=overhead,
    )
    # we generate a permutation
    new_order = np.random.permutation(features.shape[0])
    # its save because of advanced indexing https://numpy.org/doc/stable/user/basics.indexing.html

    # these are the images which we insert into the old images
    new_imgs = features[new_order]
    new_seg_masks = segmentations_mask[new_order]

    # if labels are not changed we exlcude the multi-labels from derivation function
    not_changed = []

    for i, (xx1, xx2, yy1, yy2) in enumerate(zip(x1, x2, y1, y2)):
        if random.random() < aug_p:
            new_xx1, new_xx2, new_yy1, new_yy2 = get_random_box_location(
                xx2 - xx1, yy2 - yy1, segmentations_mask[i].shape[:2]
            )

            tmp = new_imgs[i][xx1:xx2, yy1:yy2]
            features[i][new_xx1:new_xx2, new_yy1:new_yy2] = tmp

            segmentations_mask[i][new_xx1:new_xx2, new_yy1:new_yy2] = new_seg_masks[i][
                xx1:xx2, yy1:yy2
            ]
        else:
            not_changed.append(i)

    targets = derive_labels(
        old_labels=targets,
        seg_masks=segmentations_mask,
        not_changed=not_changed,
        threshold=threshold,
    )
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    features = rearrange(features, "b h w c -> b c h w")

    # overwrite the old values
    batch["features"] = features
    batch["targets"] = targets

    return batch


def generate_mask(batch, max_aug_area=0.5, min_aug_area=0.1, overhead=10):
    x1, x2, y1, y2 = get_box(batch, max_aug_area, min_aug_area, overhead=overhead)
    return x1, x2, y1, y2


def get_box(batch, max_aug_area, min_aug_area, overhead):
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
    """
    This function derives the new labels from the segmentation masks.

    the threshold here is TMAP in the paper.
    Parameters
    ----------
    old_labels
    seg_masks
    not_changed
    threshold

    Returns
    -------

    """
    new_labels = torch.zeros_like(old_labels)

    if isinstance(seg_masks, torch.Tensor):
        seg_masks = seg_masks.numpy(force=True)

    new_labels[np.sum(seg_masks, axis=(1, 2)) > threshold] = 1.0

    if not_changed != []:
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


def get_box_vectorized(batch, max_aug_area, min_aug_area, overhead):
    # Ensuring inputs are in torch format for vectorized operations
    batch_size, a, b, _ = batch.shape

    # Vectorized generation of random start and end points
    x1, x2 = torch.randint(0, a, (2, overhead * batch_size))
    y1, y2 = torch.randint(0, b, (2, overhead * batch_size))

    # Ensure x2 is always greater than x1 (and similarly for y1, y2)
    x1, x2 = torch.min(x1, x2), torch.max(x1, x2)
    y1, y2 = torch.min(y1, y2), torch.max(y1, y2)

    # Compute areas and filter based on min and max area constraints
    areas = (x2 - x1) * (y2 - y1) / (a * b)
    valid = (areas > min_aug_area) & (areas < max_aug_area)

    # Ensure we have at least one valid box per image in the batch
    while valid.view(batch_size, overhead).sum(dim=1).min() == 0:
        # Regenerate for invalid ones only
        not_valid_idx = valid.view(batch_size, overhead).sum(dim=1) == 0
        nv_size = not_valid_idx.sum() * overhead

        # Regenerate points for not valid cases
        x1[not_valid_idx], x2[not_valid_idx] = torch.randint(0, a, (2, nv_size))
        y1[not_valid_idx], y2[not_valid_idx] = torch.randint(0, b, (2, nv_size))

        x1, x2 = torch.min(x1, x2), torch.max(x1, x2)
        y1, y2 = torch.min(y1, y2), torch.max(y1, y2)

        areas = (x2 - x1) * (y2 - y1) / (a * b)
        valid = (areas > min_aug_area) & (areas < max_aug_area)

    # Select the first valid set of points for each image
    x1 = x1[valid][:batch_size]
    x2 = x2[valid][:batch_size]
    y1 = y1[valid][:batch_size]
    y2 = y2[valid][:batch_size]

    return x1, x2, y1, y2


if __name__ == "__main__":
    # Test the vectorized function
    batch_size, channels, height, width = 10, 3, 224, 224
    max_aug_area = 0.5
    min_aug_area = 0.1
    overhead = 10

    # Original method setup
    batch = torch.randn(batch_size, channels, height, width)
    x1, x2, y1, y2 = get_box(batch, max_aug_area, min_aug_area, overhead)

    # Vectorized method setup
    x1_v, x2_v, y1_v, y2_v = get_box_vectorized(
        batch, max_aug_area, min_aug_area, overhead
    )

    # Compare outputs (This is illustrative. Exact matching might not be possible due to randomness)
    print("Original:", x1, x2, y1, y2)
    print("Vectorized:", x1_v, x2_v, y1_v, y2_v)
