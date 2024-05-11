import os

import numpy as np
import pandas as pd
import torch
import tqdm
from einops import rearrange
from torchmetrics import Accuracy, F1Score

from data.data_utils import (
    get_dataloader_from_cfg,
    parse_batch,
    segmask_to_multilabel_torch,
)  # noqa: E402
from data.lmdb_handler import LMDBDataHandler
from training.augmentations import generate_mask, get_random_box_location, derive_labels
from utility.cluster_logging import logger  # noqa: E402


def setup_device_and_data_config(cfg):
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {cfg['device']}")
    cfg["data"].update({"num_workers": 1, "batch_size": 32, "method": "explain"})
    return cfg


def sanity_check_augmentation_thresholds(cfg):
    """
    This function calculates the accuracies and f1 scores for different thresholds of cutmix and segmentation
    for each explanation method. The results are saved to a csv file.

    Parameters
    ----------
    cfg

    Returns
    -------

    """
    logger.debug("Calculating threshold for cutmix and segmentation")
    logger.debug(f"Using CUDA: {torch.cuda.is_available()}")

    cfg = setup_device_and_data_config(cfg)

    cfg, data_loader = get_dataloader_from_cfg(cfg, loader_name="train")

    # Create LMDB HANDLER for each explanation method
    lmdb_base_path = cfg["data"]["lmdb_base_path"]
    lmdb_hander_dict = {}
    for explanation_method in cfg["explanation_methods"]:
        lmdb_hander_dict[explanation_method] = LMDBDataHandler(
            path=f"{lmdb_base_path}/{explanation_method}.lmdb", read_only=True
        )

    # Setup Hyperparameters
    max_aug_area = cfg.get("max_aug_area", 0.5)
    min_aug_area = cfg.get("min_aug_area", 0.1)
    overhead = cfg.get("overhead", 10)

    segmentation_thresholds = [0, 0.05, 0.1, 0.2, 0.5]
    cutmix_thresholds = [0, 5, 10, 20, 50]

    # We permute the batch to avoid the same image being compared to itself
    # The order can be the same for all images
    new_order = [
        (i + 1) % cfg["data"]["batch_size"] for i in range(cfg["data"]["batch_size"])
    ]

    accuracy = Accuracy(task="multilabel", num_labels=6)
    f1_score = F1Score(task="multilabel", num_labels=6)
    # generate empty dataframe with the segmentation thresholds as columns and cutmix thresholds as rows
    accuracy_arrays = {
        explanation_method: np.zeros(shape=(5, 5))
        for explanation_method in cfg["explanation_methods"]
    }
    f1_arrays = {
        explanation_method: np.zeros(shape=(5, 5))
        for explanation_method in cfg["explanation_methods"]
    }
    csv_dir_path = f"{cfg["results_path"]}/{cfg["experiment_name"]}/sanity_check_augmentation_thresholds"
    os.makedirs(csv_dir_path, exist_ok=True)

    i = 0
    # Iterate over the data loader
    for batch in tqdm.tqdm(data_loader):
        i += cfg["data"]["batch_size"]
        try:
            # This method does inplace operations on the accuracy and f1 arrays
            process_batch(
                accuracy,
                accuracy_arrays,
                batch,
                cfg,
                cutmix_thresholds,
                data_loader,
                f1_arrays,
                f1_score,
                lmdb_hander_dict,
                max_aug_area,
                min_aug_area,
                new_order,
                overhead,
                segmentation_thresholds,
            )

        except Exception as e:
            logger.error(f"Error in batch {i}: {e}")
            continue

    # divide number of samples to get the average
    for explanation_method in cfg["explanation_methods"]:
        accuracy_arrays[explanation_method] /= i
        f1_arrays[explanation_method] /= i

        # convert to dataframe
        accuracy_df = pd.DataFrame(
            accuracy_arrays[explanation_method],
            index=cutmix_thresholds,
            columns=segmentation_thresholds,
        )
        f1_df = pd.DataFrame(
            f1_arrays[explanation_method],
            index=cutmix_thresholds,
            columns=segmentation_thresholds,
        )
        logger.debug(f"Accuracy DataFrame for {explanation_method}:\n {accuracy_df}")
        logger.debug(f"F1 DataFrame for {explanation_method}:\n {f1_df}")

        # save to csv
        accuracy_df.to_csv(f"{csv_dir_path}/accuracy_{explanation_method}.csv")
        f1_df.to_csv(f"{csv_dir_path}/f1_{explanation_method}.csv")
    logger.debug(
        f"Finished sanity check augmentation thresholds. Saved to \n{csv_dir_path}"
    )
    # close lmdb handlers
    for lmdb_handler in lmdb_hander_dict.values():
        lmdb_handler.close()


def process_batch(
    accuracy,
    accuracy_arrays,
    batch,
    cfg,
    cutmix_thresholds,
    data_loader,
    f1_arrays,
    f1_score,
    lmdb_hander_dict,
    max_aug_area,
    min_aug_area,
    new_order,
    overhead,
    segmentation_thresholds,
):
    (
        features,
        true_targets,
        _,
        true_segmentations,
        indices,
        _,
    ) = parse_batch(batch)
    # Create original Segmentations
    features = rearrange(features, "b c h w -> b h w c")
    true_segmentations = true_segmentations.squeeze()
    #
    x1, x2, y1, y2 = generate_mask(
        batch=features,
        max_aug_area=max_aug_area,
        min_aug_area=min_aug_area,
        overhead=overhead,
    )
    new_seg_masks = true_segmentations[new_order]
    new_indices = indices[new_order]
    # The indices that will be replaced in the original image
    to_replace_indices = get_indices_to_replace(true_segmentations, x1, x2, y1, y2)
    for i in range(cfg["data"]["batch_size"]):
        # access to replace indices
        (
            to_replace_x1,
            to_replace_x2,
            to_replace_y1,
            to_replace_y2,
        ) = to_replace_indices[i]
        # access x1, x2, y1, y2
        x1_i, x2_i, y1_i, y2_i = x1[i], x2[i], y1[i], y2[i]
        # generate new true segmentation
        true_segmentation = true_segmentations[i]
        true_target = true_targets[i].unsqueeze(0)

        new_true_target = get_new_true_target(
            batch,
            i,
            new_seg_masks,
            to_replace_x1,
            to_replace_x2,
            to_replace_y1,
            to_replace_y2,
            true_segmentation,
            x1_i,
            x2_i,
            y1_i,
            y2_i,
        )

        # logger.debug(
        #     f"True Target: {true_targets[i].shape} \nNew Target: {new_true_target.shape}"
        # )
        for explanation_method in cfg["explanation_methods"]:
            lmdb_handler = lmdb_hander_dict[explanation_method]
            acc_arr = accuracy_arrays[explanation_method]
            f1_arr = f1_arrays[explanation_method]

            explanation_mask = get_mask_from_lmdb(data_loader, i, indices, lmdb_handler)
            new_explanation_mask = get_mask_from_lmdb(
                data_loader, i, new_indices, lmdb_handler
            )

            for seg_idx, segmentation_threshold in enumerate(segmentation_thresholds):
                threshed_explanation_mask = get_new_threshed_explanation_mask(
                    explanation_mask,
                    new_explanation_mask,
                    segmentation_threshold,
                    to_replace_x1,
                    to_replace_x2,
                    to_replace_y1,
                    to_replace_y2,
                    x1_i,
                    x2_i,
                    y1_i,
                    y2_i,
                )

                # logger.debug(f"True Target: {true_target.shape}")
                for cutmix_idx, cutmix_threshold in enumerate(cutmix_thresholds):
                    derived_targets = derive_labels(
                        old_labels=true_target,
                        seg_masks=threshed_explanation_mask,
                        not_changed=[],
                        threshold=cutmix_threshold,
                    ).to(dtype=batch["targets"].dtype)

                    # logger.debug(
                    #     f"Derived Target: {derived_targets} \nNew True Target: {new_true_target}"
                    # )
                    acc = accuracy(derived_targets, new_true_target)
                    f1 = f1_score(derived_targets, new_true_target)

                    # logger.debug(
                    #     f"Accuracy for {explanation_method} with segmentation thresh {segmentation_threshold} and "
                    #     f"cutmix thresh {cutmix_threshold}:\n {acc}"
                    # )
                    #
                    # logger.debug(
                    #     f"F1 Score for {explanation_method} with segmentation thresh {segmentation_threshold} and "
                    #     f"cutmix thresh {cutmix_threshold}:\n {f1}"
                    # )
                    acc_arr[cutmix_idx, seg_idx] += acc.item()
                    f1_arr[cutmix_idx, seg_idx] += f1.item()


def get_new_threshed_explanation_mask(
    explanation_mask,
    new_explanation_mask,
    segmentation_threshold,
    to_replace_x1,
    to_replace_x2,
    to_replace_y1,
    to_replace_y2,
    x1_i,
    x2_i,
    y1_i,
    y2_i,
):
    threshed_explanation_mask = (
        explanation_mask.clone() > segmentation_threshold
    ).bool()
    threshed_new_explanation_mask = (
        new_explanation_mask.clone() > segmentation_threshold
    ).bool()
    # replace the explanation mask
    threshed_explanation_mask[
        to_replace_x1:to_replace_x2, to_replace_y1:to_replace_y2
    ] = threshed_new_explanation_mask[x1_i:x2_i, y1_i:y2_i]
    # unsqueeze the explanation mask
    threshed_explanation_mask = threshed_explanation_mask.unsqueeze(0)
    return threshed_explanation_mask


def get_new_true_target(
    batch,
    i,
    new_seg_masks,
    to_replace_x1,
    to_replace_x2,
    to_replace_y1,
    to_replace_y2,
    true_segmentation,
    x1_i,
    x2_i,
    y1_i,
    y2_i,
):
    true_segmentation[to_replace_x1:to_replace_x2, to_replace_y1:to_replace_y2] = (
        new_seg_masks[i][x1_i:x2_i, y1_i:y2_i]
    )
    # unsqueeze the true segmentation
    true_segmentation = true_segmentation.unsqueeze(0)
    new_true_target = segmask_to_multilabel_torch(true_segmentation, num_classes=6).to(
        dtype=batch["targets"].dtype
    )
    return new_true_target


def get_mask_from_lmdb(data_loader, i, indices, lmdb_handler):
    patchname = data_loader.dataset.get_patch_name(indices[i])
    explanation_mask = lmdb_handler[patchname]
    explanation_mask = torch.tensor(explanation_mask)
    explanation_mask = explanation_mask.squeeze()
    return explanation_mask


def get_indices_to_replace(true_segmentations, x1, x2, y1, y2):
    to_replace_indices = []
    for i, (xx1, xx2, yy1, yy2) in enumerate(zip(x1, x2, y1, y2)):
        (
            to_replace_x1,
            to_replace_x2,
            to_replace_y1,
            to_replace_y2,
        ) = get_random_box_location(
            xx2 - xx1, yy2 - yy1, true_segmentations[i].shape[:2]
        )
        indices = (to_replace_x1, to_replace_x2, to_replace_y1, to_replace_y2)
        to_replace_indices.append(indices)
    return to_replace_indices
