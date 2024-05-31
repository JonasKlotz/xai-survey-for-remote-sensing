import os
import pprint

import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
import torch
from tqdm import tqdm

from data.data_utils import (
    get_index_to_name,
    load_data_module,
    get_loader_for_datamodule,
)
from utility.cluster_logging import logger


def plot_datamodule_distribution(cfg: dict):
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 64

    # Load the dataset
    data_module, cfg = load_data_module(cfg)
    contains_segmentation = False  # cfg["dataset_name"] in ["deepglobe"]
    multilabel = cfg["task"] == "multilabel"
    if cfg["debug"]:
        loader_names = ["test"]
    else:
        loader_names = ["test", "val", "train"]
    full_label_distribution = torch.zeros(data_module.num_classes, dtype=torch.int64)
    total_samples = 0
    summary = {}

    for loader_name in loader_names:
        data_loader = get_loader_for_datamodule(data_module, loader_name=loader_name)
        label_distribution, segmentation_distribution = get_dataloader_distribution(
            data_loader,
            data_module.num_classes,
            contains_segmentation=contains_segmentation,
            multilabel=multilabel,
        )
        full_label_distribution += label_distribution

        print(f"Label distribution for {loader_name}: {label_distribution}")
        if contains_segmentation:
            print(
                f"Segmentation distribution for {loader_name}: {segmentation_distribution}"
            )
        n_dataloader_samples = len(data_loader.dataset)
        summary[loader_name] = {
            "label_distribution": label_distribution,
            "n_samples": n_dataloader_samples,
            "avg_classes_per_sample": label_distribution.sum().float()
            / n_dataloader_samples,
        }
        total_samples += len(data_loader.dataset)

    summary["full"] = {
        "label_distribution": full_label_distribution,
        "n_samples": total_samples,
        "avg_classes_per_sample": full_label_distribution.sum().float() / total_samples,
    }
    logger.debug(f"General config: {pprint.pformat(summary)}")
    normalized_distribution = full_label_distribution.float() / total_samples
    fig = plot_distribution(cfg, normalized_distribution)

    # save as pdf
    output_path = f"{cfg['results_path']}/{cfg['experiment_name']}/visualizations"
    os.makedirs(output_path, exist_ok=True)
    outputname = f"{cfg['dataset_name']}_label_distribution.pdf"
    if cfg["debug"]:
        fig.show()
    # garbage graph
    garbage = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    garbage.write_image(f"{output_path}/{outputname}")
    fig.write_image(f"{output_path}/{outputname}")


def get_dataloader_distribution(
    data_loader,
    num_classes,
    contains_segmentation=False,
    multilabel=False,
    target_key="targets",
    segmentation_key="segmentations",
):
    """
    Get the distribution of the labels in the dataset.

    Dataloader should return a dictionary with the keys:
        "features": patch,
        "targets": label,
        "index": idx,
        "segmentations": segmentation_patch,
    Otherwise a ValueError will be raised.
    If it's a multilabel dataset, the targets should be a tensor of shape (n_classes, ),
    else it should be a scalar.
    If the dataset contains segmentations, the segmentation_patch should be a tensor of shape (height, width),
    with values in the range [0, n_classes]. If the dataset does not contain segmentations, the segmentation_patch should be None.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        DataLoader to analyze.
    num_classes : int
        Number of classes in the dataset.
    contains_segmentation : bool, optional
        Whether the data includes segmentation masks.
    multilabel : bool, optional
        Whether the dataset is multilabel.

    Returns
    -------
    label_distribution : torch.Tensor
        A tensor of shape (num_classes,) where each entry is the count of labels for that class in the dataset.

    Raises
    ------
    ValueError
        If the data in the DataLoader does not conform to the expected format.
    """

    # Initialize a tensor to store class counts
    label_distribution = torch.zeros(num_classes, dtype=torch.int64)
    segmentation_distribution = torch.zeros(num_classes, dtype=torch.int64)
    for i, batch_dict in enumerate(tqdm(data_loader)):
        if contains_segmentation:
            _add_segmentation(
                batch_dict, segmentation_distribution, num_classes, segmentation_key
            )
        _add_label(batch_dict, label_distribution, multilabel, num_classes, target_key)

    return label_distribution, segmentation_distribution


def _add_segmentation(
    batch, segmentation_distribution, num_classes, segmentation_key="segmentations"
):
    """
    Count the number of pixels for each class in the segmentation masks.
    The segmentation masks should be tensors of shape (height, width) with each pixel's value representing a class label.
    Each pixel value should be within the range [0, num_classes - 1].

    Parameters
    ----------
    batch : dict
        The batch containing the "segmentations" key with the segmentation masks.The tensor should be (batchsize, height, width).
    segmentation_distribution : torch.Tensor
        A tensor to store the pixel counts for each class. It should be of size (num_classes,).
    num_classes : int
        The number of classes, which determines the range of possible pixel values in the segmentation masks.

    Raises
    ------
    ValueError
        If the segmentation masks are not in the expected format or the pixel values are out of the expected range.
    """
    if segmentation_key not in batch:
        raise ValueError(f"Batch does not contain '{segmentation_key}'.")

    segmentations = batch[segmentation_key]

    # Check if segmentations is a tensor of the correct shape (height, width)
    if segmentations.ndim != 4:
        raise ValueError(
            "Segmentation mask should be a tensor of shape (batchsize,1, height, width)."
        )

    # Validate that all values are within the valid range [0, num_classes-1]
    if not torch.all((segmentations >= 0) & (segmentations < num_classes)):
        raise ValueError(
            "Segmentation mask values should be in the range [0, num_classes - 1]."
        )

    # Count pixels for each class
    for class_index in range(num_classes):
        class_count = torch.sum(segmentations == class_index)
        segmentation_distribution[class_index] += class_count


def _add_label(
    batch, label_distribution, multilabel, num_classes, target_key="targets"
):
    if target_key not in batch:
        raise ValueError(f"Batch does not contain target key '{target_key}'.")
    # Extract labels
    labels = batch[target_key]
    # if dtypes not the same, convert to int
    if labels.dtype != label_distribution.dtype:
        labels = labels.to(dtype=label_distribution.dtype)
    # Update label counts based on the configuration
    if multilabel:
        if labels.ndim != 2 or labels.shape[1] != num_classes:
            raise ValueError(
                "Expected targets to be a tensor of shape (batch_size, num_classes) for multilabel data."
            )
        # Accumulate counts for each class
        label_distribution += labels.sum(dim=0)
    else:
        if labels.ndim != 1:
            raise ValueError(
                "Expected targets to be a tensor of shape (batch_size,) for single-label data."
            )
        # Count label occurrences
        for label in labels:
            label_distribution[label.item()] += 1


def plot_distribution(
    cfg, label_distribution: torch.Tensor
) -> "plotly.graph_objects.Figure":
    # Convert the tensor to a numpy array and flatten it
    data = label_distribution.numpy().flatten()

    # Prepare the data: sort by occurrence and get corresponding labels
    sorted_indices = data.argsort()
    sorted_data = data[sorted_indices]
    sorted_data = np.round(sorted_data, 2)  # round to 2 decimal places
    labels_dict = get_index_to_name(cfg)
    labels = [labels_dict[i] for i in sorted_indices]
    scaled_sorted_data = [np.round(value, 2) * 100 for value in sorted_data]
    # Format percentages for display
    text_labels = [f"{int(np.round(value, 2))}%" for value in scaled_sorted_data]

    # Create the bar chart
    fig = go.Figure(
        data=go.Bar(
            x=labels,
            y=scaled_sorted_data,
            marker=dict(
                color=scaled_sorted_data,  # Applying a gradient colorscale
                colorscale="YlOrRd",
            ),
            text=text_labels,  # Use formatted text labels for display
            textposition="auto",
        )
    )

    # Customize layout
    range_min = 0
    range_max = int(min(100, max(scaled_sorted_data) + 10))
    fig.update_layout(
        title=f'Label Distribution for {cfg["dataset_name"]}',
        xaxis=dict(title="Labels"),
        yaxis=dict(
            title="Percent of Total Occurrences",
            range=[
                range_min,
                range_max,
            ],  # Adjust the range to be from 0 to 100 for percentage representation
            tickformat=". %",  # Format the ticks to one decimal place
            tickmode="array",  # Optional: specify tick values for clarity
            tickvals=list(
                range(range_min, range_max, 10)
            ),  # Optional: specific tick values
        ),
        xaxis_tickangle=-45,  # Rotate labels for better readability
        width=1000,  # Set width to 1000 pixels
        height=1000,  # Set height to 1000 pixels
        font=dict(size=25),  # Increase font size
    )
    return fig
