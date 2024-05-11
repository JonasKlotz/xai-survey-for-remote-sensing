from typing import Union

import numpy as np
import plotly
import plotly.graph_objects as go
import torch

from data.zarr_handler import load_batches
from utility.cluster_logging import logger
from xai.metrics.metrics_utiliies import get_colors
from data.data_utils import get_index_to_name


def plot_dataset_distribution_zarr(cfg: dict):
    logger.debug("Loading batches as zarr")
    all_zarrs = load_batches(cfg)

    y_batch = all_zarrs["y_batch"]

    # convert zarr to numpy
    y_batch = y_batch[:]

    # convert to int
    y_batch = y_batch.astype(int)
    # sum over batch
    label_distribution = y_batch.sum(axis=0)
    label_distribution_tensor = torch.from_numpy(label_distribution)
    plot_distribution(cfg, label_distribution_tensor)

    pass


def plot_pixel_distribution_zarr(cfg):
    logger.debug("Loading batches as zarr")
    all_zarrs = load_batches(cfg)

    s_batch = all_zarrs["s_batch"][:]
    plot_pixel_distribution(cfg, s_batch)


def plot_distribution(
    cfg, label_distribution: torch.Tensor
) -> "plotly.graph_objects.Figure":
    # Convert the tensor to a numpy array and flatten it
    data = label_distribution.numpy().flatten()

    # Prepare the data: sort by occurrence and get corresponding labels
    sorted_indices = data.argsort()
    sorted_data = data[sorted_indices]
    labels = [get_index_to_name(cfg) for i in sorted_indices]

    # get and sort colors
    colors = get_colors(n_colors=len(labels))
    sorted_colors = [colors[i] for i in sorted_indices]

    # Create the bar chart
    fig = go.Figure(
        data=go.Bar(
            x=labels,
            y=sorted_data,
            marker_color=sorted_colors,  # You can customize the color
        )
    )

    # Customize layout
    fig.update_layout(
        title=f'Label Distribution for {cfg["dataset_name"]}',
        xaxis=dict(title="Labels"),
        yaxis=dict(title="Occurrences"),
    )
    return fig


def plot_pixel_distribution(cfg, s_batch: Union[np.ndarray, torch.Tensor]):
    if isinstance(s_batch, torch.Tensor):
        s_batch = s_batch.numpy()
    # convert to int
    s_batch = s_batch.astype(int)
    unique, counts = np.unique(s_batch, return_counts=True)
    plot_distribution(cfg, torch.from_numpy(counts))
