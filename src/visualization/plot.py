import numpy as np
from earthpy import plot as ep
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
from enum import Enum


class Stage(Enum):
    TRAIN = "fit"
    VAL = "validate"
    TEST = "test"


def quant_norm_data(
    data: np.ndarray, lower_quant: float = 0.01, upper_quant: float = 0.99
) -> np.ndarray:
    """
    Normalize the data by quantiles `lower_quant/upper_quant`.
    The quantiles are calculated globally/*across all channels*.
    """
    masked_data = np.ma.masked_equal(data, 0)
    lq, uq = np.quantile(masked_data.compressed(), (lower_quant, upper_quant))
    data = np.clip(data, a_min=lq, a_max=uq)
    data = (data - lq) / (uq - lq)
    return data


def plot_rgb(img, title="", rgb=(3, 2, 1), quant_norm=True, **kwargs):
    """
    Arguments:

    arr : numpy array
        An n-dimensional array in rasterio band order (bands, rows, columns)
        containing the layers to plot.

    rgb : list (default = (3, 2, 1))
        Indices of the three bands to be plotted.
    """
    if quant_norm:
        img = quant_norm_data(img)  # enhance contrast and clip values

    rgb = ep.plot_rgb(arr=img, rgb=rgb, figsize=(20, 10), title=title, **kwargs)

    plt.show()
    return rgb


def plot_hist(img, title="", **kwargs):
    """
    Arguments:

    arr : numpy array
        An n-dimensional array in rasterio band order (bands, rows, columns)
        containing the layers to plot.

    rgb : list (default = (3, 2, 1))
        Indices of the three bands to be plotted.
    """
    hist = ep.hist(arr=img, figsize=(20, 10), title=title, **kwargs)
    plt.show()
    return hist


def plot_bands(img, title="", quant_norm=True, **kwargs):
    """
    Arguments:

    arr : numpy array
        An n-dimensional array in rasterio band order (bands, rows, columns)
        containing the layers to plot.

    rgb : list (default = (3, 2, 1))
        Indices of the three bands to be plotted.
    """
    if quant_norm:
        img = quant_norm_data(img)  # enhance contrast and clip values
    bands = ep.plot_bands(arr=img, figsize=(20, 10), title=title, **kwargs)
    plt.show()
    return bands


def plot_distribution_from_dataloader(data_loader, save_path=None, sort=True, **kwargs):
    """
    Plot the distribution of the labels in the dataloader
    """
    num_samples = 0
    label_names = data_loader.dataset.classes
    num_classes = len(label_names)
    labels_sum = torch.zeros(num_classes)
    for batch in data_loader:
        if isinstance(batch, dict):
            labels = batch["label"]
        else:
            labels = batch[1]
        counts = torch.bincount(labels, minlength=num_classes)
        labels_sum += counts
        num_samples += labels.shape[0]

    fig, ax = plt.subplots(figsize=(20, 5), dpi=300)

    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title(f"Number of samples per class (total: {num_samples})")

    cmap = cm.get_cmap("Blues", 128)
    colors = cmap(np.linspace(0.25, 1, num_classes))

    if sort:
        labels_sum, label_names = zip(
            *sorted(zip(labels_sum, label_names), reverse=True)
        )
    bars = ax.bar(label_names, labels_sum, color=colors, **kwargs)
    ax.bar_label(bars)

    plt.show()

    if save_path:
        fig.savefig(save_path)


def plot_distribution_from_datamodule(
    data_module, loader: Stage = Stage.TRAIN, **kwargs
):
    """
    Plot the distribution of the labels in the dataloader
    """
    if not isinstance(loader, Stage):
        raise TypeError(f"loader must be of type {Stage}, not {type(loader)}")

    data_module.setup(loader.value)
    plot_distribution_from_dataloader(data_module.train_dataloader(), **kwargs)
