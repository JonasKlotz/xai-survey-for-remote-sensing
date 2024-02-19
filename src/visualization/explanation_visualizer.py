import os
from typing import Union

import numpy as np
import plotly.graph_objects as go
import torch
import torchvision
from plotly.subplots import make_subplots

from data.constants import DEEPGLOBE_IDX2NAME


def _assert_single_image_and_attrs_shape(
    attrs: torch.Tensor, image_tensor: torch.Tensor
):
    assert len(attrs.shape) == 3, f"attrs shape is {attrs.shape}"
    assert len(image_tensor.shape) == 3, f"image_tensor shape is {image_tensor.shape}"

    assert (
        attrs.shape[-1] == image_tensor.shape[-1]
    ), f"attrs shape is {attrs.shape} and image_tensor shape is {image_tensor.shape}"
    assert (
        attrs.shape[-2] == image_tensor.shape[-2]
    ), f"attrs shape is {attrs.shape} and image_tensor shape is {image_tensor.shape}"


def _min_max_normalize(image: np.ndarray):
    if image.max() == image.min():  # if all values are the same (0)
        return image
    return (image - image.min()) / (image.max() - image.min())


def _tensor_to_numpy(image: torch.Tensor):
    if len(image.shape) == 2:  # grayscale (segmentations)
        return image.cpu().detach().numpy()
    return np.transpose(image.cpu().detach().numpy(), (1, 2, 0))


def remove_axis(fig, row, col):
    fig.update_xaxes(
        showline=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        showline=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=row,
        col=col,
    )


class ExplanationVisualizer:
    def __init__(self, cfg: dict, explanation_method_name: str, index_to_name: dict):
        """
        Initialize the ExplanationVisualizer class.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.
        explanation_method_name : str
            Name of the explanation method.
        index_to_name : dict, optional
            Dictionary mapping index to name.
        """
        self.cfg = cfg
        self.explanation_method_name = explanation_method_name
        self.index_to_name = index_to_name
        self.last_fig = None
        self.num_classes = cfg["num_classes"]
        self.save_path = (
            f"{cfg['results_path']}/visualizations/{cfg['experiment_name']}"
        )

        self.segmentation_colors = [
            "grey",
            "green",
            "lightgreen",
            "darkgreen",
            "blue",
            "yellow",
        ]

        # self.segmentation_colors = px.colors.sample_colorscale(
        #     "turbo", [n / (self.num_classes - 1) for n in range(self.num_classes)],
        # )

        self.size = 400
        # convert to uint8
        # self.segmentation_colors = [tuple(int(255 * x) for x in color) for color in self.segmentation_colors]

        # self.explanation_methods = cfg["explanation_methods"]
        # self.channels_to_visualize = cfg["channels_to_visualize"]
        # self.strategy = cfg["strategy"]

    def visualize(
        self,
        attrs: Union[torch.Tensor, dict[torch.Tensor]],
        image_tensor: torch.Tensor,
        segmentation_tensor: torch.Tensor = None,
        label_tensor: torch.Tensor = None,
        predictions_tensor: torch.Tensor = None,
        show=True,
        task: str = "multilabel",
    ):
        if task == "multilabel":
            # here we can either supply the labels or the predictions
            self.visualize_multi_label_classification(
                attrs=attrs,
                image_tensor=image_tensor,
                label_tensor=label_tensor,
                segmentation_tensor=segmentation_tensor,
                predictions_tensor=predictions_tensor,
                show=show,
            )
        else:
            self.visualize_single_label_classification(
                attrs_dict=attrs,
                image_tensor=image_tensor,
                label_tensor=label_tensor,
                segmentation_tensor=segmentation_tensor,
                predictions_tensor=predictions_tensor,
                show=show,
            )

    def visualize_batch(
        self,
        attrs: Union[torch.Tensor, dict[torch.Tensor]],
        image_tensor: torch.Tensor,
        segmentation_tensor: torch.Tensor = None,
        label_tensor: torch.Tensor = None,
        predictions_tensor: torch.Tensor = None,
        show=True,
        task: str = "multilabel",
    ):
        for i in range(len(image_tensor)):
            self.visualize(
                attrs=attrs[i],
                image_tensor=image_tensor[i],
                label_tensor=label_tensor[i],
                segmentation_tensor=segmentation_tensor[i],
                predictions_tensor=predictions_tensor[i],
                show=show,
                task=task,
            )

    def visualize_multi_label_classification(
        self,
        attrs: Union[torch.Tensor, dict[torch.Tensor]],
        image_tensor: torch.Tensor,
        segmentation_tensor: torch.Tensor = None,
        label_tensor: torch.Tensor = None,
        predictions_tensor: torch.Tensor = None,
        show=True,
    ):
        """
        Visualize multi-label classification.

        Parameters
        ----------
        predictions_tensor
        attrs : Union[torch.Tensor, dict[torch.Tensor]]
            Attributes tensor or dictionary of attributes tensors.
        image_tensor : torch.Tensor
            Image tensor.
        segmentation_tensor : torch.Tensor, optional
            Segmentations tensor.
        label_tensor : torch.Tensor, optional
            Labels tensor.
        """
        # permute image with numpy
        # the dict contains different attrs, e.g. gradcam, deeplift. each attrs is a tensor of shape (num_classes, h, w)
        if isinstance(attrs, dict):
            data = self._create_data_for_dict_plot(
                attrs, image_tensor, segmentation_tensor
            )

            fig = self._create_fig_from_dict_data(
                data, labels=label_tensor, predictions_tensor=predictions_tensor
            )

        else:
            data, titles = self.create_data_for_plot(
                attrs, image_tensor, segmentation_tensor
            )

            fig = self._create_fig_from_data(data, titles=titles)

        # Show the figure
        self.last_fig = fig

        if show:
            fig.show()

    def _create_data_for_dict_plot(self, attrs_dict: dict, image_tensor, segmentations):
        """
        Create data for dictionary plot.

        Parameters
        ----------
        attrs_dict : dict
            Dictionary of attributes tensors.
        image_tensor : torch.Tensor
            Image tensor.
        segmentations : torch.Tensor
            Segmentations tensor.

        Returns
        -------
        dict
            Dictionary of data for plot.
        """
        data = {}

        image = self._preprocess_image(image_tensor)
        # Add Image and flip to fix orientation
        data["Image"] = go.Image(z=np.flipud(image))

        # Add Segmentation
        segmentation_fig = self.plot_segmentation(segmentations)
        segmentation_list = []
        for trace in segmentation_fig.data:
            segmentation_list.append(trace)
        data["Segmentation"] = segmentation_list

        for key, attrs in attrs_dict.items():
            attrs = [self._preprocess_attrs(attr) for attr in attrs]
            # squeeze batch dimension in attrs

            attrs = [attr.squeeze() for attr in attrs]
            attrs = [attr for attr in attrs if not np.isnan(attr).all()]

            # remove np.all(attr == 0)
            attrs = [attr for attr in attrs if not np.all(attr == 0)]

            data[key] = [go.Heatmap(z=attr, colorscale="RdBu_r") for attr in attrs]

        return data

    def _create_fig_from_dict_data(
        self,
        data: dict,
        labels,
        predictions_tensor: torch.Tensor = None,
    ):
        """
        Create figure from dictionary data.

        Parameters
        ----------
        data : dict
            Dictionary of data for plot.
        labels : torch.Tensor
            Labels tensor.
        plot_title : str, optional
            Title of the plot.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure object.
        """
        # Create a subplot layout with 1 row and the number of non-empty plots

        image, segmentations = data["Image"], data["Segmentation"]
        # get all other attrs
        attrs = {
            key: value
            for key, value in data.items()
            if key not in ["Image", "Segmentation"]
        }
        num_attrs = len(attrs)

        predictions_tensor = np.nonzero(predictions_tensor.numpy())[0]
        plots_per_attr = int(len(predictions_tensor))
        cols = num_attrs + 1
        rows = max(plots_per_attr, 3)  # at least 2 rows for segmentation

        subplot_titles = self._create_subplot_titles(
            list(attrs.keys()), cols, predictions_tensor, rows
        )

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
        )
        # Add image
        fig.add_trace(image, row=1, col=1)
        remove_axis(fig, row=1, col=1)
        # Add segmentation
        fig.add_trace(segmentations[0], row=2, col=1)
        remove_axis(fig, row=2, col=1)

        # # Add each attribution to the subplot and update axes
        for col_key, (attr_name, attr_list) in enumerate(attrs.items(), start=2):
            for row_key, plot in enumerate(attr_list, start=1):
                fig.add_trace(plot, row=row_key, col=col_key)
                remove_axis(fig, row=row_key, col=col_key)

                # Add empty plots to add legends
        for i, plot in enumerate(segmentations[1:], start=1):
            fig.add_trace(plot, row=1, col=1)

            # Shift the legend to the bottom of the plot
            fig.update_layout(
                legend=dict(orientation="h", yanchor="top", y=0, xanchor="left", x=0)
            )

        labels_tensor = np.nonzero(predictions_tensor)[0]
        label_names = [self.index_to_name[label] for label in labels_tensor]
        predictions_names = [self.index_to_name[pred] for pred in predictions_tensor]

        # join labels and predictions to title
        label_names = ", ".join(label_names)
        predictions_names = ", ".join(predictions_names)

        title = f"Explanation for true label {label_names} and predicted label {predictions_names}"

        fig.update_layout(
            height=self.size * rows,
            width=self.size * cols,
            margin=dict(t=40, b=40),
            title_text=title,
        )
        return fig

    def _create_subplot_titles(self, attr_names, cols, labels, rows):
        """
        Create figure from data.

        Parameters
        ----------
        data : list
            List of data for plot.
        plot_title : str, optional
            Title of the plot.
        titles : list, optional
            List of titles.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure object.
        """
        label_names = [self.index_to_name[label] for label in labels]
        subplot_titles = [
            [""] * cols for _ in range(rows)
        ]  # Use list comprehension here

        subplot_titles[0][0] = "Image"

        subplot_titles[1][0] = "Segmentation"

        # # Add each attribution to the subplot and update axes

        for col_key, attr_name in enumerate(attr_names, start=1):
            attr_name = attr_name[:-5]  # remove .. _data
            attr_name = attr_name[2:]  # remove a_
            for row_key, label_name in enumerate(label_names):
                subplot_titles[row_key][col_key] = f"{attr_name} {label_name}"
        # flatten list
        subplot_titles = [item for sublist in subplot_titles for item in sublist]
        return subplot_titles

    def save_last_fig(self, name, format="svg"):
        """
        Save the last figure.

        Parameters
        ----------
        path : str
            Path to save the figure.
        format : str, optional
            Format to save the figure.
        """
        # create save path if it does not exist
        os.makedirs(self.save_path, exist_ok=True)
        self.last_fig.write_image(f"{self.save_path}/{name}.{format}", format=format)

    def _preprocess_image(self, image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Preprocess image.

        Parameters
        ----------
        image : Union[torch.Tensor, np.ndarray]
            Image tensor or numpy array.

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)

        image = self._denormalize(image)
        image = _min_max_normalize(image)
        image = _tensor_to_numpy(image)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = image[:, :, ::-1]  # bgr to rgb
        return image

    def _preprocess_attrs(self, attrs: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Preprocess attributes.

        Parameters
        ----------
        attrs : Union[torch.Tensor, np.ndarray]
            Attributes tensor or numpy array.

        Returns
        -------
        np.ndarray
            Preprocessed attributes.
        """
        if not isinstance(attrs, torch.Tensor):
            attrs = torch.tensor(attrs)

        # if 3 channels sum over channels
        if len(attrs.shape) == 3:
            attrs = attrs.sum(axis=0)

        attrs = _min_max_normalize(attrs)
        attrs = _tensor_to_numpy(attrs)
        return attrs

    def _denormalize(self, image: torch.tensor):
        """
        Denormalize image.

        Parameters
        ----------
        image : torch.tensor
            Image tensor.

        Returns
        -------
        torch.tensor
            Denormalized image.
        """
        mean, std = self._get_mean_and_std()
        return NormalizeInverse(mean, std)(image)

    def _get_mean_and_std(self):
        """
        Get mean and standard deviation.

        Returns
        -------
        tuple
            Mean and standard deviation.
        """
        if self.cfg["dataset_name"] == "deepglobe":
            mean = torch.tensor([0.4095, 0.3808, 0.2836])
            std = torch.tensor([0.1509, 0.1187, 0.1081])

            return mean, std
        elif self.cfg["dataset_name"] == "caltech101":
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            return mean, std

        raise NotImplementedError(f"No mean and std for {self.cfg['dataset_name']}")

    def plot_segmentation(
        self, segmentation: Union[np.ndarray], title: str = "Segmentation Map"
    ):
        """
        Plot segmentation.

        Parameters
        ----------
        segmentation : Union[np.ndarray]
            Segmentation numpy array.
        title : str, optional
            Title of the plot.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure object.
        """
        # convert to np if tensor
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().detach().numpy()
        # convert segmentation to int
        segmentation = segmentation.astype(int)
        n_colors = len(self.segmentation_colors)
        # Map the colors to the normalized positions on the scale
        colorscale = [
            (i / (n_colors - 1), color)
            for i, color in enumerate(self.segmentation_colors)
        ]
        # Create the heatmap with a fixed color scale range
        fig = go.Figure(
            data=go.Heatmap(
                z=segmentation,
                colorscale=colorscale,  # Use the custom colorscale
                zmin=0,
                zmax=5,  # Set the fixed scale range from 0 to 6
                showscale=False,  # Hide the default color scale
            )
        )

        # Add custom legend (manual approach)
        for i, color in enumerate(self.segmentation_colors, start=1):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    legendgroup=str(i),
                    showlegend=True,
                    name=self.index_to_name[i - 1],
                )
            )

        # Customize layout
        fig.update_layout(
            title=title,
            xaxis=dict(title="", showgrid=False, showticklabels=False),
            yaxis=dict(title="", showgrid=False, showticklabels=False),
        )
        # fig.show()
        return fig

    def visualize_single_label_classification(
        self,
        attrs_dict: Union[torch.Tensor, dict[torch.Tensor]],
        image_tensor: torch.Tensor,
        segmentation_tensor: torch.Tensor = None,
        label_tensor: torch.Tensor = None,
        predictions_tensor: torch.Tensor = None,
        show=True,
    ):
        data = {}

        image = self._preprocess_image(image_tensor)
        # Add Image and flip to fix orientation
        data["Image"] = go.Image(z=image)

        if segmentation_tensor is not None:
            # Add Segmentation
            segmentation_fig = self.plot_segmentation(segmentation_tensor)
            segmentation_list = []
            for trace in segmentation_fig.data:
                segmentation_list.append(trace)
            data["Segmentation"] = segmentation_list

        for key, attrs in attrs_dict.items():
            attr = self._preprocess_attrs(attrs)
            # squeeze batch dimension in attrs

            attr = attr.squeeze()

            data[key] = go.Heatmap(z=np.flipud(attr), colorscale="RdBu_r")

        num_plots = len(data.keys())
        cols = num_plots
        rows = 1
        titles = list(data.keys())

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=titles,
        )

        # Add each plot to the subplot and update axes
        for i, (key, plot) in enumerate(data.items(), start=1):
            fig.add_trace(plot, row=1, col=i)
            remove_axis(fig, row=1, col=i)

        fig.update_layout(
            height=self.size * rows,
            width=self.size * cols,
            margin=dict(t=60, b=60),
            title_text=f"Explanation for true label {label_tensor.item()} "
            f"and predicted label {predictions_tensor.item()}",
        )
        self.last_fig = fig

        # Show the figure
        if show:
            fig.show()

        return fig


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


if __name__ == "__main__":
    # load config
    import yaml

    p = "/home/jonasklotz/Studys/MASTERS/XAI/config/general_config.yml"

    with open(p, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["results_path"] = "/home/jonasklotz/Studys/MASTERS/XAI/results"
    cfg["experiment_name"] = "deepglobe"
    # test segmentation map plot
    vis = ExplanationVisualizer(cfg, "gradcam", DEEPGLOBE_IDX2NAME)
    seg = np.random.randint(0, 3, (224, 224))
    vis.plot_segmentation(seg)

    seg1 = np.random.randint(0, 4, (224, 224))
    vis.plot_segmentation(seg1)
