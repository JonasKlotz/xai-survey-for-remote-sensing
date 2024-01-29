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

    def create_data_for_plot(self, attrs, image_tensor, segmentations):
        """
        Create data for plot.

        Parameters
        ----------
        attrs : torch.Tensor
            Attributes tensor.
        image_tensor : torch.Tensor
            Image tensor.
        segmentations : torch.Tensor
            Segmentations tensor.

        Returns
        -------
        list
            List of data for plot.
        """
        _assert_single_image_and_attrs_shape(attrs[0], image_tensor)
        image = self._preprocess_image(image_tensor)
        attrs = [self._preprocess_attrs(attr) for attr in attrs]
        # squeeze batch dimension in attrs
        attrs = [attr.squeeze() for attr in attrs]
        attrs_titles = [f"{self.index_to_name[i]}" for i in range(len(attrs))]
        # extract attrs that are not nan
        attrs_titles = [
            title
            for title, attr in zip(attrs_titles, attrs)
            if not np.isnan(attr).all()
        ]
        attrs = [attr for attr in attrs if not np.isnan(attr).all()]
        n_attrs = len(attrs)
        data = []
        titles = []
        # Add Image and flip to fix orientation
        data.append(go.Image(z=np.flipud(image)))
        titles.append("Image")
        # Add Segmentation
        segmentation_fig = self.plot_segmentation(segmentations)
        titles.append("Segmentation")
        for trace in segmentation_fig.data:
            data.append(trace)
        # Add Attrs
        titles += attrs_titles
        for i in range(n_attrs):
            data.append(go.Heatmap(z=attrs[i], colorscale="RdBu_r"))
        return data, titles

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
            # if attrs are all zero replace with 5x5 zeros
            attrs = [np.zeros((5, 5)) if np.all(attr == 0) else attr for attr in attrs]

            data[key] = [go.Heatmap(z=attr, colorscale="RdBu_r") for attr in attrs]

        return data

    def _create_fig_from_dict_data(
        self, data: dict, labels, predictions_tensor: torch.Tensor = None, plot_title=""
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

        labels = labels.numpy()
        plots_per_attr = int(len(labels))
        cols = num_attrs + 1
        rows = max(plots_per_attr, 3)  # at least 2 rows for segmentation

        subplot_titles = self._create_subplot_titles(
            list(attrs.keys()), cols, labels, rows
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

        fig.update_layout(
            height=self.size * rows,
            width=self.size * cols,
            margin=dict(t=40, b=40),
            title_text=plot_title,
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
        label_names = [self.index_to_name[index] for index in range(len(labels))]
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

    def _create_fig_from_data(self, data, plot_title="Explanation", titles=None):
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
        # Create a subplot layout with 1 row and the number of non-empty plots
        non_empty_plots = [plot for plot in data if not isinstance(plot, go.Scatter)]
        empty_plots = [plot for plot in data if (isinstance(plot, go.Scatter))]
        # Define explicit column widths to control subplot sizes and reduce unwanted margins
        num_plots = len(non_empty_plots)

        # Define the domain for each subplot (equally divided)
        # Define the horizontal domain for each subplot (equally divided)
        domain_step = 1.0 / num_plots
        domains = [
            [i * domain_step, (i + 1) * domain_step] for i in range(num_plots)
        ]  # Slight gap between plots

        fig = make_subplots(
            rows=1,
            cols=num_plots,
        )

        # Add each plot to the subplot and update axes
        for i, plot in enumerate(non_empty_plots, start=1):
            fig.add_trace(plot, row=1, col=i)

            # Update x-axis for each subplot
            fig.update_xaxes(
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                row=1,
                col=i,
                domain=domains[i - 1],
            )
            # No need to update y-axis domain, only adjust appearance
            fig.update_yaxes(
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                row=1,
                col=i,
            )

            # Additional adjustments specifically for images
            if isinstance(plot, go.Image):
                fig.update_xaxes(scaleanchor="x", scaleratio=1, row=1, col=i)
                fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=i)

        # Add empty plots to add legends
        for i, plot in enumerate(empty_plots, start=1):
            fig.add_trace(plot, row=1, col=len(non_empty_plots))
        # Customize the layout as needed
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1)
        )

        # Add annotations for titles at the bottom of each subplot
        for i, title in enumerate(titles, start=1):
            # Calculate the midpoint of each subplot's domain
            midpoint = (domains[i - 1][0] + domains[i - 1][1]) / 2

            fig.add_annotation(
                dict(
                    text=title,  # Subplot title text
                    xref="paper",
                    yref="paper",
                    x=midpoint,  # Set x to the midpoint of the subplot's domain
                    y=-0.1,  # Adjust y position
                    showarrow=False,
                    font=dict(size=12),
                )
            )

        fig.update_layout(
            height=self.size,
            width=self.size * num_plots,
            margin=dict(t=40, b=40),
            title_text=plot_title,
        )
        return fig

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
