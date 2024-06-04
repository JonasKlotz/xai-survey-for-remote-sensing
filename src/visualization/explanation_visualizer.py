import os
from typing import Union

import numpy as np
import plotly.graph_objects as go
import torch
import torchvision
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree
from skimage import exposure
from sklearn.metrics import auc

from data.data_utils import parse_batch
from data.ben.BENv2Stats import means as all_mean
from data.ben.BENv2Stats import stds as all_std
from utility.cluster_logging import logger

rename_dict = {
    "gradcam": "GradCAM",
    "guided_gradcam": "Guided GradCAM",
    "lime": "LIME",
    "deeplift": "DeepLift",
    "integrated_gradients": "Integrated Gradients",
    "lrp": "LRP",
    "occlusion": "Occlusion",
    "Segmentation": "Segmentation",
    "Image": "Image",
}
RELU = True
zmin = 0 if RELU else -1

COLORSCALE = "OrRd"


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


def scale_tensor(x):
    """
    Scale a tensor so that:
    - Positive values are scaled between 0 and 1.
    - Negative values are scaled between -1 and 0.
    """
    # Separate positive and negative parts
    positive_part = torch.where(x > 0, x, torch.tensor(0.0))
    negative_part = torch.where(x < 0, x, torch.tensor(0.0))

    # Scale positive values between 0 and 1
    max_positive = positive_part.max()
    if max_positive > 0:
        positive_scaled = positive_part / max_positive
    else:
        positive_scaled = positive_part

    # Scale negative values between 0 and -1
    min_negative = negative_part.min()
    if min_negative < 0:
        negative_scaled = negative_part / abs(min_negative)
    else:
        negative_scaled = negative_part

    # Combine scaled positive and negative parts
    return positive_scaled + negative_scaled


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
    def __init__(self, cfg: dict, index_to_name: dict):
        """
        Initialize the ExplanationVisualizer class.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.

        index_to_name : dict, optional
            Dictionary mapping index to name.
        """
        self.cfg = cfg
        self.index_to_name = index_to_name
        self.last_fig = None
        self.num_classes = cfg["num_classes"]
        self.output_path = (
            f"{cfg['results_path']}/{cfg['experiment_name']}/visualizations"
        )

        self.segmentation_colors = [
            "grey",
            "green",
            "lightgreen",
            "darkgreen",
            "blue",
            "yellow",
        ]

        torch.set_printoptions(sci_mode=False)

        self.size = 700

    def visualize(
        self,
        attribution_dict: Union[torch.Tensor, dict[torch.Tensor]],
        image_tensor: torch.Tensor,
        segmentation_tensor: torch.Tensor = None,
        label_tensor: torch.Tensor = None,
        predictions_tensor: torch.Tensor = None,
        show=True,
        task: str = "multilabel",
        normalize=True,
        title=None,
    ):
        # assert that the label and prediction tensors are not
        if task == "multilabel":
            # here we can either supply the labels or the predictions
            self.visualize_multi_label_classification(
                attrs=attribution_dict,
                image_tensor=image_tensor,
                label_tensor=label_tensor,
                segmentation_tensor=segmentation_tensor,
                predictions_tensor=predictions_tensor,
                show=show,
                normalize=normalize,
                title=title,
            )
        else:
            self.visualize_single_label_classification(
                attrs_dict=attribution_dict,
                image_tensor=image_tensor,
                label_tensor=label_tensor,
                segmentation_tensor=segmentation_tensor,
                predictions_tensor=predictions_tensor,
                show=show,
                normalize=normalize,
                title=title,
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
                attribution_dict=attrs[i],
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
        normalize=True,
        title=None,
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

        data = self._create_data_for_dict_plot(
            attrs, image_tensor, segmentation_tensor, normalize=normalize
        )

        fig = self._create_fig_from_dict_data(
            data,
            labels=label_tensor,
            predictions_tensor=predictions_tensor,
            title=title,
        )

        # Show the figure
        self.last_fig = fig

        if show:
            fig.show()

    def _create_data_for_dict_plot(
        self,
        attrs_dict: dict,
        image_tensor,
        segmentations,
        normalize=True,
        masked_image_dict=None,
    ):
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

        if masked_image_dict:
            for key, masked_image_list in masked_image_dict.items():
                temp = []
                for i, masked_image in enumerate(masked_image_list):
                    # preprocessing is done bef
                    temp.append(go.Image(z=np.flipud(masked_image)))
                data[key] = temp

        # Add Segmentation
        if segmentations is not None:
            segmentation_fig = self.plot_segmentation(segmentations)
            segmentation_list = []
            for trace in segmentation_fig.data:
                segmentation_list.append(trace)
            data["Segmentation"] = segmentation_list

        for key, attrs in attrs_dict.items():
            attrs = [
                self._preprocess_attrs(attr, normalize=normalize) for attr in attrs
            ]
            # squeeze batch dimension in attrs

            attrs = [attr.squeeze() for attr in attrs]
            attrs = [attr for attr in attrs if not np.isnan(attr).all()]

            # remove np.all(attr == 0)
            attrs = [attr for attr in attrs if not np.all(attr == 0)]

            data[key] = [
                go.Heatmap(z=attr, colorscale=COLORSCALE, zmin=0, zmax=1)
                for attr in attrs
            ]

        return data

    def _create_fig_from_dict_data(
        self,
        data: dict,
        labels,
        predictions_tensor: torch.Tensor = None,
        title: str = None,
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

        label_names = [
            self.index_to_name[label.item()]
            for label in torch.nonzero(labels, as_tuple=True)[0]
        ]
        if predictions_tensor is not None:
            predictions_names = [
                self.index_to_name[pred.item()]
                for pred in torch.nonzero(predictions_tensor, as_tuple=True)[0]
            ]
        else:
            predictions_names = ""

        # Create a subplot layout with 1 row and the number of non-empty plots
        if title is None:
            # join labels and predictions to title
            label_names = ", ".join(label_names)
            predictions_names = ", ".join(predictions_names)
            title = f"Explanation for true label {label_names} and predicted label {predictions_names}"

        image, segmentations = data["Image"], data.get("Segmentation", None)
        # get all other attrs
        attrs = {
            key: value
            for key, value in data.items()
            if key not in ["Image", "Segmentation"]
        }
        num_attrs = len(attrs)
        if isinstance(predictions_tensor, torch.Tensor):
            predictions_tensor = predictions_tensor.numpy()
        predictions_tensor = np.nonzero(predictions_tensor)[0]
        plots_per_attr = int(len(predictions_tensor))
        cols = num_attrs + 1
        rows = max(plots_per_attr, 2)  # at least 2 rows for segmentation
        titles = list(attrs.keys())

        # titles = [title[2:-5] if title.startswith("a_") and title.endswith("_data") else title for title in titles]
        subplot_titles = self._create_subplot_titles(
            titles, cols, predictions_tensor, rows, segmentations is not None
        )

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
        )
        # Add image
        fig.add_trace(image, row=1, col=1)
        remove_axis(fig, row=1, col=1)
        if segmentations is not None:
            # Add segmentation
            fig.add_trace(segmentations[0], row=2, col=1)
            remove_axis(fig, row=2, col=1)

            # Add empty plots to add legends
            for i, plot in enumerate(segmentations[1:], start=1):
                fig.add_trace(plot, row=1, col=1)

                # Shift the legend to the bottom of the plot
                fig.update_layout(
                    legend=dict(
                        orientation="h", yanchor="top", y=0, xanchor="left", x=0
                    )
                )

        # # Add each attribution to the subplot and update axes
        for col_key, (attr_name, attr_list) in enumerate(attrs.items(), start=2):
            for row_key, plot in enumerate(attr_list, start=1):
                fig.add_trace(plot, row=row_key, col=col_key)
                remove_axis(fig, row=row_key, col=col_key)

        fig.update_layout(
            height=self.size * rows,
            width=self.size * cols,
            margin=dict(t=150, b=70),
            title_text=title,
            font=dict(
                size=35,  # Set the font size here
            ),
        )
        # Update the font size of the annotations (subtitles)
        for annotation in fig["layout"]["annotations"]:
            annotation["font"] = dict(size=40)  # Change the font size as needed

        # enforce the colorscale to be from -1 to 1
        fig.update_coloraxes(colorscale=COLORSCALE, cmin=0, cmax=1)

        fig.update_annotations(font_size=48)
        return fig

    def _create_subplot_titles(
        self, attr_names, cols, labels, rows, has_segmentation=False
    ):
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
        if has_segmentation:
            subplot_titles[1][0] = "Segmentation"

        # # Add each attribution to the subplot and update axes

        for col_key, attr_name in enumerate(attr_names, start=1):
            attr_name = attr_name[:-5]  # remove .. _data
            attr_name = attr_name[2:]  # remove a_
            for row_key, label_name in enumerate(label_names):
                subplot_titles[row_key][col_key] = f"{attr_name} {label_name}"
        # flatten list
        subplot_titles = [item for sublist in subplot_titles for item in sublist]

        rename_dict = {
            "gradcam": "GradCAM",
            "guided_gradcam": "G-GradCAM",
            "lime": "LIME",
            "deeplift": "DeepLift",
            "integrated_gradients": "IG",
            "lrp": "LRP",
            "occlusion": "Occlusion",
        }
        new_titles = []
        for i, title in enumerate(subplot_titles):
            if title == "Image" or title == "Segmentation":
                if title == "Segmentation":
                    title = "Reference Map"
                new_titles.append(title)
                continue
            # split at ' '
            titles = title.split(" ")
            if len(titles) >= 2 and not titles[0] == "k":
                method_name = titles[0]
                class_name = " ".join(titles[1:])
                method_name = rename_dict.get(method_name, method_name)
                class_name = class_name.replace("_", " ")
                new_titles.append(f"{method_name}; {class_name}")
            else:
                new_titles.append(title.replace("_land", ""))

        return new_titles

    def save_last_fig(self, name, format="svg", batch_dict=None):
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
        os.makedirs(self.output_path, exist_ok=True)
        if self.last_fig is None:
            return
        # print(f"Saving figure to {self.save_path}/{name}.{format}")
        self.last_fig.write_image(f"{self.output_path}/{name}.{format}", format=format)
        self.last_fig = None
        if batch_dict is not None:
            torch.save(batch_dict, f"{self.output_path}/batch_dict_{name}.pt")
        print(f"Saved figure to {self.output_path}/{name}.{format}")

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

        if self.cfg["dataset_name"] == "ben":
            image = exposure.equalize_hist(image, nbins=256)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = image[:, :, ::-1]  # bgr to rgb
        return image

    def _preprocess_attrs(
        self, attrs: Union[torch.Tensor, np.ndarray], normalize=True
    ) -> np.ndarray:
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
        if RELU:
            attrs = torch.nn.functional.relu(attrs)
        if normalize:
            attrs = scale_tensor(attrs)

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

        if self.cfg["dataset_name"] == "ben":
            # select RGB bands
            #  B04, B03, and B02 are the red, green, and blue (RGB) band
            image = image[[3, 2, 1]]
            # convert to torch format BGR
            image = image[[2, 1, 0]]

            # also permute the mean and std
            mean = mean[[2, 1, 0]]
            std = std[[2, 1, 0]]

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
        elif self.cfg["dataset_name"] == "ben":
            onetwenty_nearest_mean = all_mean["120_nearest"]
            onetwenty_nearest_std = all_std["120_nearest"]
            #  B04, B03, and B02 are the red, green, and blue (RGB) bands
            red_mean = onetwenty_nearest_mean["B04"]
            red_std = onetwenty_nearest_std["B04"]

            green_mean = onetwenty_nearest_mean["B03"]
            green_std = onetwenty_nearest_std["B03"]

            blue_mean = onetwenty_nearest_mean["B02"]
            blue_std = onetwenty_nearest_std["B02"]

            mean = torch.tensor([red_mean, green_mean, blue_mean])
            std = torch.tensor([red_std, green_std, blue_std])
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
                zmin=zmin,
                zmax=self.num_classes - 1,  # Set the fixed scale range from 0 to 6
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
        title="",
        normalize=True,
    ):
        data = {}

        image = self._preprocess_image(image_tensor)
        # Add Image and flip to fix orientation
        data["Image"] = go.Image(z=image)

        if segmentation_tensor is not None:
            segmentation_map = go.Heatmap(
                z=np.flipud(segmentation_tensor),
                zmin=zmin,
                zmax=1,
            )
            data["Segmentation"] = segmentation_map

        for key, attrs in attrs_dict.items():
            attr = self._preprocess_attrs(attrs)
            # squeeze batch dimension in attrs

            attr = attr.squeeze()

            data[key] = go.Heatmap(
                z=np.flipud(attr), colorscale=COLORSCALE, zmin=zmin, zmax=1
            )

        num_plots = len(data.keys())
        if num_plots > 5:
            cols = 5
            rows = int(np.ceil(num_plots / cols))
        else:
            cols = num_plots
            rows = 1

        titles = [
            "Image",
            "deeplift",
            "gradcam",
            "guided_gradcam",
            "integrated_gradients",
            "Segmentation",
            "lime",
            "lrp",
            "occlusion",
        ]
        data_keys = [
            "deeplift",
            "gradcam",
            "guided_gradcam",
            "integrated_gradients",
            "lime",
            "lrp",
            "occlusion",
        ]
        # remove a_ prefix and _data suffix in data
        image = data.pop("Image")
        segmentation = data.pop("Segmentation")
        data = {
            title: data[f"a_{title}_data"]
            for title in titles
            if f"a_{title}_data" in data
        }
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[rename_dict[title] for title in titles],
        )
        # add image
        fig.add_trace(image, row=1, col=1)
        remove_axis(fig, row=1, col=1)

        fig.add_trace(segmentation, row=2, col=1)
        remove_axis(fig, row=2, col=1)

        for row in range(1, rows + 1):
            for col in range(2, cols + 1):
                # remove first data_key
                if not data_keys:
                    continue
                value = data[data_keys.pop(0)]
                fig.add_trace(value, row=row, col=col)
                remove_axis(fig, row=row, col=col)

        if not title:
            title = f"Explanation for true label {label_tensor.item()} and predicted label {predictions_tensor.item()}"

        fig.update_layout(
            height=self.size * rows,
            width=self.size * cols,
            margin=dict(t=100, b=60),
            title_text=title,
            font=dict(
                size=30,  # Set the font size here
            ),
        )

        # enforce the colorscale to be from -1 to 1
        fig.update_coloraxes(colorscale=COLORSCALE, cmin=0, cmax=1)
        fig.update_annotations(font_size=30)

        self.last_fig = fig

        # Show the figure
        if show:
            fig.show()

        return fig

    def visualize_top_k_attributions_with_predictions(
        self,
        attribution_dict: Union[torch.Tensor, dict[torch.Tensor]],
        image_tensor: torch.Tensor,
        k_list: list,
        segmentation_tensor: torch.Tensor = None,
        label_tensor: torch.Tensor = None,
        predictions_tensor: torch.Tensor = None,
        show=True,
        model=None,
        removal_strategy="MORF",
        title="Top k attributions",
        save_name="",
    ):
        if model is None:
            raise ValueError("Model has to be supplied for this visualization")

        for attr_name, attributions in attribution_dict.items():
            (
                masked_predictions_list,
                masked_image_dict,
                attributions_dict,
            ) = self._generate_k_attr_dict(
                attr_name,
                attributions,
                image_tensor,
                k_list,
                model,
                removal_strategy=removal_strategy,
            )

            data = self._create_data_for_dict_plot(
                attributions_dict,
                image_tensor,
                segmentation_tensor,
                masked_image_dict=masked_image_dict,
                normalize=False,
            )
            fig = self._create_fig_from_dict_data(
                data,
                labels=label_tensor,
                predictions_tensor=predictions_tensor,
                title=save_name + " " + title,
            )

            # flatten masked_predictions_list
            masked_predictions_list = [
                item for sublist in masked_predictions_list for item in sublist
            ]
            # update the titles
            mp_index = 0
            for i in range(len(fig.layout.annotations)):
                if not fig.layout.annotations[i].text.startswith("k = "):
                    continue

                fig.layout.annotations[i].update(
                    text=masked_predictions_list[mp_index].replace("_land", "")
                )
                mp_index += 1

            fig.update_annotations(font_size=40)
            fig.update_layout(
                title=dict(
                    x=0.5,
                    y=0.99,
                    xanchor="center",
                    yanchor="top",
                    pad=dict(
                        t=0,
                        b=0,
                    ),
                    font=dict(
                        size=40,
                        # color='#000000'
                    ),
                )
            )

            fig.update_layout(margin=dict(l=50, r=50, t=150, b=50))

            # Show the figure
            self.last_fig = fig
            if show:
                fig.show()
            if save_name:
                self.save_last_fig(name=save_name + title, format="png")

            return fig

    def _generate_k_attr_dict(
        self, attr_name, attributions, image_tensor, k_list, model, removal_strategy
    ):
        masked_image_dict = {f"a_k = {k}_data": [] for k in k_list}
        attributions_dict = {attr_name: torch.zeros(attributions.shape)}

        new_titles_list = []
        for index, attr in enumerate(attributions):
            # if all attributions are 0, continue
            if torch.all(attributions[index] == 0):
                continue
            tmp_new_titles = []

            # here normalization has to happen before masking
            # todo: attr = _min_max_normalize(attr)
            attr = scale_tensor(attr)
            attributions_dict[attr_name][index] = attr

            for k_index, k in enumerate(k_list):
                top_k_key = f"a_k = {k}_data"
                image_k = self._get_image_k(
                    image_tensor, k, removal_strategy=removal_strategy
                )
                mask = self.get_top_k_mask(
                    attr, image_k, removal_strategy=removal_strategy
                )

                # multiply the attributions with the mask
                masked_image = image_tensor.clone() * mask
                masked_predictions, logits = model.prediction_step(
                    masked_image.unsqueeze(0).float()
                )
                pred = round(logits.squeeze()[index].item(), 2)
                tmp_new_titles.append(
                    f"k = {k}, {self.index_to_name[index]} p = {pred}"
                )

                # add the masked attributions to the dictionary
                preprocessed_image = self._preprocess_image(image_tensor.clone())
                permuted_mask = mask.permute(1, 2, 0).numpy(force=True)
                masked_image = preprocessed_image * permuted_mask
                masked_image_dict[top_k_key].append(masked_image)

            new_titles_list.append(tmp_new_titles)
        return new_titles_list, masked_image_dict, attributions_dict

    def _get_image_k(self, image_tensor, k, removal_strategy=None):
        if removal_strategy == "MORF" or removal_strategy == "LERF":
            k = 1 - k
        image_k = int(k * image_tensor.shape[1] * image_tensor.shape[2])

        return image_k

    def get_top_k_mask(self, attr, k, removal_strategy="MORF"):
        """Get the mask for the top k attributions.

        Parameters
        ----------
        attr : torch.Tensor
            Attributions tensor.
        k : int
            Number of top attributions.
        remove_top_k_features : bool, optional
            Whether to get the largest or smallest attributions.

        Returns
        -------
        torch.Tensor
            Mask for the top k attributions.
        """

        attr_shape = attr.shape
        attr = attr.flatten()

        # get the top k attributions

        if removal_strategy == "MORF":
            top_k_indices = torch.topk(attr, k=k, largest=False).indices
        elif removal_strategy == "LERF":
            top_k_indices = torch.topk(attr, k=k, largest=True).indices
        elif removal_strategy == "MORF_BASELINE":
            top_k_indices = torch.topk(attr, k=k, largest=True).indices
        else:
            raise ValueError(f"Removal strategy {removal_strategy} not recognized")
        # create a mask for the top k attributions
        mask = torch.zeros(attr_shape)
        mask = mask.flatten()
        mask[top_k_indices] = 1
        mask = mask.reshape(attr_shape)
        return mask

    def visualize_from_batch_dict(
        self,
        batch_dict,
        show=True,
        skip_non_multilabel=True,
        skip_wrong_preds=True,
        title=None,
    ):
        """

        Parameters
        ----------
        batch_dict
            From the ExplanationManager

        Returns
        -------

        """
        batch = parse_batch(batch_dict)
        batch_tensor_list = batch[:-1]
        # remove non tensors

        attributions_dict = batch[-1]
        attributions_dict = {
            k: torch.tensor(v).squeeze() for k, v in attributions_dict.items()
        }

        # unpack the batch
        (
            image_tensor,
            true_labels,
            predicted_label_tensor,
            segments_tensor,
            indices,
        ) = batch_tensor_list
        image_tensor = torch.tensor(image_tensor).squeeze()
        true_labels = torch.tensor(true_labels).squeeze()
        if segments_tensor is not None:
            segments_tensor = torch.tensor(segments_tensor).squeeze()
        predicted_label_tensor = torch.tensor(predicted_label_tensor).squeeze()

        if skip_non_multilabel and torch.sum(predicted_label_tensor) <= 1:
            logger.debug(f"Skipping {indices} as it is not multilabel")
            return
        if skip_wrong_preds and not torch.all(predicted_label_tensor == true_labels):
            logger.debug(f"Skipping {indices} as it is a wrong prediction")
            return

        self.visualize(
            attribution_dict=attributions_dict,
            image_tensor=image_tensor,
            label_tensor=true_labels,
            segmentation_tensor=segments_tensor,
            predictions_tensor=predicted_label_tensor,
            show=show,
            task=self.cfg["task"],
            normalize=True,
            title=title,
        )

    def visualize_image(self, batch_dict, show=True):
        batch = parse_batch(batch_dict)
        (
            image_tensor,
            true_labels,
            predicted_label_tensor,
            segments_tensor,
            indices,
            attributions_dict,
        ) = batch

        for i in range(len(image_tensor)):
            image = image_tensor[i]
            label = true_labels[i]
            image = torch.tensor(image).squeeze()
            image = self._preprocess_image(image)
            fig = go.Figure(go.Image(z=image))
            fig.update_layout(
                title=f"True label: {label}",
                xaxis=dict(title="", showgrid=False, showticklabels=False),
                yaxis=dict(title="", showgrid=False, showticklabels=False),
            )
            if show:
                fig.show()
            self.last_fig = fig

    def visualize_top_k_predictions_function(
        self,
        attribution_dict: Union[torch.Tensor, dict[torch.Tensor]],
        image_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
        k_list: list,
        model=None,
        removal_strategy="MORF",
        title="Top k attributions",
        save_name="",
        show=True,
    ):
        if model is None:
            raise ValueError("Model has to be supplied for this visualization")
        label_names = [
            self.index_to_name[label.item()]
            for label in torch.nonzero(label_tensor, as_tuple=True)[0]
        ]
        for attr_name, attributions in attribution_dict.items():
            k_prediction_dict_dict = {}

            for index, attr in enumerate(attributions):
                # if all attributions are 0, continue
                if torch.all(attributions[index] == 0):
                    continue
                k_prediction_dict = {}
                label = label_names[index]

                attr = scale_tensor(attr)

                for k_index, k in enumerate(k_list):
                    image_k = self._get_image_k(
                        image_tensor, k, removal_strategy=removal_strategy
                    )
                    mask = self.get_top_k_mask(
                        attr, image_k, removal_strategy=removal_strategy
                    )
                    # calculate how many pixels are removed
                    # sum mask
                    masked_image = image_tensor.clone() * mask
                    masked_predictions, logits = model.prediction_step(
                        masked_image.unsqueeze(0).float()
                    )
                    pred = round(logits.squeeze()[index].item(), 2)
                    k_prediction_dict[k] = pred
                k_prediction_dict_dict[f"{label}"] = k_prediction_dict

            self.plot_predictions(k_prediction_dict_dict, title, save_name, show=show)

    def plot_predictions(self, k_prediction_dict_dict, title, save_name, show=True):
        # Number of subplots
        num_plots = len(k_prediction_dict_dict)

        # Create a subplot figure
        fig = make_subplots(
            rows=1, cols=num_plots, subplot_titles=list(k_prediction_dict_dict.keys())
        )

        for i, (subplot_title, k_prediction_dict) in enumerate(
            k_prediction_dict_dict.items()
        ):
            # Create lists for x and y axis
            x = list(k_prediction_dict.keys())
            y = list(k_prediction_dict.values())

            # Calculate the AUC
            auc_value = auc(x, y)
            subplot_title_with_auc = (
                f"{subplot_title.replace("_"," ")} (AUC = {auc_value:.2f})"
            )
            k_value_in_percent = [int(k * 100) for k in x]
            # Create the line plot
            fig.add_trace(
                go.Scatter(
                    x=k_value_in_percent,
                    y=y,
                    mode="lines+markers",
                    line=dict(color="royalblue", width=2),
                    marker=dict(color="red", size=5),
                    showlegend=False,
                ),
                row=1,
                col=i + 1,
            )

            # Update the subplot title with AUC
            fig.update_xaxes(
                title_text="K Pixels affected in %",
                row=1,
                col=i + 1,
                range=[0, 100],
                showline=True,
                linecolor="black",
                mirror=True,
            )
            fig.update_yaxes(
                title_text="Prediction Confidence",
                row=1,
                col=i + 1,
                range=[0, 1],
                showline=True,
                linecolor="black",
                mirror=True,
            )
            fig.layout.annotations[i].update(text=subplot_title_with_auc, yshift=10)

        # Set the overall title and improve layout
        fig.update_layout(
            title="",
            title_font=dict(size=22, family="Arial, bold", color="black"),
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40),
            width=500 * num_plots,
            height=500,
        )
        fig.update_annotations(font_size=22)

        self.last_fig = fig
        if show:
            fig.show()
        if save_name:
            self.save_last_fig(name=save_name + title, format="png")


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


def calculate_avg_point_density(heatmap, radius=5):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy(force=True)
    non_zero_points = np.argwhere(heatmap)

    # Build a KD-tree for efficient nearest-neighbor queries
    tree = cKDTree(non_zero_points)

    # Count points within the radius for each non-zero point
    density_counts = tree.query_ball_point(non_zero_points, r=radius, p=2)

    # Calculate density for each point (subtract 1 to exclude the point itself)
    point_density = [len(count) - 1 for count in density_counts]

    # Calculate average density
    average_density = np.mean(point_density)

    return average_density
