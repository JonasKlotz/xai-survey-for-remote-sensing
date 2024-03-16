import copy
import os
from typing import Union

import numpy as np
import plotly.graph_objects as go
import torch
import torchvision
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree


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
        torch.set_printoptions(sci_mode=False)

        self.size = 400
        # convert to uint8
        # self.segmentation_colors = [tuple(int(255 * x) for x in color) for color in self.segmentation_colors]

        # self.explanation_methods = cfg["explanation_methods"]
        # self.channels_to_visualize = cfg["channels_to_visualize"]
        # self.strategy = cfg["strategy"]

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
        title="",
    ):
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
        title="",
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
        self, attrs_dict: dict, image_tensor, segmentations, normalize=True
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

        # Add Segmentation
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
                go.Heatmap(z=attr, colorscale="RdBu_r", zmin=0, zmax=1)
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
        # Create a subplot layout with 1 row and the number of non-empty plots

        image, segmentations = data["Image"], data["Segmentation"]
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

        label_names = [self.index_to_name[label.item()] for label in labels]
        predictions_names = [
            self.index_to_name[pred.item()] for pred in predictions_tensor
        ]

        if title is None:
            # join labels and predictions to title
            label_names = ", ".join(label_names)
            predictions_names = ", ".join(predictions_names)
            title = f"Explanation for true label {label_names} and predicted label {predictions_names}"

        fig.update_layout(
            height=self.size * rows,
            width=self.size * cols,
            margin=dict(t=40, b=40),
            title_text=title,
            font=dict(
                size=20,  # Set the font size here
            ),
        )
        fig.update_annotations(font_size=15)
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
        print(f"Saving figure to {self.save_path}/{name}.{format}")
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
        if normalize:
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
                zmin=0,
                zmax=1,
            )
            data["Segmentation"] = segmentation_map

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

        if not title:
            title = f"Explanation for true label {label_tensor.item()} and predicted label {predictions_tensor.item()}"

        fig.update_layout(
            height=self.size * rows,
            width=self.size * cols,
            margin=dict(t=60, b=60),
            title_text=title,
        )
        self.last_fig = fig

        # Show the figure
        if show:
            fig.show()

        return fig

    def visualize_top_k_attributions(
        self,
        attribution_dict: Union[torch.Tensor, dict[torch.Tensor]],
        image_tensor: torch.Tensor,
        k: Union[int, float] = 0.1,
        segmentation_tensor: torch.Tensor = None,
        label_tensor: torch.Tensor = None,
        predictions_tensor: torch.Tensor = None,
        show=True,
        task: str = "multilabel",
        largest=True,
    ):
        k_image = self._get_image_k(image_tensor, k, largest)

        tmp_attribution_dict = copy.deepcopy(attribution_dict)

        for attr_name, attributions in tmp_attribution_dict.items():
            for index, attr in enumerate(attributions):
                # if all attributions are 0, continue
                if torch.all(attributions[index] == 0):
                    continue

                # here normalization has to happen before masking
                attr = _min_max_normalize(attr)

                mask = self.get_top_k_mask(attr, k_image, largest)
                # multiply the attributions with the mask

                masked_attr = attr * mask
                # add the masked attributions to the dictionary
                tmp_attribution_dict[attr_name][index] = masked_attr

        for attr_name, attributions in tmp_attribution_dict.items():
            for index, attr in enumerate(attributions):
                # if all attributions are 0, continue
                if torch.all(attr == 0):
                    continue
                # calculate the average point density
                avg_point_density = calculate_avg_point_density(attr)
                print(
                    f"Average point density for {attr_name} {index}: {avg_point_density}"
                )

        self.visualize(
            attribution_dict=tmp_attribution_dict,
            image_tensor=image_tensor,
            label_tensor=label_tensor,
            segmentation_tensor=segmentation_tensor,
            predictions_tensor=predictions_tensor,
            show=show,
            task=task,
            normalize=False,
        )
        print("Visualized top k attributions")

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
        remove_top_k_features=True,
        title="Top k attributions",
        save_name="",
    ):
        if model is None:
            raise ValueError("Model has to be supplied for this visualization")

        for attr_name, attributions in attribution_dict.items():
            print(attr_name)
            masked_predictions_list, tmp_attribution_dict = self._generate_k_attr_dict(
                attr_name,
                attributions,
                image_tensor,
                k_list,
                model,
                remove_top_k_features,
            )

            data = self._create_data_for_dict_plot(
                tmp_attribution_dict, image_tensor, segmentation_tensor, normalize=False
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

                fig.layout.annotations[i].update(text=masked_predictions_list[mp_index])
                mp_index += 1

            fig.update_annotations(font_size=18)
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
                        size=25,
                        # color='#000000'
                    ),
                )
            )

            fig.update_layout(margin=dict(l=50, r=50, t=80, b=50))

            # Show the figure
            self.last_fig = fig
            if show:
                fig.show()
            if save_name:
                self.save_last_fig(name=save_name + title, format="svg")

            return fig

    def _generate_k_attr_dict(
        self,
        attr_name,
        attributions,
        image_tensor,
        k_list,
        model,
        remove_top_k_features,
    ):
        tmp_attribution_dict = {attr_name: torch.zeros(attributions.shape)}
        # populate dictionary with zeros
        for k in k_list:
            top_k_key = f"a_k = {k}_data"
            tmp_attribution_dict[top_k_key] = torch.zeros(attributions.shape)

        new_titles_list = []
        for index, attr in enumerate(attributions):
            # if all attributions are 0, continue
            if torch.all(attributions[index] == 0):
                continue
            tmp_new_titles = []

            # here normalization has to happen before masking
            attr = _min_max_normalize(attr)
            tmp_attribution_dict[attr_name][index] = attr

            for k_index, k in enumerate(k_list):
                top_k_key = f"a_k = {k}_data"
                image_k = self._get_image_k(image_tensor, k, remove_top_k_features)
                mask = self.get_top_k_mask(attr, image_k, remove_top_k_features)

                # multiply the attributions with the mask
                masked_attr = attr * mask

                masked_image = image_tensor.clone() * mask
                masked_predictions, logits = model.prediction_step(
                    masked_image.unsqueeze(0).double()
                )
                pred = round(logits.squeeze()[index].item(), 2)
                tmp_new_titles.append(
                    f"k = {k}, {self.index_to_name[index]} p = {pred}"
                )

                # add the masked attributions to the dictionary
                tmp_attribution_dict[top_k_key][index] = masked_attr

            new_titles_list.append(tmp_new_titles)
        return new_titles_list, tmp_attribution_dict

    def _get_image_k(self, image_tensor, k, reverse=False):
        if isinstance(k, float):
            if reverse:
                k = 1 - k
            image_k = int(k * image_tensor.shape[1] * image_tensor.shape[2])
        else:
            image_k = k
        return image_k

    def get_top_k_mask(self, attr, k, remove_top_k_features):
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
        top_k_indices = torch.topk(attr, k=k, largest=not remove_top_k_features).indices
        # create a mask for the top k attributions
        mask = torch.zeros(attr_shape)
        mask = mask.flatten()
        mask[top_k_indices] = 1
        mask = mask.reshape(attr_shape)
        return mask


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
