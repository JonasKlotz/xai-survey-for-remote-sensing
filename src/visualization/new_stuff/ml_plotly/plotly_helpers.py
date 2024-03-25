from typing import Tuple, List
from plotly.subplots import make_subplots

from visualization.ml_plotly.plotly_objects import (
    PlotlyBaseObject,
    SegmentationObject,
    AttributionObject,
)


def plot_square_object_array(
    object_array: List[List[PlotlyBaseObject]],
    title: str = None,
    figsize: Tuple[int, int] = (15, 15),
    **kwargs,
):
    """
    Plot an array of plotly objects.
    Parameters
    ----------
    object_array : List[List[PlotlyBaseObject]]
        Array of plotly objects.
    title : str
        Title of the plot.
    figsize : Tuple[int, int]
        Size of the plot.
    kwargs
        Additional arguments to pass to the plot
    Returns
    -------
    plotly.Figure
        Plotly figure.
    """
    _assert_content(object_array)
    n_rows = len(object_array)
    n_cols = len(object_array[0])

    subplot_titles = get_subplot_titles_from_obj_arr(object_array)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
    )

    fig = add_objects_to_fig(fig, object_array)

    fig.update_layout(
        # height=figsize[0] * n_rows,
        # width=figsize[1] * n_cols,
        margin=dict(t=40, b=40),
        title_text=title,
        font=dict(
            size=20,  # Set the font size here
        ),
    )
    fig.update_annotations(font_size=15)
    fig.show()
    return fig


def add_objects_to_fig(
    fig, object_array: List[List[PlotlyBaseObject]], remove_axis_for_plot: bool = True
):
    for row_key, object_list in enumerate(object_array, start=1):
        for col_key, object in enumerate(object_list, start=1):
            traces = object.get_traces()
            for trace in traces:
                fig.add_trace(trace, row=row_key, col=col_key)
            if isinstance(object, SegmentationObject) or isinstance(
                object, AttributionObject
            ):
                fig.update_yaxes(autorange="reversed", row=row_key, col=col_key)

            # Set the aspect mode and ratio for each subplot
            # Set the background color for each subplot
            fig.update_xaxes(scaleanchor="y", scaleratio=1, row=row_key, col=col_key)
            fig.update_yaxes(scaleanchor="x", scaleratio=1, row=row_key, col=col_key)
            fig.update_layout(
                plot_bgcolor="white",
            )

            if remove_axis_for_plot:
                remove_axis(fig, row=row_key, col=col_key)
    return fig


def get_subplot_titles_from_obj_arr(object_array: List[List[PlotlyBaseObject]]):
    titles = []
    for i, row in enumerate(object_array):
        for j, obj in enumerate(row):
            title = obj.get_title()
            titles.append(title)
    return titles


def _assert_content(object_array):
    assert isinstance(object_array, list), "object_array must be a list"
    assert isinstance(object_array[0], list), "object_array must be a list of lists"
    assert isinstance(
        object_array[0][0], PlotlyBaseObject
    ), "object_array must be a list of PlotlyBaseObject"
    assert len(object_array) == len(
        object_array[0]
    ), "object_array must be a square array"


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
