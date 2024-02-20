from typing import Union, List

import numpy as np
import plotly.express as px


def custom_aggregation_function(input_: Union[list, np.ndarray, dict]) -> float:
    """Aggregates the input to a single value.

    Parameters
    ----------
    input_: Union[list, np.ndarray, dict]
        The input to aggregate.

    Returns
    -------
    aggregated: float
        The aggregated value.
    """
    if isinstance(input_, list):
        return np.mean(
            [
                custom_aggregation_function(value)
                for value in input_
                if value is not None
            ]
        )

    elif isinstance(input_, np.ndarray):
        return np.mean(input_)

    elif isinstance(input_, dict):
        aggregated = 0
        for key, value in input_.items():
            aggregated = custom_aggregation_function(value)
        return aggregated

    return input_


def custom_aggregation_function_mlc(
    input_: Union[list, np.ndarray, dict], first_recursion=True
) -> Union[float, list]:
    """Aggregates the input to a single value.

    Parameters
    ----------
    input_: Union[list, np.ndarray, dict]
        The input to aggregate.

    Returns
    -------
    aggregated: float
        The aggregated value.
    """
    """
    Aggregations:
    - Region Perturbation: (batchsize x regions_evaluation) 
    """
    if first_recursion and isinstance(input_, list):
        return [
            [
                custom_aggregation_function_mlc(label, False)
                for label in sample
                if label is not None
            ]
            for sample in input_
            if sample is not None
        ]
    elif isinstance(input_, list):
        return np.mean(
            [
                custom_aggregation_function_mlc(value, False)
                for value in input_
                if value is not None
            ]
        )

    elif isinstance(input_, np.ndarray):
        return np.mean(input_)

    elif isinstance(input_, dict):
        aggregated = 0
        for key, value in input_.items():
            aggregated = custom_aggregation_function_mlc(value, False)
        return aggregated

    return input_


def aggregate_continuity_metric(input_: List[dict]) -> float:
    """Aggregates the continuity metric.

    Parameters
    ----------
    input_: List[dict]
        The input to aggregate.

    Returns
    -------
    aggregated: float
        The aggregated value.
    """
    aggregated = 0
    for input_dict in input_:
        for key, value in input_dict.items():
            aggregated += custom_aggregation_function(value)
    return aggregated


def get_colors(n_colors: int) -> list:
    """Returns a list of colors.

    Parameters
    ----------
    n_colors: int
        The number of colors to return.

    Returns
    -------
    colors: list
        The list of colors.
    """
    return px.colors.sample_colorscale(
        "turbo",
        [n / (n_colors - 1) for n in range(n_colors)],
    )
