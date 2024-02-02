from typing import Union

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
    """
    Aggregations:
    - Region Perturbation: (batchsize x regions_evaluation) 
    """
    if isinstance(input_, list):
        aggregated = np.mean([np.mean(a) for a in input_])

    elif isinstance(input_, np.ndarray):
        aggregated = np.mean(input_)

    elif isinstance(input_, dict):
        aggregated = 0
        for key, value in input_.items():
            aggregated = custom_aggregation_function(value)
    else:
        raise ValueError(f"Input type {type(input_)} not supported.")
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
