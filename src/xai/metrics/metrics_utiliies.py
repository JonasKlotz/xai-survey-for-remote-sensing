from typing import Union

import numpy as np


def aggregation_function(input_: Union[list, np.ndarray, dict]) -> float:
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
    if isinstance(input_, list) or isinstance(input_, np.ndarray):
        aggregated = np.mean(input_)
    elif isinstance(input_, dict):
        aggregated = 0
        for key, value in input_.items():
            aggregated = aggregation_function(value)
    else:
        raise ValueError(f"Input type {type(input_)} not supported.")
    return aggregated
