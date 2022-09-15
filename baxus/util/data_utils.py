from copy import deepcopy
from typing import Dict

import numpy as np


def join_data(X: np.ndarray, dims_and_bins: Dict[int, int]) -> np.ndarray:
    """
    After splitting, copy the data from the splitting dim(s) into the new dim(s)

    Args:
        X (np.ndarray): the x-values before the splitting
        dims_and_bins (Dict[int, int]): the splitting: dims and number of bins. Be warned that we assume an ordered dict
         which we are allowed to in newer Python versions.

    Returns: the x-values after splitting

    """
    X = deepcopy(X)
    for dim, bin in dims_and_bins.items():
        data_row = X[:, dim]
        X = np.hstack((X, np.tile(data_row, bin - 1).reshape(-1, (len(data_row))).T))
    return X
