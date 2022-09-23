from copy import deepcopy
from typing import Dict, Sequence

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


def right_pad_sequence(sequence: Sequence[np.ndarray], dtype=np.float64, fill_value: float = 0.0) -> np.ndarray:
    """
    Pads a sequence of 1D NumPy arrays to the same length.

    Args:
        sequence: sequence of 1D NumPy arrays
        dtype: the dtype of the result matrix
        fill_value: the value for the padding

    Returns: a matrix of shape (len(sequence), max_sequence_length) where all rows are filled up with fill_value on the right

    """
    max_len = max(len(s) for s in sequence)
    padded_matrix = np.full(shape=(len(sequence), max_len), dtype=dtype, fill_value=fill_value)
    for i, seq in enumerate(sequence):
        assert seq.ndim == 1, "Only 1D arrays are supported"
        padded_matrix[i, 0:len(seq)] = seq
    return padded_matrix
