###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

# Derived from the TuRBO implementation (https://github.com/uber-research/TuRBO)
# Author: anonymous

import argparse
from logging import warning

try:
    from collections.abc import Iterator
except ImportError as e:
    warning("Failed to import Iterator from collections.abc. Python < 3.10 won't be supported in the future.")
    from collections import Iterator

try:
    from reprlib import repr
except ImportError:
    pass
import functools

import numpy as np
import seaborn as sns


def to_unit_cube(x: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
    """
    Project to [0, 1]^d from hypercube with bounds lb and ub

    Args:
        x: the points to scale
        lower_bounds: the lower bounds in the unscaled space
        upper_bounds: the upper bounds un the unscaled space

    Returns: scaled points

    """
    assert lower_bounds.ndim == 1 and upper_bounds.ndim == 1 and x.ndim == 2
    xx = (x - lower_bounds) / (upper_bounds - lower_bounds)
    return xx


def to_1_around_origin(x: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
    """
    Project to [-1, 1]^d from hypercube with bounds lb and ub

    Args:
        x: the points to scale
        lower_bounds: the lower bounds in the unscaled space
        upper_bounds: the upper bounds un the unscaled space

    Returns: the scaled points.

    """
    assert lower_bounds.ndim == 1 and upper_bounds.ndim == 1 and x.ndim == 2
    x = to_unit_cube(x, lower_bounds, upper_bounds)
    xx = x * 2 - 1
    return xx


def from_unit_cube(x: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
    """
    Project points that were scaled to unit cube back to full space.

    Args:
        x: the points
        lower_bounds: the lower bounds of the full space
        upper_bounds: the upper bounds of the full space

    Returns: scaled points

    """
    assert lower_bounds.ndim == 1 and upper_bounds.ndim == 1 and x.ndim == 2
    xx = x * (upper_bounds - lower_bounds) + lower_bounds
    return xx


def from_1_around_origin(x: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
    """
    Project points that were scaled to one-around-origin cube back to full space.

    Args:
        x: the points
        lower_bounds: the lower bounds of the full space
        upper_bounds: the upper bounds of the full space

    Returns: scaled points

    """
    xx = (x + 1) / 2
    return from_unit_cube(xx, lower_bounds, upper_bounds)


def one_around_origin_latin_hypercube(n_pts: int, dim: int) -> np.ndarray:
    """
    Basic Latin hypercube implementation with center perturbation in a one-around-origin cube.

    Args:
        n_pts: number of points to sample
        dim: dimensionality of the space

    Returns: the LHS points

    """
    X = latin_hypercube(n_pts=n_pts, dim=dim)
    return X * 2 - 1


def latin_hypercube(n_pts: int, dim: int) -> np.ndarray:
    """
    Basic Latin hypercube implementation with center perturbation.

    Args:
        n_pts: number of points to sample
        dim: dimensionality of the space

    Returns: the LHS points

    """
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locations for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X


def str2bool(value: str) -> bool:
    """
    Parse string to boolean or throw error if string has no boolean type.

    Args:
        value: the string to parse

    Returns: True, if string is truthy, false if string is falsy

    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class ColorIterator(Iterator):
    """
    A color iterator
    """
    colors = sns.color_palette("husl", 23)

    def __init__(self):
        self.cc = 0

    def __iter__(self):
        return self

    def __next__(self):
        c = self.colors[self.cc % 23]
        self.cc += 1
        return c


def in_range(x: np.ndarray, incumbent: np.ndarray, lb: np.ndarray, ub: np.ndarray):
    """
    Whether the point x is within the range of the next slower trust region around incumbent
    given the current bounds lb, ub

    Args:
        x: the point to test
        incumbent: the point to center the next smallest TR around
        lb: lower bound of the current trust region
        ub: upper bound of the current trust region

    Returns: true if point would fall in the next smaller trust region, false otherwise

    """
    offsets = [(ub.squeeze()[i] - lb.squeeze()[i]) / 4 for i in range(len(incumbent))]
    return all(
        incumbent[i] - offsets[i] < x[i] < incumbent[i] + offsets[i]
        for i in range(len(incumbent))
    )


def star_string(wrap_string: str) -> str:
    """
    Wrap string in stars.

    Args:
        wrap_string: string to wrap

    Returns: wrapped string

    """
    return f"{''.join(['*'] * (len(wrap_string) + 4))}\n* {wrap_string} *\n{''.join(['*'] * (len(wrap_string) + 4))}"


def partialclass(cls, *args, **kwargs):
    """
    A partially initialized class

    Args:
        cls: the base class
        *args:
        **kwargs:

    Returns:

    """

    class PartialClass(cls):
        __init__ = functools.partial(cls.__init__, *args, **kwargs)

    return PartialClass