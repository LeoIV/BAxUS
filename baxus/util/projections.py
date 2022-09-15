from abc import ABC
from copy import deepcopy
from logging import warning, info, debug
from typing import Optional, Dict, List

import numpy as np
from numpy.random import RandomState

from baxus.util.behaviors.embedding_configuration import EmbeddingType
from baxus.util.exceptions import OutOfBoundsException, UnknownBehaviorError


class ProjectionModel(ABC):
    def project_up(self, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def project_down(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class IdentityProjector(ProjectionModel):
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def project_up(self, Y: np.ndarray) -> np.ndarray:
        return Y

    def project_down(self, X: np.ndarray) -> np.ndarray:
        return X


class AxUS(ProjectionModel):
    """
    AxUS embedding. Also support HeSBO embedding by choosing RANDOM bin sizing
    """

    def __init__(
            self,
            input_dim: int,
            target_dim: int,
            seed: Optional[int] = None,
            bin_sizing=EmbeddingType.BAXUS,
    ):
        self.seed = seed
        self.target_dim: int = target_dim
        self.input_dim: int = input_dim
        self.bin_sizing = bin_sizing
        self._S = None
        self._random_state = RandomState(self.seed)
        self._reset()

    def _target_to_input_dim(
            self, input_to_target_dim_h: Dict[int, int]
    ) -> Dict[int, List[int]]:
        """
        Revert the input to target dim mapping
        :param input_to_target_dim_h:
        :return: the target to input dim mapping
        """
        input_to_target_dim_h = deepcopy(input_to_target_dim_h)
        target_to_input_dim: Dict[int, List[int]] = {
            i: [] for i in range(self.target_dim)
        }
        for k, v in input_to_target_dim_h.items():
            target_to_input_dim[v].append(k)
        return target_to_input_dim

    def _input_to_target_dim(
            self, target_to_input_dim: Dict[int, List[int]]
    ) -> Dict[int, int]:
        """
        Revert the target to input dim mapping
        :param target_to_input_dim:
        :return: the input to target dim mapping
        """
        target_to_input_dim = deepcopy(target_to_input_dim)
        input_to_target_dim = {
            i: [k for k, v in target_to_input_dim.items() if i in v][0]
            for i in range(self.input_dim)
        }
        return input_to_target_dim

    def _reset(self):
        """
        Reset the AxUS embedding. Sample a new AxUS embedding.
        :return:
        """

        if self.target_dim > self.input_dim:
            warning(
                "HeSBO: Got a target dim larger than the input dim. Setting target dim to input dim."
            )
            self.target_dim = self.input_dim
        if self.target_dim == self.input_dim:
            info("HeSBO: Target dim = input dim. Using identity mapping.")
            _input_to_target_dim_h: Dict[int, int] = {
                i: i for i in range(self.input_dim)
            }
        else:
            if self.bin_sizing == EmbeddingType.BAXUS:
                debug("Creating uniform HeSBO embedding.")
                input_dim_permutation = np.random.permutation(
                    list(range(self.input_dim))
                )
                input_dim_bins = np.array_split(input_dim_permutation, self.target_dim)
                _target_to_input_dim_h: Dict[int, List[int]] = {
                    input_dim_nr: input_dim_bin
                    for input_dim_nr, input_dim_bin in enumerate(input_dim_bins)
                }
                _input_to_target_dim_h = self._input_to_target_dim(
                    _target_to_input_dim_h
                )

            elif self.bin_sizing == EmbeddingType.HESBO:
                debug("Creating random HeSBO embedding.")
                _input_to_target_dim_h: Dict[int, int] = {
                    i: self._random_state.choice(list(range(self.target_dim)))
                    for i in range(self.input_dim)
                }
            else:
                raise UnknownBehaviorError(
                    f"No such HeSBO bin-sizing behavior: {self.bin_sizing}"
                )

        self.target_to_input_dim: Dict[int, List[int]] = self._target_to_input_dim(
            _input_to_target_dim_h
        )

        self.input_dim_to_sign_sigma: Dict[int, int] = {
            i: int(self._random_state.choice([1, -1])) for i in range(self.input_dim)
        }

        self.S_prime: np.ndarray = self._compute_proj_mtrx(
            target_dim=self.target_dim,
            input_dim=self.input_dim,
            input_dim_to_sign_sigma=self.input_dim_to_sign_sigma,
            target_to_input_dim=self.target_to_input_dim,
        )
        self._S = None

    @staticmethod
    def _compute_proj_mtrx(
            target_dim: int,
            input_dim: int,
            input_dim_to_sign_sigma: Dict[int, int],
            target_to_input_dim: Dict[int, List[int]],
    ) -> np.ndarray:
        """
        Compute the projection matrix S', mapping from ambient to the target space.
        :param target_dim:
        :param input_dim:
        :param input_dim_to_sign_sigma:
        :param target_to_input_dim:
        :return:
        """
        rows = []
        for i in range(target_dim):
            rows.append(
                [
                    input_dim_to_sign_sigma[j] if j in target_to_input_dim[i] else 0
                    for j in range(input_dim)
                ]
            )
        return np.array(rows, dtype=np.float32).T

    @property
    def S(self) -> np.ndarray:
        return self.S_prime.T

    @property
    def input_to_target_dim(self) -> Dict[int, int]:
        d = {}
        for k, v in self.target_to_input_dim.items():
            for x in v:
                d[x] = k
        return d

    def project_down(self, X: np.ndarray) -> np.ndarray:
        """
        Project one or multiple points from the ambient into the target space.
        :param X: Points in the ambient space. Shape: [num_points, input_dim]
        :return: numpy array, shape: [num_points, target_dim]
        """
        X = np.array(X)
        assert len(X.shape) <= 2
        assert X.shape[0] == self.input_dim
        if not -1 <= X.min() <= X.max() <= 1:
            raise OutOfBoundsException()
        return self.S @ X

    def project_up(self, Y: np.ndarray) -> np.ndarray:
        """
        Project one or multiple points from the target into the ambient space.
        :param X: Points in the target space. Shape: [num_points, target_dim]
        :return: numpy array, shape: [num_points, input_dim]
        """
        Y = np.array(Y)
        assert len(Y.shape) <= 2
        assert Y.shape[0] == self.target_dim
        if not -1 <= Y.min() <= Y.max() <= 1:
            raise OutOfBoundsException()
        return self.S_prime @ Y

    def contributing_dimensions(self, target_dimension: int):
        """
        Returns the dimensions in the ambient space that contribute to a target dimension.
        :param target_dimension: the target dimension for which to return the contributing input dimensions
        :return: the input dimensions contributing to the target dimension
        """
        return self.target_to_input_dim[target_dimension]

    def increase_target_dimensionality(self, dims_and_bins: Dict[int, int]):
        """
        Split up one target dimension. The contributing input dimensions will be randomly assigned to two bins.
        One bin is the current target dimension, the other bin will be assigned to a new target dimension.
        Therefore, the target dimensionality will be increased by one. The projection matrix will change by this!
        The affected target dimension and the new dimension will only have half the number of contributing input
        dimensions than the target dimension prior to the splitting.
        :param splitting_target_dim: the target dimension to split
        :return: None
        """
        dims = list(dims_and_bins.keys())

        dims_and_contributing_input_dims = {
            i: deepcopy(self.contributing_dimensions(i)) for i in dims
        }
        # contributing_input_dims: np.ndarray = deepcopy(
        #    self.contributing_dimensions(splitting_target_dim)
        # )
        for d in dims:
            assert len(dims_and_contributing_input_dims[d]) >= dims_and_bins[d], (
                f"Only {len(dims_and_contributing_input_dims[d])} contributing input dimensions but want to split "
                f"into {dims_and_bins[d]} new bins"
            )
        for splitting_target_dim, n_new_bins in dims_and_bins.items():
            self.target_dim += n_new_bins - 1  # one bin is in the current dim
            contributing_input_dims = dims_and_contributing_input_dims[
                splitting_target_dim
            ]
            bins: List[np.ndarray] = []
            for b in range(n_new_bins):
                if b < n_new_bins - 1:
                    bin: np.ndarray = self._random_state.choice(
                        contributing_input_dims,
                        size=len(self.contributing_dimensions(splitting_target_dim))
                             // n_new_bins,
                        replace=False,
                    )
                    contributing_input_dims = np.setdiff1d(contributing_input_dims, bin)
                else:
                    bin: np.ndarray = contributing_input_dims
                bins.append(bin)

            self.target_to_input_dim[splitting_target_dim] = bins[0].tolist()
            for i, b in enumerate(bins[1:]):
                self.target_to_input_dim[self.target_dim - i - 1] = b.tolist()
        # re-compute S'
        S_prime_new = self._compute_proj_mtrx(
            target_dim=self.target_dim,
            input_dim=self.input_dim,
            input_dim_to_sign_sigma=self.input_dim_to_sign_sigma,
            target_to_input_dim=self.target_to_input_dim,
        )
        self.S_prime = S_prime_new

    def merge_dims(self, d1: int, d2: int):
        self.target_dim -= 1
        contrib_b1 = self.contributing_dimensions(d1)
        contrib_b2 = self.contributing_dimensions(d2)
        all_contrib = contrib_b1 + contrib_b2
        tds = self.target_to_input_dim.keys()
        dims_that_stay = [d for d in tds if d < min(d1, d2)]
        dims_minus_1 = [d for d in tds if min(d1, d2) < d < max(d1, d2)]
        dims_minus_2 = [d for d in tds if d > max(d1, d2)]
        new_target_to_input_dim = (
                {d: self.target_to_input_dim[d] for d in dims_that_stay}
                | {d - 1: self.target_to_input_dim[d] for d in dims_minus_1}
                | {d - 2: self.target_to_input_dim[d] for d in dims_minus_2}
        )
        max_td = max(new_target_to_input_dim.keys())

        new_target_to_input_dim[max_td + 1] = all_contrib
        self.target_to_input_dim = new_target_to_input_dim

        S_prime_new = self._compute_proj_mtrx(
            target_dim=self.target_dim,
            input_dim=self.input_dim,
            input_dim_to_sign_sigma=self.input_dim_to_sign_sigma,
            target_to_input_dim=self.target_to_input_dim,
        )
        self.S_prime = S_prime_new
