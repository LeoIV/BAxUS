from abc import ABC
from logging import warning, info, debug
from typing import Optional, Dict, List

import numpy as np
from numpy.random import RandomState

from baxus.util.behaviors.embedding_configuration import EmbeddingType
from baxus.util.data_utils import right_pad_sequence
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
        self._reset()

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
            self.S = np.eye(self.target_dim)
        else:
            if self.bin_sizing == EmbeddingType.BAXUS:
                debug("Creating BAxUS embedding.")
                input_dim_permutation = np.random.permutation(list(range(self.input_dim)))

                input_dim_bins = np.array_split(input_dim_permutation + 1, self.target_dim)
                input_dim_bins = right_pad_sequence(input_dim_bins, dtype=np.int)

                mtrx = np.zeros((self.target_dim, self.input_dim + 1))
                np.put_along_axis(arr=mtrx, indices=input_dim_bins,
                                  values=np.random.choice(np.array([-1, +1]), size=input_dim_bins.shape), axis=1)
                self.S = mtrx[:, 1:]

            elif self.bin_sizing == EmbeddingType.HESBO:
                debug("Creating HeSBO embedding.")
                target_dims = np.random.choice(np.arange(self.target_dim), size=self.input_dim)
                mtrx = np.zeros((self.target_dim, self.input_dim))
                np.put_along_axis(arr=mtrx, indices=target_dims.reshape((1, self.input_dim)),
                                  values=np.random.choice(np.array([-1, +1]), size=self.input_dim), axis=0)
                self.S = mtrx
            else:
                raise UnknownBehaviorError(
                    f"No such HeSBO bin-sizing behavior: {self.bin_sizing}"
                )

    @property
    def S_prime(self) -> np.ndarray:
        return self.S.T

    @property
    def input_to_target_dim(self) -> Dict[int, int]:
        """
        Return the target dimension each input dimension is mapped to.

        Returns: the target dimension each input dimension is mapped to.

        """
        return {
            D: int(np.nonzero(self.S[:, D])[0]) for D in range(self.input_dim)
        }

    @property
    def target_to_input_dim(self) -> Dict[int, List[int]]:
        """
        Return a list of input dimensions the target dimension maps to.

        Returns: A list of input dimensions the target dimension maps to.

        """
        return {
            d: np.nonzero(self.S[d])[0].tolist() for d in range(self.target_dim)
        }

    def project_down(self, X: np.ndarray) -> np.ndarray:
        """
        Project one or multiple points from the ambient into the target space.

        Args:
            X: Points in the ambient space. Shape: [num_points, input_dim]

        Returns: numpy array, shape: [num_points, target_dim]

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

        Args:
            Y: Points in the target space. Shape: [num_points, target_dim]

        Returns: numpy array, shape: [num_points, input_dim]

        """
        Y = np.array(Y)
        assert len(Y.shape) <= 2
        assert Y.shape[0] == self.target_dim
        if not -1 <= Y.min() <= Y.max() <= 1:
            raise OutOfBoundsException()
        return self.S_prime @ Y

    def contributing_dimensions(self, target_dimension: int) -> np.ndarray:
        """
        Returns the dimensions in the ambient space that contribute to a target dimension.

        Args:
            target_dimension: the target dimension for which to return the contributing input dimensions

        Returns: the input dimensions contributing to the target dimension

        """

        return np.nonzero(self.S[target_dimension])[0]

    def increase_target_dimensionality(self, dims_and_bins: Dict[int, int]):
        """
        Split up one target dimension. The contributing input dimensions will be randomly assigned to two bins.
        One bin is the current target dimension, the other bin will be assigned to a new target dimension.
        Therefore, the target dimensionality will be increased by one. The projection matrix will change by this!
        The affected target dimension and the new dimension will only have half the number of contributing input
        dimensions than the target dimension prior to the splitting.

        Args:
            dims_and_bins: the dimensions and the number of bins to split them into

        Returns: Nothing, S_prime gets updated

        """

        for splitting_target_dim, n_new_bins in dims_and_bins.items():
            contributing_input_dims = np.random.permutation(self.contributing_dimensions(splitting_target_dim))
            non_zero_elements = self.S[splitting_target_dim, contributing_input_dims].squeeze()

            assert len(contributing_input_dims) >= dims_and_bins[splitting_target_dim], (
                f"Only {len(contributing_input_dims)} contributing input dimensions but want to split "
                f"into {dims_and_bins[splitting_target_dim]} new bins"
            )
            self.target_dim += n_new_bins - 1  # one bin is in the current dim
            new_bins = np.array_split(contributing_input_dims + 1, n_new_bins)[1:]
            elements_to_move = np.array_split(non_zero_elements, n_new_bins)[1:]

            new_bins_padded = right_pad_sequence(new_bins, dtype=np.int)
            elements_to_move_padded = right_pad_sequence(elements_to_move)

            S_stack = np.zeros((n_new_bins - 1, self.S.shape[1] + 1))
            np.put_along_axis(arr=S_stack, indices=new_bins_padded, values=elements_to_move_padded, axis=1)
            self.S[splitting_target_dim, np.hstack(new_bins) - 1] = 0

            self.S = np.vstack((self.S, S_stack[:, 1:]))
