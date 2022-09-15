from logging import debug
from typing import Tuple

import numpy as np
from torch.quasirandom import SobolEngine


def create_Xcand(
        x_center: np.ndarray,
        weights: np.ndarray,
        length: float,
        dim: int,
        n_cand: int,
        dtype,
        device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    :param x_center: the TR center
    :param weights: the weights of the dims
    :param length: baselength
    :param dim: the target dim
    :param n_cand: number of candidates
    :param dtype: the data type
    :param device: the device
    :return: triple, X_cand, lb, ub
    """
    debug(f"creating {n_cand} candidates")
    lb = np.clip(x_center - weights * length, -1.0, 1.0)
    ub = np.clip(x_center + weights * length, -1.0, 1.0)

    # Draw a Sobolev sequence in [lb, ub]
    seed = np.random.randint(int(1e6))
    sobol = SobolEngine(dim, scramble=True, seed=seed)
    pert = sobol.draw(n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
    pert = lb + (ub - lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = np.random.rand(n_cand, dim) <= prob_perturb
    ind = np.where(np.sum(mask, axis=1) == 0)[0]
    mask[ind, np.random.randint(0, dim - 1, size=len(ind))] = 1

    # Create candidate points
    X_cand = x_center.copy() * np.ones((n_cand, dim))
    X_cand[mask] = pert[mask]
    return X_cand, lb, ub
