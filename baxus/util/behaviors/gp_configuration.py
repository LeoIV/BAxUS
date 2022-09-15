from dataclasses import dataclass
from enum import Enum


class MLLEstimation(Enum):
    MULTI_START_GRADIENT_DESCENT = 1
    """
    Sample a number of points and start gradient-based optimization on every point.
    """
    LHS_PICK_BEST_START_GD = 2
    """
    Sample a number of points and start gradient-based optimization on the best initial points.
    """


@dataclass
class GPBehaviour:
    mll_estimation: MLLEstimation = MLLEstimation.LHS_PICK_BEST_START_GD
    """
    The maximum-likelihood-estimation method.
    """
    n_initial_samples: int = 50
    """
    The initial samples.
    """
    n_best_on_lhs_selection: int = 5
    """
    The number of best samples on which to start the gradient-based optimizer.
    """
    n_mle_training_steps: int = 50
    """
    The number of gradient updates.
    """

    def __str__(self):
        return (
            f"mle_{self.mll_estimation.name}_n_init_mle_{self.n_initial_samples}_"
            f"n_best_lhs_{self.n_best_on_lhs_selection}_mle_steps_{self.n_mle_training_steps}"
        )
