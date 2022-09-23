from dataclasses import dataclass
from typing import Any, Dict

from baxus.util.behaviors.embedded_turbo_configuration import EmbeddedTuRBOBehavior


@dataclass
class BaxusBehavior(EmbeddedTuRBOBehavior):
    """
    The behavior of the BAxUS algorithm.

    """

    n_new_bins: int = 3
    """
    Number of new bins after a splitting. Default: 3
    
    """

    budget_until_input_dim: int = 0
    """
    The budget after which we have reached the input dimension under the assumption that we always fail.
    If zero: use the entire evaluation budget.
    """

    adjust_initial_target_dim: bool = True
    """
    Whether to adjust the initial target dim such that the final split is as close to the ambient dim as possible.
    """

    def __str__(self):
        return (
            f"{super().__str__()}"
            f"_nbos_{self.n_new_bins}"
            f"_aitd_{self.adjust_initial_target_dim}"
            f"_buad_{self.budget_until_input_dim}"
        )

    @property
    def conf_dict(self) -> Dict[str, Any]:
        """
        The configuration as a dictionary.

        Returns: The configuration as a dictionary.

        """
        base_class_dict = super().conf_dict
        this_dict = {
            "number of new bins per dimension": self.n_new_bins,
            "adjust initial target dimension": self.adjust_initial_target_dim,
            "budget until input dimension": self.budget_until_input_dim,
        }
        return {**base_class_dict, **this_dict}
