from dataclasses import dataclass
from typing import Any, Dict

from baxus.util.acquisition_function_types import AcquisitionFunctionType
from baxus.util.behaviors.embedding_configuration import EmbeddingType


@dataclass
class EmbeddedTuRBOBehavior:
    """
    The behavior of the embedded TuRBO algorithm

    """

    initial_base_length: float = 0.8
    """
    The initial base side length (see TuRBO paper)
    
    """
    max_base_length: float = 1.6
    """
    The maximum base side length (see TuRBO paper)
    
    """
    min_base_length: float = 0.5 ** 7
    """
    The minimum base side length (see TuRBO paper). If you get lower than this, the trust region dies out.
    
    """
    success_tolerance: int = 3
    """
    The number of times we consecutively have to find a better point in order to expand the trust region, initial value
    
    """
    acquisition_function: AcquisitionFunctionType = AcquisitionFunctionType.THOMPSON_SAMPLING
    """
    The different acquisition functions to use in a multi-batch setting (default: only Thompson sampling)
    
    """
    noise: float = 0.
    """
    The noise of the problem.
    """

    embedding_type: EmbeddingType = EmbeddingType.BAXUS
    """
    Uniform bin sizing means that all target bins have approx. equally many contributing input dimensions.
    Random bin sizing means that a random target dimension is chosen for each input dimension (standard HeSBO
    behavior).
    """

    success_decision_factor: float = 0.001
    """
    The difference wrt to the current incumbent solution required for a next point to be considered a success.

    """

    def __str__(self):
        return (
            f"_linit_{self.initial_base_length}"
            f"_lmax_{self.max_base_length}"
            f"_lmin_{self.min_base_length}"
            f"_successtol_{self.success_tolerance}"
            f"_acq_{self.acquisition_function.name}"
            f"_noise_{self.noise}"
            f"_et_{self.embedding_type.name}"
            f"_sdf_{self.success_decision_factor}"
        )

    @property
    def conf_dict(self) -> Dict[str, Any]:
        """
        The configuration as a dictionary.

        Returns: The configuration as a dictionary.

        """
        return {
            "initial base length": self.initial_base_length,
            "maximum base length": self.max_base_length,
            "minimum base length": self.min_base_length,
            "success tolerance": self.success_tolerance,
            "acquisition_functions": self.acquisition_function.name,
            "observation noise": self.noise,
            "embedding type": self.embedding_type.name,
            "success decision factor": self.success_decision_factor,
        }

    def pretty_print(self) -> str:
        """
        A nice string of the configuration.

        Returns: A nice string of the configuration.

        """
        pstring = ""
        for k, v in self.conf_dict.items():
            pstring += f"\t-{k}: {v}\n"
        return pstring
