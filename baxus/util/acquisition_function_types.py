from enum import Enum


class AcquisitionFunctionType(Enum):
    EXPECTED_IMPROVEMENT = 1
    """
    Expected improvement acquisition function.
    """
    THOMPSON_SAMPLING = 2
    """
    Thompson sampling acquisition function.
    """
