from enum import Enum


class EmbeddingType(Enum):
    BAXUS = 0
    """
    BAxUS embedding where each target bin has approx. the same number of contributing input dimensions.
    """
    HESBO = 1
    """
    HeSBO embedding where a target dimension is sampled for each input dimension.
    """
