class ArgumentError(Exception):
    """
    An exception for an illegal input argmument.
    """
    pass


class EffectiveDimTooLargeException(Exception):
    """
    When the effective dimensionality is too large (for example when larger than the input dimensionality).
    """
    pass


class OutOfBoundsException(Exception):
    """
    When a point falls outside the search space.
    """
    pass


class BoundsMismatchException(Exception):
    """
    When the search space bounds don't have the same length.
    """
    pass


class UnknownBehaviorError(Exception):
    pass
