"""Common definitions shared by NSOLT layers."""

from enum import IntEnum

__all__ = ["Direction", "ChannelGroup"]


class Direction(IntEnum):
    """Index of each direction in ``decimation_factor`` and similar parameters.

    The order is (vertical, horizontal, depth) as in MATLAB
    ``saivdr.dictionary.utility.Direction``, but 0-based.
    """

    VERTICAL = 0
    HORIZONTAL = 1
    DEPTH = 2


class ChannelGroup(IntEnum):
    """Index of each channel group in ``number_of_channels`` (0-based)."""

    UPPER = 0
    LOWER = 1
