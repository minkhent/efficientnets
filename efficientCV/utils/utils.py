import math
import numpy as np


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


def _make_divisible(value: float, divisor: int = 8) -> float:
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters: int, width_mult: float) -> int:
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats: int, depth_mult: float) -> int:
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))
