import math
import torch
import numpy as np
from ptflops import get_model_complexity_info


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


def calculate_MACs(model):
    """MACs stand for Multiply–accumulate operation introduced on MobileNetV
    to compute model complexity."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    macs, params = get_model_complexity_info(
        model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
