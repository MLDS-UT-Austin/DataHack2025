import math
import random
from typing import Tuple

import numpy as np
from numba import njit
import random

# fmt: off
PERM = [151,160,137,91,90,15,
        131,13,201,95,96,53,194,233,7,225,140,36,103,30,
        69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,
        75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,
        33,88,237,149,56,87,174,20,125,136,171,168,68,175,
        74,165,71,134,139,48,27,166,77,146,158,231,83,111,
        229,122,60,211,133,230,220,105,92,41,55,46,245,40,
        244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,
        187,208,89,18,169,200,196,135,130,116,188,159,86,
        164,100,109,198,173,186,3,64,52,217,226,250,124,123,
        5,202,38,147,118,126,255,82,85,212,207,206,59,227,
        47,16,58,17,182,189,28,42,223,183,170,213,119,248,
        152,2,44,154,163,70,221,153,101,155,167,43,172,9,
        129,22,39,253,19,98,108,110,79,113,224,232,178,185,
        112,104,218,246,97,228,251,34,242,193,238,210,144,
        12,191,179,162,241,81,51,145,235,249,14,239,107,49,
        192,214,31,181,199,106,157,184,84,204,176,115,121,50,
        45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,
        243,141,128,195,78,66,215,61,156,180] * 2
PERM = np.array(PERM, dtype=np.int32)
# fmt: on


@njit
def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


@njit
def _lerp(t, a, b):
    return a + t * (b - a)


@njit
def _grad(hash, x, y, z):
    h = hash & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if (h == 12 or h == 14) else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


@njit
def _noise(x, y, z):
    X = int(math.floor(x)) & 255
    Y = int(math.floor(y)) & 255
    Z = int(math.floor(z)) & 255
    x -= math.floor(x)
    y -= math.floor(y)
    z -= math.floor(z)
    u = _fade(x)
    v = _fade(y)
    w = _fade(z)
    A = PERM[X] + Y
    AA = PERM[A] + Z
    AB = PERM[A + 1] + Z
    B = PERM[X + 1] + Y
    BA = PERM[B] + Z
    BB = PERM[B + 1] + Z
    return _lerp(
        w,
        _lerp(
            v,
            _lerp(u, _grad(PERM[AA], x, y, z), _grad(PERM[BA], x - 1, y, z)),
            _lerp(u, _grad(PERM[AB], x, y - 1, z), _grad(PERM[BB], x - 1, y - 1, z)),
        ),
        _lerp(
            v,
            _lerp(
                u,
                _grad(PERM[AA + 1], x, y, z - 1),
                _grad(PERM[BA + 1], x - 1, y, z - 1),
            ),
            _lerp(
                u,
                _grad(PERM[AB + 1], x, y - 1, z - 1),
                _grad(PERM[BB + 1], x - 1, y - 1, z - 1),
            ),
        ),
    )



@njit
def generate_perlin_noise(amplitude:float, scale:float, size: Tuple[int, int]) -> np.ndarray:
    # def generate_perlin_noise(seed: int, size: Tuple[int, int], octave: int, start_frequency: float) -> np.ndarray:
    x_offset = random.random() * 1000000
    y_offset = random.random() * 1000000
    output = np.zeros(size, dtype=np.float32)
    for j in range(size[0]):
        for i in range(size[1]):
            output[j, i] = (
                _noise(i * scale + x_offset, j * scale + y_offset, 0) * amplitude
            )
    return output.astype(np.float64)


@njit
def rescale(x, new_min, new_max):
    """
    Rescale an array to a new range.

    Parameters:
        x (np.ndarray): Input array to rescale.
        new_min (float): New minimum value.
        new_max (float): New maximum value.

    Returns:
        np.ndarray: Rescaled array.
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x)) * (new_max - new_min) + new_min


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    noise = generate_perlin_noise(0.01, 0.02, (100, 100))
    plt.imshow(noise, cmap="gray")
    plt.show()
