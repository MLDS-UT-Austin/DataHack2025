from typing import Tuple

import numpy as np
from numba import njit


@njit
def _fade(t):
    """Fade function for Perlin noise."""
    return t * t * t * (t * (t * 6 - 15) + 10)


@njit
def _lerp(t, a, b):
    """Linear interpolation for Perlin noise."""
    return a + t * (b - a)


@njit
def _grad(hash_val, x, y):
    """Gradient function for 2D Perlin noise."""
    h = hash_val & 15
    grad_x = 1 + (h & 7)  # Gradient x: 1, 2, ..., 8
    grad_x = -grad_x if (h & 8) != 0 else grad_x
    grad_y = 1 + ((h >> 4) & 7)  # Gradient y: 1, 2, ..., 8
    grad_y = -grad_y if ((h >> 4) & 8) != 0 else grad_y
    return grad_x * x + grad_y * y


@njit
def _perlin2d(x, y, p):
    """Compute 2D Perlin noise value at point (x,y)."""
    # Calculate integer coordinates
    xi, yi = int(x) & 255, int(y) & 255

    # Calculate fractional parts
    xf, yf = x - int(x), y - int(y)

    # Compute fade curves
    u, v = _fade(xf), _fade(yf)

    # Hash coordinates of the 4 corners
    aa = p[(p[xi] + yi) & 255]
    ab = p[(p[xi] + yi + 1) & 255]
    ba = p[(p[xi + 1] + yi) & 255]
    bb = p[(p[xi + 1] + yi + 1) & 255]

    # Interpolate between gradients
    x1 = _lerp(u, _grad(aa, xf, yf), _grad(ba, xf - 1, yf))
    x2 = _lerp(u, _grad(ab, xf, yf - 1), _grad(bb, xf - 1, yf - 1))
    return _lerp(v, x1, x2)


@njit
def _generate_permutation(seed):
    """Generate permutation table with given seed."""
    np.random.seed(seed)
    p = np.arange(256, dtype=np.int32)
    np.random.shuffle(p)
    return np.concatenate((p, p))


@njit
def generate_perlin_noise(seed: int, size: Tuple[int, int], octave: int, start_frequency: float) -> np.ndarray:
    """
    Generate a 2D Perlin noise array with Numba support.

    Parameters:
        seed (int): Seed for noise generation.
        size (Tuple[int, int]): Dimensions of the output noise array.
        octave (int): Number of octaves to add detail to the noise.

    Returns:
        np.ndarray: 2D array containing Perlin noise values.
    """
    noise_array = np.zeros(size, dtype=np.float32)
    p = _generate_permutation(seed)

    # Generate noise with multiple octaves
    scale = 100.0
    persistence = 0.5

    # This loops can't be jitted as a whole due to function call restrictions
    for i in range(size[0]):
        for j in range(size[1]):
            frequency = start_frequency
            total = 0.0
            amplitude = 1.0
            max_value = 0.0

            for _ in range(octave):
                total += (
                    _perlin2d(i * frequency / scale, j * frequency / scale, p)
                    * amplitude
                )
                max_value += amplitude
                amplitude *= persistence
                frequency *= 2.0

            # Normalize and store
            noise_array[i, j] = total / max_value

    return noise_array


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    noise = generate_perlin_noise(0, (1024, 512), 4)
    plt.imshow(noise, cmap="gray")
    plt.show()
