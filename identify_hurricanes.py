import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit
from scipy import ndimage

from util import *


@njit
def identify_hurricanes(u_velocity, v_velocity, grid_resolution=1.0, threshold=0.8, size_mult=140, size_max=25):
    """
    Identify hurricane-like structures in a velocity field using curl analysis.

    Parameters:
    -----------
    u_velocity : 2D numpy array
        Eastward (x-direction) component of the velocity field
    v_velocity : 2D numpy array
        Northward (y-direction) component of the velocity field
    grid_resolution : float
        Grid spacing in consistent units
    threshold : float
        Threshold for curl magnitude to be considered significant

    Returns:
    --------
    hurricane_centers : list of tuples
        List of (row, col) positions of identified hurricane centers
    hurricane_strength : list of float
        Corresponding vorticity strength at each hurricane center
    """
    # Calculate curl (vorticity) of the velocity field
    # For 2D flow, curl is a scalar representing rotation in the z-direction
    dy_u = sobel(u_velocity, axis=0) / (2 * grid_resolution)
    dx_v = sobel(v_velocity, axis=1) / (2 * grid_resolution)
    curl = dx_v - dy_u

    # Calculate flow direction angles
    flow_angles = np.arctan2(v_velocity, u_velocity)

    # Initialize direction grids for each of the four directions
    shape = curl.shape
    upward_grid = np.zeros(shape)
    rightward_grid = np.zeros(shape)
    leftward_grid = np.zeros(shape)
    downward_grid = np.zeros(shape)

    # temp = np.zeros(shape)

    # Process the curl field to identify curved features
    # Use wind direction to determine flow patterns around grid points
    for y in range(1, shape[0] - 1):
        for x in range(1, shape[1] - 1):
            if y == 98 and x == 49:
                pass  # bottom middle
                # curl_value = 8.24
                # .12
            if y == 75 and x == 49:
                pass  # bottom/middle middle
                # curl_value = 15.68
                # .06

            v = np.array([u_velocity[y, x], v_velocity[y, x]])
            v_unit = v / np.linalg.norm(v)
            v_len = np.linalg.norm(v)
            v_perp = np.array([-v_unit[1], v_unit[0]])
            curl_value = curl[y, x]
            if abs(curl_value) == 0:
                continue
            cp = np.array([x, y]) + v_perp * v_len / curl_value * 8.0
            cp_x, cp_y = int(cp[0]), int(cp[1])
            angle = flow_angles[y, x]
            if 0 <= cp_x < shape[1] and 0 <= cp_y < shape[0]:
                # Assign weights based on angle difference
                if -np.pi / 4 <= angle <= np.pi / 4:
                    downward_grid[cp_y, cp_x] += abs(curl_value) * 0.2
                elif np.pi / 4 <= angle <= 3 * np.pi / 4:
                    leftward_grid[cp_y, cp_x] += abs(curl_value) * 0.2
                elif -3 * np.pi / 4 <= angle <= -np.pi / 4:
                    rightward_grid[cp_y, cp_x] += abs(curl_value) * 0.2
                else:
                    upward_grid[cp_y, cp_x] += abs(curl_value) * 0.2

    # Aggregate the direction grids to create a hurricane indicator field
    # In a hurricane, we expect strong rotation with a mix of all directions around a center
    def plot_grid(grid):
        return
        plt.imshow(grid, cmap="viridis")
        plt.colorbar()
        plt.show()

    # plot_grid(u_velocity)
    # plot_grid(v_velocity)

    sigma = 1.5

    upward_grid = gaussian_filter(upward_grid, sigma=sigma)
    plot_grid(upward_grid)
    rightward_grid = gaussian_filter(rightward_grid, sigma=sigma)
    plot_grid(rightward_grid)
    leftward_grid = gaussian_filter(leftward_grid, sigma=sigma)
    plot_grid(leftward_grid)
    downward_grid = gaussian_filter(downward_grid, sigma=sigma)
    plot_grid(downward_grid)

    hurricane_indicator = (
        upward_grid * rightward_grid * leftward_grid * downward_grid
    ) ** 0.25

    # Apply a smoothing filter to the hurricane indicator field
    # hurricane_indicator = gaussian_filter(hurricane_indicator, sigma=2.0)

    # Find local maxima in the hurricane indicator field
    # These are candidate hurricane centers
    plot_grid(hurricane_indicator)

    max_filter = maximum_filter(hurricane_indicator, size=5)
    hurricane_mask = (hurricane_indicator == max_filter) & (
        hurricane_indicator > threshold
    )

    # Get the coordinates and strengths of identified hurricanes
    hurricane_centers = list(zip(*np.where(hurricane_mask)))
    hurricane_strength = [min(hurricane_indicator[center]*size_mult, size_max) for center in hurricane_centers]

    # remove overlapping hurricane (keep the strongest)
    # Remove overlapping hurricane centers (keep the strongest)
    if hurricane_centers:
        # Sort centers by strength in descending order
        sorted_indices = np.argsort(np.array([-s for s in hurricane_strength]))
        sorted_centers = [hurricane_centers[i] for i in sorted_indices]
        sorted_strengths = [hurricane_strength[i] for i in sorted_indices]
        
        # Keep track of which centers to keep
        keep_centers = []
        keep_strengths = []
        
        # Define minimum distance between hurricane centers (avoid overlaps)
        min_distance = 2
        
        for i, (center, strength) in enumerate(zip(sorted_centers, sorted_strengths)):
            # Check if this center is too close to any already-kept center
            for kept_center, kept_strength in zip(keep_centers, keep_strengths):
                distance = np.sqrt((center[0] - kept_center[0])**2 + (center[1] - kept_center[1])**2)
                if distance < kept_strength + min_distance:
                    break
            else:
                # If not too close to any kept center, keep it
                keep_centers.append(center)
                keep_strengths.append(strength)
        
        hurricane_centers = keep_centers
        hurricane_strength = keep_strengths

    return hurricane_centers, hurricane_strength, hurricane_indicator


def visualize_hurricanes(u_velocity, v_velocity, hurricane_centers=None):
    """
    Visualize the velocity field and identified hurricane centers.

    Parameters:
    -----------
    u_velocity : 2D numpy array
        Eastward component of the velocity field
    v_velocity : 2D numpy array
        Northward component of the velocity field
    hurricane_centers : list of tuples
        List of (row, col) positions of identified hurricane centers
    """
    plt.figure(figsize=(10, 8))

    # Create a grid for the quiver plot
    y, x = np.mgrid[: u_velocity.shape[0], : u_velocity.shape[1]]

    # Plot the velocity field
    plt.quiver(x, y, u_velocity * -1, v_velocity * -1, scale=50)

    # Plot hurricane centers if provided
    if hurricane_centers:
        hurricane_y, hurricane_x = zip(*hurricane_centers)
        plt.scatter(hurricane_x, hurricane_y, color="r", s=100, marker="o")

    plt.title("Velocity Field with Hurricane Centers")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    while True:
        pass


# ------------------ Example Usage ------------------
if __name__ == "__main__":
    # Fake velocity field, e.g. swirling around center
    ny, nx = 100, 100
    # x = np.linspace(-1, 1, nx)
    # y = np.linspace(-1, 1, ny)
    x = np.linspace(1, -1, nx)
    y = np.linspace(1, -1, ny)
    X, Y = np.meshgrid(x, y)

    # Example swirl: a vortex around (0,0)
    # (u, v) = (-y, x) is a classic counter-clockwise swirl
    u = -Y
    v = X
    scale = 100.0

    centers, sizes, _ = identify_hurricanes(u * scale, v * scale, 1.0, 0.25)

    visualize_hurricanes(u, v, centers)
