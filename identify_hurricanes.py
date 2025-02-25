import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit
from scipy import ndimage


# @njit
def sobel_filter(image):
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    rows, cols = image.shape
    Gx = np.zeros_like(image, dtype=np.float32)
    Gy = np.zeros_like(image, dtype=np.float32)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i - 1 : i + 2, j - 1 : j + 2]
            Gx[i, j] = np.sum(Kx * region)
            Gy[i, j] = np.sum(Ky * region)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    return magnitude


# @jit
def identify_hurricanes(u_velocity, v_velocity, grid_resolution=1.0, threshold=0.5):
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
    dy_u = ndimage.sobel(u_velocity, axis=0) / (2 * grid_resolution)
    dx_v = ndimage.sobel(v_velocity, axis=1) / (2 * grid_resolution)
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
            if y == 25 and x == 49:
                pass  # bottom middle-left
            v = np.array([u_velocity[y, x], v_velocity[y, x]])
            v_perp = np.array([-v[1], v[0]])
            curl_value = curl[y, x]
            cp = np.array([x, y]) + v_perp * curl_value
            cp = cp.astype(int)
            angle = flow_angles[y, x]
            try:
                # Assign weights based on angle difference
                if -np.pi / 4 <= angle <= np.pi / 4:
                    downward_grid[cp[1], cp[0]] += abs(curl_value) * 0.2
                elif np.pi / 4 <= angle <= 3 * np.pi / 4:
                    leftward_grid[cp[1], cp[0]] += abs(curl_value) * 0.2
                elif -3 * np.pi / 4 <= angle <= -np.pi / 4:
                    rightward_grid[cp[1], cp[0]] += abs(curl_value) * 0.2
                else:
                    # temp[y, x] += 1
                    upward_grid[cp[1], cp[0]] += abs(curl_value) * 0.2
            except:
                pass

    # Aggregate the direction grids to create a hurricane indicator field
    # In a hurricane, we expect strong rotation with a mix of all directions around a center
    def plot_grid(grid):
        return
        plt.imshow(grid, cmap="viridis")
        plt.colorbar()
        plt.show()

    plot_grid(u_velocity)
    plot_grid(v_velocity)

    upward_grid = ndimage.gaussian_filter(upward_grid, sigma=2.0)
    plot_grid(upward_grid)
    rightward_grid = ndimage.gaussian_filter(rightward_grid, sigma=2.0)
    plot_grid(rightward_grid)
    leftward_grid = ndimage.gaussian_filter(leftward_grid, sigma=2.0)
    plot_grid(leftward_grid)
    downward_grid = ndimage.gaussian_filter(downward_grid, sigma=2.0)
    plot_grid(downward_grid)

    hurricane_indicator = (
        upward_grid * rightward_grid * leftward_grid * downward_grid
    ) ** 0.25

    # Apply a smoothing filter to the hurricane indicator field
    # hurricane_indicator = ndimage.gaussian_filter(hurricane_indicator, sigma=2.0)

    # Find local maxima in the hurricane indicator field
    # These are candidate hurricane centers
    plot_grid(hurricane_indicator)

    max_filter = ndimage.maximum_filter(hurricane_indicator, size=5)
    hurricane_mask = (hurricane_indicator == max_filter) & (
        hurricane_indicator > threshold
    )

    # Get the coordinates and strengths of identified hurricanes
    hurricane_centers = list(zip(*np.where(hurricane_mask)))
    hurricane_strength = [hurricane_indicator[center] for center in hurricane_centers]

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
    ny, nx = 50, 100
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Example swirl: a vortex around (0,0)
    # (u, v) = (-y, x) is a classic counter-clockwise swirl
    u = -Y
    v = X

    centers, sizes, _ = identify_hurricanes(u, v, 0.02, 0.25)

    visualize_hurricanes(u, v, centers)
