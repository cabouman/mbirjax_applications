import numpy as np
import jax.numpy as jnp
from matplotlib.path import Path
import matplotlib.pyplot as plt

import mbirjax as mj


def gen_polygon_phantom(num_rows=512, num_slices=256):

    baseline_num_rows = 512
    linear_fraction = num_rows / baseline_num_rows
    # Create blank 2D image
    image = np.zeros((num_rows, num_rows), dtype=np.float32)

    # Define an asymmetric polygon
    pts = np.array([
        [100, 440],
        [80,  260],
        [280, 60],
        [420, 280],
        [360, 460],
    ])
    pts = (pts * linear_fraction).astype(int).reshape((-1, 1, 2))

    # Draw the polygon
    fill_convex_polygon(image, pts, 1.0)

    # Stack into a 3D volume of shape (512, 512, 256)
    phantom = np.repeat(image[:, :, np.newaxis], num_slices, axis=2)

    return phantom


def fill_convex_polygon(array: np.ndarray, vertices: np.ndarray, fill_value: float) -> None:
    """
    Fill a convex polygon in a 2D array.  This function is comparable to cv2.fillConvexPoly except that the vertices here
    are an ndarray and there are some minor differences at the boundary.

    Args:
        array : np.ndarray
            2D array to be modified in-place.
        vertices : np.ndarray
            An (m, 1, 2) array of integer vertex indices (row, col) into the array.
        fill_value : float
            Value to fill inside the polygon.

    Returns:
        None
            The array is modified in-place.
    """
    if vertices.shape[1:] != (1, 2):
        raise ValueError("vertices must have shape (m, 1, 2)")

    # Convert to (m, 2) shape
    verts = vertices[:, 0, ::-1]

    # Create a path object for the polygon
    path = Path(verts[:, ::-1])  # path expects (x, y) = (col, row)

    # Create a grid of array indices
    rows, cols = array.shape
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    coords = np.stack((cc.ravel(), rr.ravel()), axis=-1)  # shape (rows*cols, 2)

    # Find which points are inside the polygon
    mask = path.contains_points(coords).reshape(array.shape)

    # Fill the selected region
    array[mask] = fill_value


def show_image_with_angles(image: np.ndarray, *, angles_deg: np.ndarray = None, angles_rad: np.ndarray = None) -> None:
    """
    Display a square image and overlay lines representing angles.  Exactly one of angles_deg or angles_rad must be
    None, and the other must be an array of floats

    Args:
        image : np.ndarray
            A 2D square numpy array representing the image.
        angles_deg : np.ndarray
            A 1D array of angles in degrees. Each angle will be shown as a line
            through the image center in both directions.
        angles_rad : np.ndarray
            A 1D array of angles in radians. Each angle will be shown as a line
            through the image center in both directions.

    Returns:
        None
    """
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Image must be a square 2D array")
    if (angles_deg is None and angles_rad is None) or (angles_deg is not None and angles_rad is not None):
        raise ValueError("Exactly one of angles_deg or angles_rad must be None, and the other must be an array of floats")

    if angles_rad is None:
        angles_rad = np.deg2rad(angles_deg)

    side_length = image.shape[0]
    center = side_length / 2
    radius = side_length / 2  # Half-length of the line to reach from center to edge

    # Plot the image
    plt.imshow(image, cmap='gray', origin='upper', extent=[0, side_length, side_length, 0])
    plt.gca().set_aspect('equal')

    # Overlay lines for each angle
    colors = plt.cm.tab10(np.arange(len(angles_rad)) % 10)

    for i, theta in enumerate(angles_rad):
        dx = radius * np.cos(theta)
        dy = radius * np.sin(theta)

        x0, x1 = center - dx, center + dx
        y0, y1 = center - dy, center + dy

        plt.plot([x0, x1], [y0, y1], color=colors[i], linewidth=2)

    plt.title("Image with Overlaid Angles")
    plt.axis('off')
    plt.show()