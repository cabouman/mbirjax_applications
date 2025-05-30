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

def Create2DMask(slice):

    y_indices, x_indices = np.where(slice > 0)

    # Calculate x_min, x_max, y_min, y_max
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Calculate the center of the circle
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Calculate the radius of the circle as the maximum distance from the center
    radius = np.max(np.sqrt((x_indices - x_center) ** 2 + (y_indices - y_center) ** 2))
    radius_bigger = 1.01 * radius

    # Generate the mask: if the distance from the center is less than the radius, set value to 1
    h, w = slice.shape
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]

    # Compute squared distances
    dist2 = (x - x_center) ** 2 + (y - y_center) ** 2

    # Build boolean mask in one shot
    mask = (dist2 <= radius_bigger ** 2).astype(np.float32)

    return mask

def Create3DMask(phantom,repeat=False):

    if repeat:
        mask2d = Create2DMask(phantom[:,:,0])
        mask = np.repeat(mask2d[:, :, np.newaxis], phantom.shape[2], axis=2)

    else:
        mask = np.zeros(phantom.shape)
        for i in range(phantom.shape[2]):
            mask[:,:,i] = Create2DMask(phantom[i])

    return mask

def Subsampling3DIndices(mask, r_1):

    num_rows, num_cols, num_slices = mask.shape
    num_samples = int(num_rows*num_cols*r_1)

    random_indices = []
    for slice_idx in range(num_slices):
        # Randomly select unique indices for this slice
        mask_indices = np.where(mask[:,:,slice_idx] == 1)  # Get 2D indices where mask == 1
        # Ensure num_samples does not exceed the number of available points
        if num_samples > len(mask_indices[0]):
            num_samples_temp = len(mask_indices[0])
        else:
            num_samples_temp = num_samples
        slice_choice = np.random.choice(len(mask_indices[0]), num_samples_temp, replace=False)
        row_indices = mask_indices[0][slice_choice]
        col_indices = mask_indices[1][slice_choice]
        random_indices.append((row_indices, col_indices, slice_idx * np.ones(num_samples_temp, dtype=int)))

    # Convert to a single index array for advanced indexing
    random_indices = tuple(np.concatenate(idx) for idx in zip(*random_indices))

    return random_indices

def Subsampling2DIndices(mask, num_slices, r_1):

    num_rows, num_cols = mask.shape
    num_samples = int(num_rows*num_cols*r_1)
    mask_indices = np.where(mask[:, :] == 1)  # Get 2D indices where mask == 1
    # Ensure num_samples does not exceed the number of available points
    if num_samples > len(mask_indices[0]):
        num_samples = len(mask_indices[0])
    slice_choice = np.random.choice(len(mask_indices[0]), num_samples, replace=False)
    row_inds = mask_indices[0][slice_choice]
    col_inds = mask_indices[1][slice_choice]

    row_inds_3d   = np.tile(row_inds, num_slices)
    col_inds_3d   = np.tile(col_inds, num_slices)
    slice_inds_3d = np.repeat(np.arange(num_slices, dtype=int), num_samples)

    # pack into the same format as your other function
    random_indices_3d = (row_inds_3d, col_inds_3d, slice_inds_3d)

    random_indices_2d = row_inds * num_cols + col_inds
    #random_indices_2d = np.sort(random_indices_2d)
    random_indices_2d = jnp.array(random_indices_2d)
    #random_indices_2d = jnp.array(random_indices_raster, dtype=jnp.int32)[None, :]  # shape (1, N)

    return random_indices_3d, random_indices_2d, (row_inds,col_inds)


def show_image_with_angles(image: np.ndarray, angles_deg: np.ndarray) -> None:
    """
    Display a square image and overlay lines representing angles.

    Parameters
    ----------
    image : np.ndarray
        A 2D square numpy array representing the image.
    angles_deg : np.ndarray
        A 1D array of angles in degrees. Each angle will be shown as a line
        through the image center in both directions.

    Returns
    -------
    None
    """
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Image must be a square 2D array")

    side_length = image.shape[0]
    center = side_length / 2
    radius = side_length / 2  # Half-length of the line to reach from center to edge

    # Plot the image
    plt.imshow(image, cmap='gray', origin='upper', extent=[0, side_length, side_length, 0])
    plt.gca().set_aspect('equal')

    # Overlay lines for each angle
    colors = plt.cm.tab10(np.arange(len(angles_deg)) % 10)

    for i, angle_deg in enumerate(angles_deg):
        theta = np.deg2rad(angle_deg)
        dx = radius * np.cos(theta)
        dy = radius * np.sin(theta)

        x0, x1 = center - dx, center + dx
        y0, y1 = center - dy, center + dy

        plt.plot([x0, x1], [y0, y1], color=colors[i], linewidth=2)

    plt.title("Image with Overlaid Angles")
    plt.axis('off')
    plt.show()