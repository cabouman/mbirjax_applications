import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


def create_artificial_roi(reference_object):
    """Create an ROI that includes the two upper edges of the object."""
    num_rows, num_cols, num_slices = reference_object.shape

    roi_2d = np.zeros((num_rows, num_cols), dtype=bool)
    roi_2d[int(0.07 * num_rows):int(0.57 * num_rows),
           int(0.25 * num_cols):int(0.76 * num_cols)] = True

    roi = np.repeat(roi_2d[:, :, np.newaxis], num_slices, axis=2)

    return roi


def show_roi(reference_object, roi):
    """Display the ROI on one reference-object slice."""
    slice_index = reference_object.shape[2] // 2

    fig, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(reference_object[:, :, slice_index], cmap='gray')
    axis.contour(roi[:, :, slice_index], levels=[0.5], colors='tab:red')
    axis.set_title('Reference Object with ROI')
    axis.axis('off')
    fig.tight_layout()
    plt.show()


def show_view_angle_comparison(image, angle_sets, titles):
    """Display several sets of projection angles on the same reference image."""
    rows, cols = image.shape
    center_x, center_y = cols / 2, rows / 2
    radius = min(rows, cols) / 2

    fig, axes = plt.subplots(1, len(angle_sets), figsize=(5 * len(angle_sets), 5))
    for axis, angles, title in zip(axes, angle_sets, titles):
        rotation_angles = np.pi / 2 + np.asarray(angles)
        colors = plt.cm.tab10(np.arange(len(rotation_angles)) % 10)

        axis.imshow(image, cmap='gray', origin='upper', extent=[0, cols, rows, 0])
        axis.set_aspect('equal')
        for angle, color in zip(rotation_angles, colors):
            dx = 0.95 * radius * np.cos(angle)
            dy = -0.95 * radius * np.sin(angle)
            axis.arrow(
                center_x, center_y, dx, dy, color=color, linewidth=1.75,
                head_width=min(rows, cols) * 0.02, length_includes_head=True
            )

        axis.set_title(title)
        axis.axis('off')

    fig.suptitle('Selected View Angle Comparison')
    fig.tight_layout()
    plt.show()


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


def create_uniform_index(angle_candidates, end_index, num_views):
    """
    Select `num_views` indices uniformly from the top `end_index` angles,
    without matching values back into the original array via np.where.

    This function:
      1. Computes the descending sort order of `angle_candidates`.
      2. Takes the first `end_index` indices from that sort.
      3. Computes `num_views` positions spread evenly (using floor) across
         those `end_index` slots, always including the first and last.
      4. Returns the corresponding indices into the original `angle_candidates`
         array.

    Args:
        angle_candidates : array-like of shape (N,)
            1D array (or list) of candidate angles.
        end_index : int
            Number of top angles (in descending order) to consider. Must be
            between 1 and len(angle_candidates), inclusive.
        num_views : int
            Number of indices to return. Must be at least 2.

    Returns:
        List[int]
            A list of length `num_views` containing indices into `angle_candidates`.
            These correspond to uniformly spaced angles within the top `end_index`
            largest values of `angle_candidates`.

    Raises:
        ValueError:
            If end_index < 1 or end_index > len(angle_candidates),
            or if num_views < 2.
    """
    angles = np.asarray(angle_candidates)
    N = angles.shape[0]

    if not (1 <= end_index <= N):
        raise ValueError(f"end_index must be between 1 and {N}, got {end_index}.")
    if num_views < 2:
        raise ValueError(f"num_views must be at least 2, got {num_views}.")

    # 1. Get indices that would sort angles descending
    sorted_desc_indices = np.argsort(angles)[::-1]

    # 2. Keep only the top `end_index` indices in descending-value order
    truncated_indices = sorted_desc_indices[:end_index]

    # 3. Compute uniform positions (using floor) between 0 and end_index-1
    #    We want to include positions 0 and end_index-1 exactly.
    step = (end_index - 1) / (num_views - 1)
    chosen_positions = [int(np.floor(step * i)) for i in range(num_views - 1)]
    chosen_positions.append(end_index - 1)  # ensure last position is included

    # 4. Map those positions back into the original indices
    uniform_indices = [int(truncated_indices[pos]) for pos in chosen_positions]

    return uniform_indices
