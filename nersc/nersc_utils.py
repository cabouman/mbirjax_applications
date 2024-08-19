import numpy as np

def create_circular_mask(height, width, center=None, radius=None):
    """ This function creates a 2D binary mask, which denotes the circular region specified by (height, width, center, radius).
    Args:
        height (int): height of the mask
        width (int): height of the mask
        center (tuple): [Default=None] central coordinates of the circle. If None, (int(width/2), int(height/2)) will be used as the center.
        radius (float): radius of the circle.
    Returns:
        3D boolean array denoting the circular region defined by center and radius. Any pixels inside the circular region will be marked with 1.
    """
    if center is None: # use the middle of the image
        center = (int(width/2), int(height/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], width-center[0], height-center[1])

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
