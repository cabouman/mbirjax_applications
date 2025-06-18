import jax.numpy as jnp
import mbirjax as mj
from nsi.mar_utils import apply_cylindrical_mask  # Adjust if your function is in a different module

# Create a 3D volume of ones with shape (rows, cols, slices)
shape = (128, 128, 300)
all_ones_volume = jnp.ones(shape, dtype=jnp.float32)

# Define mask parameters
radial_margin = 7
num_top_slices = 5
num_bottom_slices = 8

# Apply cylindrical mask
masked_volume = apply_cylindrical_mask(
    all_ones_volume,
    radial_margin=radial_margin,
    num_top_slices=num_top_slices,
    num_bottom_slices=num_bottom_slices
)

# Display original and masked volume side by side
mj.slice_viewer(all_ones_volume, masked_volume,
                title='Original vs Cylindrically Masked Volume',
                slice_label=['Original', 'Masked'])