import jax
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp

__all__ = ["BHC_plastic_metal", "recon_BH_plastic_metal"]


@jax.jit
def _compute_scaling_factor(v: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the optimal scalar α that minimizes the squared error ‖v – α u‖².

    Args:
        v (jnp.ndarray):
            Target reconstruction array of shape (N,) or higher-dimensional.
        u (jnp.ndarray):
            Mask array of same shape as `v`, indicating component presence.

    Returns:
        jnp.ndarray:
            Scalar α minimizing ‖v – α u‖². Returns 0 if `u` is all zeros.

    Example:
        >>> v = jnp.array([1.0, 2.0, 3.0])
        >>> u = jnp.array([0.5, 1.0, 1.5])
        >>> alpha = _compute_scaling_factor(v, u)
    """
    v = jnp.asarray(v)
    u = jnp.asarray(u)

    numerator = jnp.sum(u * v)
    denominator = jnp.sum(u * u)
    return jnp.where(denominator == 0, 0.0, numerator / denominator)



def BHC_plastic_metal(ct_model, measured_sino, recon, epsilon=2e-4, order=(3, 4), include_const=False):
    """
    Beam-hardening correction for objects containing a combination of plastic and metal.

    The function takes the measured sinogram and initial reconstruction as input, and it returns a corrected sinogram.
    It is designed to reduce metal artifacts for scans of objects made from a combination of plastic and metal material.
    The metal and plastic materials are each assumed to be composed of a single material.
    However, it should work fine for a combination of different plastics as long as their optical density properites do not vary too much.

    Note:
        The corrected sinogram should result in a more accurate reconstruction of the plastic, but may not accurately reconstruct the metal portion.

    Args:
        ct_model:
            Object with `forward_project` method and `main_device` attribute.
        measured_sino (jnp.ndarray):
            Raw sinogram data of shape (views, rows, cols).
        recon (jnp.ndarray):
            Reconstructed volume array corresponding to `measured_sino`.
        epsilon (float, optional):
            Tolerance for regularization.

    Returns:
        corrected_sino (jnp.ndarray):
            Beam-hardening corrected sinogram, same shape as `measured_sino`.

    Example:
        >>> corrected = BHC_plastic_metal(ct_model, measured_sino, recon)
    """
    plastic_mask, metal_mask, plastic_scale, metal_scale = mjp.segment_plastic_metal(recon)

    # Forward projection
    device = ct_model.main_device
    ideal_plastic_sino = plastic_scale * ct_model.forward_project(jax.device_put(plastic_mask, device)).reshape(-1)
    ideal_metal_sino = metal_scale * ct_model.forward_project(jax.device_put(metal_mask, device)).reshape(-1)
    y = measured_sino.reshape(-1)

    # Compute normalized plastic and metal sinograms with max amplitude = 1
    p_normalization = jnp.max(jnp.abs(ideal_plastic_sino))
    m_normalization = jnp.max(jnp.abs(ideal_metal_sino))
    p = ideal_plastic_sino / p_normalization
    m = ideal_metal_sino / m_normalization

    # Set order of models
    cross_order, metal_order = order

    # Build H matrix
    H = [p * m ** i for i in range(cross_order)]  # p, p*m, ..., p*m^(cross_order-1)
    H += [m ** i for i in range(1, metal_order)]  # m, m^2, ..., m^(metal_order-1)

    # Include constant if desired
    if include_const:
        H.append(jnp.ones_like(p))  # constant term at the end

    # Compute H^t H and H^t y
    order_total = len(H)
    HtH = jnp.zeros((order_total, order_total))
    Hty = jnp.zeros(order_total)

    for i in range(order_total):
        Hty = Hty.at[i].set(jnp.dot(H[i], y))
        for j in range(order_total):
            HtH = HtH.at[i, j].set(jnp.dot(H[i], H[j]))

    # Regularize and solve for least square value of theta that minimizes || y - H theta ||^2
    sigma_max = jnp.linalg.norm(HtH, ord=2)
    HtH_reg = HtH + (epsilon ** 2) * sigma_max * jnp.eye(order_total)
    theta = jnp.linalg.solve(HtH_reg, Hty)

    # Separate metal terms
    metal_start_idx = cross_order
    metal_sino = jnp.zeros_like(y)
    for idx in range(metal_start_idx, order_total):
        metal_sino += theta[idx] * H[idx]

    # Build linear plastic scaling denominator
    linear_plastic_coef = jnp.zeros_like(p)
    for idx in range(cross_order):
        linear_plastic_coef += theta[idx] * (m ** idx)

    denom_floor = 1e-6 * jnp.linalg.norm(linear_plastic_coef)
    linear_plastic_coef = jnp.where(jnp.abs(linear_plastic_coef) > denom_floor, linear_plastic_coef, denom_floor)

    # Numerator: subtract metal + constant if included
    numerator = y - metal_sino
    if include_const:
        numerator -= theta[-1]

    # Compute BH corrected version of plastic sinogram with metal removed
    corrected_plastic_sino = p_normalization * numerator / linear_plastic_coef

    # Combine corrected plastic and metal sinogram and reshape
    corrected_sino_flat = corrected_plastic_sino + ideal_metal_sino
    corrected_sino = corrected_sino_flat.reshape(measured_sino.shape)

    return corrected_sino



def recon_BH_plastic_metal(ct_model, sino, weights, num_BH_iterations=3, stop_threshold_pct=0.5, verbose=0,
                           order=(3, 4), include_const=False):
    """
    Perform iterative metal artifact reduction using plastic-metal beam hardening correction.

    This function repeatedly applies `BHC_plastic_metal()` and reconstructs from the corrected
    sinogram to iteratively refine the reconstruction and reduce metal artifacts.

    Args:
        ct_model: MBIRJAX cone beam model instance used for reconstruction.
        sino (jnp.ndarray): Input sinogram data to be corrected.
        weights (jnp.ndarray): Transmission weights used in the reconstruction algorithm.
        num_BH_iterations (int, optional): Number of beam hardening correction and reconstruction iterations to perform. Defaults to 3.
        stop_threshold_pct (float, optional): Threshold for stopping reconstruction iterations based on relative change in reconstruction. Defaults to 0.5.
        verbose (int, optional): Verbosity level for printing intermediate information. Defaults to 0.
        order (list, optional):
            List of two integers specifying the order of polynomial terms for plastic and metal components respectively.
            Defaults to [3, 4].
        include_const (bool, optional):
            Whether to include a constant term in the model. Defaults to False.

    Returns:
        jnp.ndarray: The final corrected reconstruction after iterative beam hardening correction.

    Example:
        >>> recon = recon_BH_plastic_metal(ct_model, sino, weights, num_BH_iterations=3, verbose=1, order=[3,4], include_const=False)
        >>> mj.slice_viewer(recon)
    """
    if verbose > 0:
        print("\n********* Perform initial FDK reconstruction **********")
    recon = ct_model.direct_recon(sino)

    for i in range(num_BH_iterations):
        # Estimate Corrected Sinogram
        corrected_sinogram = BHC_plastic_metal(ct_model, sino, recon, order=order, include_const=include_const)

        # Reconstruct Corrected Sinogram
        recon, recon_dict = ct_model.recon(corrected_sinogram, weights=weights, init_recon=recon,
                                  stop_threshold_change_pct=stop_threshold_pct)

        if verbose > 0:
            print(f"\n************ BH Iteration {i + 1}: Display plastic and metal mask **************")
            plastic_mask, metal_mask, plastic_scale, metal_scale = mjp.segment_plastic_metal(recon)
            mj.slice_viewer(plastic_mask, metal_mask,  vmin=0, vmax=1.0,
                            slice_label=['Plastic Mask', 'Metal Mask'],
                            title=f'Iteration {i + 1}: Comparison of Plastic and Metal Masks')

    return recon

def apply_cylindrical_mask(recon: jnp.ndarray, radial_margin: int, num_axial_slices: int):
    """
    Apply a cylindrical mask to a 3D volume:
    - In each (row, col) slice, zero out pixels outside a centered circular region.
    - Along the slice (Z) axis, zero out a fixed number of slices from both top and bottom.

    Args:
        recon (jnp.ndarray): 3D volume of shape (rows, cols, slices).
        radial_margin (int): Number of pixels to subtract from the circular radius (row-col plane).
        num_axial_slices (int): Number of slices to zero from both top and bottom along the Z-axis.

    Returns:
        jnp.ndarray: Masked volume with out-of-cylinder and edge slices set to zero.
    """
    num_recon_rows, num_recon_cols, num_slices = recon.shape
    row_center = (num_recon_rows - 1) / 2
    col_center = (num_recon_cols - 1) / 2

    base_radius = max(row_center, col_center)
    radius = base_radius - radial_margin

    # Create circular mask in (row, col) plane
    row_coords, col_coords = jnp.meshgrid(jnp.arange(num_recon_rows), jnp.arange(num_recon_cols), indexing='ij')
    dist_sq = (row_coords - row_center) ** 2 + (col_coords - col_center) ** 2
    circular_mask = (dist_sq <= radius ** 2).astype(recon.dtype)

    # Apply cylindrical mask to all slices
    recon = recon * circular_mask[:, :, None]

    # Zero out top and bottom slices along Z
    if num_axial_slices > 0:
        recon = recon.at[:, :, :num_axial_slices].set(0)
        recon = recon.at[:, :, -num_axial_slices:].set(0)

    return recon