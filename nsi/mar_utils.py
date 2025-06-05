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
    Beam-hardening correction for plastic-metal case (linear plastic model only).

    H = [p, p*m, p*m^2, ..., p*m^(cross_order-1), m, m^2, ..., m^(metal_order-1), (optional const)]

    Args:
        ct_model: CT model object with forward_project().
        measured_sino: Raw sinogram.
        recon: Reconstruction used to compute masks.
        epsilon: Regularization parameter.
        order: list [cross_order, metal_order]
        include_const: whether to include constant term.

    Returns:
        corrected_sino: corrected sinogram.
    """
    plastic_mask, metal_mask, plastic_scale, metal_scale = mjp.segment_plastic_metal(recon)

    device = ct_model.main_device
    ideal_plastic_sino = plastic_scale * ct_model.forward_project(jax.device_put(plastic_mask, device)).reshape(-1)
    ideal_metal_sino = metal_scale * ct_model.forward_project(jax.device_put(metal_mask, device)).reshape(-1)
    y = measured_sino.reshape(-1)

    # Normalize projections
    p_normalization = jnp.max(jnp.abs(ideal_plastic_sino))
    m_normalization = jnp.max(jnp.abs(ideal_metal_sino))
    p = ideal_plastic_sino / p_normalization
    m = ideal_metal_sino / m_normalization

    cross_order, metal_order = order

    # Build H matrix
    H = [p * m ** i for i in range(cross_order)]  # p, p*m, ..., p*m^(cross_order-1)
    H += [m ** i for i in range(1, metal_order)]  # m, m^2, ..., m^(metal_order-1)

    if include_const:
        H.append(jnp.ones_like(p))  # constant term at the end

    order_total = len(H)
    HtH = jnp.zeros((order_total, order_total))
    Hty = jnp.zeros(order_total)

    for i in range(order_total):
        Hty = Hty.at[i].set(jnp.dot(H[i], y))
        for j in range(order_total):
            HtH = HtH.at[i, j].set(jnp.dot(H[i], H[j]))

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

    corrected_plastic_sino = p_normalization * numerator / linear_plastic_coef
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
        recon, _ = ct_model.recon(corrected_sinogram, weights=weights, init_recon=recon,
                                  stop_threshold_change_pct=stop_threshold_pct)

        if verbose > 0:
            print(f"\n************ BH Iteration {i + 1}: Display plastic and metal mask **************")
            plastic_mask, metal_mask, plastic_scale, metal_scale = mjp.segment_plastic_metal(recon)
            mj.slice_viewer(plastic_mask, metal_mask, vmin=0, vmax=1.0,
                            slice_label=['Plastic Mask', 'Metal Mask'],
                            title=f'Iteration {i + 1}: Comparison of Plastic and Metal Masks')

    return recon
