
import jax
import jax.numpy as jnp
import mbirjax as mj

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


def BHC_plastic_metal(ct_model, measured_sino, recon, epsilon=2e-4):
    """
    Beam-hardening correction for objects containing a combination of plastic and metal.
    The function takes the measured sinogram and initial reconstruction as input, and it returns a corrected sinogram.

    This function is designed to reduce metal artifacts for scans of objects made from a combination of plastic and metal material.
    The metal and plastic materials are each assumed to be composed of a single material.
    However, it should work fine for a combination of different plastics as long as their optical density properites do not vary too much.


    The function first segments the reconstruction into approximate homogeneous plastic and metal components using the Otsu algorithm.
    Next, it forward projects the plastic and metal segmentations to form idealized plastic and metal sinogram.
    It then estimates the parameters of a polynomial beam-hardening function by fitting the beam-hardened idealized plastic and metal
    sinogram to the measured sinogram.
    Once the BH parameters are estimated, it then estimates the corrected sinogram from the measured sinogram, and returns it.

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
    plastic_mask, metal_mask, plastic_scale, metal_scale = seg_plastic_metal(recon)

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

    # Form NxP matrix, H, as list of columns
    H = [
        p,            # H[0]
        p * m,        # H[1]
        p * m ** 2,   # H[2]
        m,            # H[3]
        m ** 2,       # H[4]
        m ** 3        # H[5]
    ]
    order = len(H)

    # Compute HtH and H^t y
    HtH = jnp.zeros((order, order))
    Hty = jnp.zeros(order)
    for i in range(order):
        Hty = Hty.at[i].set(jnp.dot(H[i], y))
        for j in range(order):
            HtH = HtH.at[i, j].set(jnp.dot(H[i], H[j]))

    # Regularize and solve for least square value of theta that minimizes || y - H theta ||^2
    sigma_max = jnp.linalg.norm(HtH, ord=2)
    HtH_reg = HtH + (epsilon**2) * sigma_max * jnp.eye(order)
    theta = jnp.linalg.solve(HtH_reg, Hty)

    # Separate components and recover plastic sinogram
    only_metal_BH_sinogram = theta[3]*H[3] + theta[4]*H[4] + theta[5]*H[5]

    # Compute the linear coefficient for plastic with the assumed ideal metal
    linear_plastic_coef = theta[0] + theta[1]*m + theta[2]*m**2
    denom_floor = 1e-6 * jnp.linalg.norm(linear_plastic_coef)
    linear_plastic_coef = jnp.where(jnp.abs(linear_plastic_coef) > denom_floor, linear_plastic_coef, denom_floor)

    # Compute BH corrected version of plastic sinogram with metal removed
    corrected_plastic_sino = p_normalization * (y - only_metal_BH_sinogram) / linear_plastic_coef

    # Combine corrected plastic and metal sinogram and reshape
    corrected_sino_flat = corrected_plastic_sino + ideal_metal_sino
    corrected_sino = corrected_sino_flat.reshape(measured_sino.shape)

    return corrected_sino


def recon_BH_plastic_metal(ct_model, sino, weights, num_BH_iterations=3, stop_threshold=0.5, verbose=0):
    """
    Perform iterative metal artifact reduction using plastic-metal beam hardening correction.

    This function repeatedly applies `BHC_plastic_metal()` and reconstructs from the corrected
    sinogram to iteratively refine the reconstruction and reduce metal artifacts.

    Args:
        ct_model: MBIRJAX cone beam model instance.
        sino (jnp.ndarray): Input sinogram.
        weights (jnp.ndarray): Transmission weights for reconstruction.
        num_BH_iterations (int, optional): Number of correction-reconstruction iterations. Defaults to 4.

    Returns:
        jnp.ndarray: Final corrected reconstruction.
    """
    print("\n********* Perform initial FDK reconstruction **********")
    recon = ct_model.direct_recon(sino)

    for i in range(num_BH_iterations):
        print(f"\n************ BH Iteration {i + 1}: Estimate Corrected Sinogram **************")
        corrected_sinogram = BHC_plastic_metal(ct_model, sino, recon)

        if verbose > 0:
            print(f"\n************ BH Iteration {i + 1}: Display plastic and metal mask **************")
            plastic_mask, metal_mask, plastic_scale, metal_scale = seg_plastic_metal(recon)
            mj.slice_viewer(plastic_mask, metal_mask, vmin=0, vmax=1.0, slice_label=['Plastic Mask', 'Metal Mask'], title=f'Iteration {i + 1}: Comparison of Plastic and Metal Masks')

        print(f"\n********** BH Iteration {i + 1}: Reconstruct Corrected Sinogram *************")
        recon, _ = ct_model.recon(corrected_sinogram, weights=weights, init_recon=recon, stop_threshold_change_pct=stop_threshold)

    return recon
