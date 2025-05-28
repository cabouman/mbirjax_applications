import jax
import jax.numpy as jnp
import mbirjax as mj


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


def correct_sino_for_metal(ct_model, measured_sino, recon, epsilon=2e-4):
    """
    Correct a measured sinogram for beam-hardening artifacts due to metal objects.

    This function first estimates the parameters of a beam-hardening forward model by segmenting the reconstruction
    into approximate plastic and metal components. Then using that model, it estimates the beam-harden corrected
    plastic component of the sinogram, and adds it to the beam-harden corrected metal component to produce
    a total beam-harden corrected sinogram.

    The corrected sinogram should result in a more accurate reconstruction of the plastic,
    but may not accurately reconstruct the metal portion.

    Note:
        This function can be applied repeatedly for improved reconstruction quality.

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
        metal_mask (jnp.ndarray):
            Binary mask array for metal regions in `recon`.
        plastic_mask (jnp.ndarray):
            Binary mask array for plastic regions in `recon`.

    Example:
        >>> corrected, metal_m, plastic_m = correct_sino_for_metal(ct_model, measured_sino, recon)
    """
    # Determine class thresholds based on the 5-classes
    thresholds = mj.multi_threshold_otsu(recon, classes=5)
    plastic_low_threshold = thresholds[1]
    plastic_high_threshold = thresholds[2]
    metal_threshold = thresholds[3]

    # Create masks
    plastic_mask = jnp.where((recon > plastic_low_threshold) & (recon <= plastic_high_threshold), 1.0, 0.0)
    metal_mask = jnp.where(recon > metal_threshold, 1.0, 0.0)

    # Scale factors
    plastic_scale = _compute_scaling_factor(recon, plastic_mask)
    metal_scale = _compute_scaling_factor(recon, metal_mask)

    # Forward projection
    device = ct_model.main_device
    p_raw = ct_model.forward_project(jax.device_put(plastic_mask, device)).reshape(-1)
    m_raw = ct_model.forward_project(jax.device_put(metal_mask, device)).reshape(-1)
    y = measured_sino.reshape(-1)

    # Normalize to max amplitude = 1
    p_scaled = plastic_scale * p_raw
    m_scaled = metal_scale * m_raw
    p_norm = p_scaled / jnp.maximum(jnp.max(jnp.abs(p_scaled)), 1e-8)
    m_norm = m_scaled / jnp.maximum(jnp.max(jnp.abs(m_scaled)), 1e-8)

    # Build H and compute H^T H, H^T y
    H = [p_norm,
         p_norm * m_norm,
         p_norm * m_norm**2,
         m_norm,
         m_norm**2,
         m_norm**3]
    n = len(H)

    HtH = jnp.zeros((n, n))
    Hty = jnp.zeros(n)
    for i in range(n):
        Hty = Hty.at[i].set(jnp.dot(H[i], y))
        for j in range(n):
            HtH = HtH.at[i, j].set(jnp.dot(H[i], H[j]))

    # Regularize and solve
    sigma_max = jnp.linalg.norm(HtH, ord=2)
    HtH_reg = HtH + (epsilon**2) * sigma_max * jnp.eye(n)
    theta = jnp.linalg.solve(HtH_reg, Hty)

    # Separate components and recover plastic sinogram
    theta_p, theta_m = theta[:3], theta[3:]
    H_m = theta_m[0]*H[3] + theta_m[1]*H[4] + theta_m[2]*H[5]

    denom = theta_p[0] + theta_p[1]*m_norm + theta_p[2]*m_norm**2
    denom_floor = 1e-6 * jnp.linalg.norm(denom)
    denom = jnp.where(jnp.abs(denom) > denom_floor, denom, denom_floor)

    p_hat = p_scaled * (y - H_m) / denom

    # Recombine and reshape
    corrected_flat = p_hat + m_scaled
    corrected_sino = corrected_flat.reshape(measured_sino.shape)

    return corrected_sino, metal_mask, plastic_mask