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


def correct_sino_for_metal(
    ct_model,
    measured_sino: jnp.ndarray,
    recon: jnp.ndarray,
    verbose: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Correct a measured sinogram for beam-hardening artifacts due to metal objects.

    This method models beam hardening as a polynomial function of metal and plastic components:
    1. Segment `recon` into plastic and metal masks via Otsu's method.
    2. Estimate scale factors matching `recon` to each mask.
    3. Forward-project scaled masks to obtain sinogram components `p` and `m`.
    4. Normalize to unit peak amplitude for stability.
    5. Build design matrix H = [p, p*m, p*m^2, m, m^2, m^3] and solve
       y ≈ H θ via Tikhonov-regularized least squares.
    6. Recover plastic-only sinogram and recombine with metal component.

    Args:
        ct_model:
            Object with `forward_project` method and `main_device` attribute.
        measured_sino (jnp.ndarray):
            Raw sinogram data of shape (views, rows, cols).
        recon (jnp.ndarray):
            Reconstructed volume array corresponding to `measured_sino`.
        verbose (int, optional):
            Verbosity level; 0 = silent, 1 = print Otsu thresholds. Defaults to 0.

    Returns:
        corrected_sino (jnp.ndarray):
            Beam-hardening corrected sinogram, same shape as `measured_sino`.
        metal_mask (jnp.ndarray):
            Binary mask array for metal regions in `recon`.
        plastic_mask (jnp.ndarray):
            Binary mask array for plastic regions in `recon`.

    Notes:
        - The function is not JIT-compiled to simplify debugging. Remove this note
          or add `@jax.jit` if you wish to re-enable JIT.
        - Type hints are purely for static analysis and do not affect runtime.

    Example:
        >>> corrected, metal_m, plastic_m = correct_sino_for_metal(
        ...     ct_model, measured_sino, recon, verbose=1
        ... )
    """
    # 1. Segment into classes
    if verbose:
        print("Computing Otsu thresholds...")
    thresholds = mj.multi_threshold_otsu(recon, classes=5)
    p_th, pm_th, mh_th, m_th = thresholds[1], thresholds[2], thresholds[3], thresholds[4]

    plastic_mask = jnp.where((recon > p_th) & (recon <= mh_th), 1.0, 0.0)
    metal_mask = jnp.where(recon > m_th, 1.0, 0.0)

    # 2. Scale factors
    plastic_scale = _compute_scaling_factor(recon, plastic_mask)
    metal_scale = _compute_scaling_factor(recon, metal_mask)

    # 3. Forward projection
    device = ct_model.main_device
    p_raw = ct_model.forward_project(jax.device_put(plastic_mask, device)).reshape(-1)
    m_raw = ct_model.forward_project(jax.device_put(metal_mask, device)).reshape(-1)
    y = measured_sino.reshape(-1)

    # 4. Normalize to max amplitude = 1
    p_scaled = plastic_scale * p_raw
    m_scaled = metal_scale * m_raw
    p_norm = p_scaled / jnp.maximum(jnp.max(jnp.abs(p_scaled)), 1e-8)
    m_norm = m_scaled / jnp.maximum(jnp.max(jnp.abs(m_scaled)), 1e-8)

    # 5. Build H and compute H^T H, H^T y
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

    # 6. Regularize and solve
    lambda_reg = 2e-4
    sigma_max = jnp.linalg.norm(HtH, ord=2)
    HtH_reg = HtH + (lambda_reg**2) * sigma_max * jnp.eye(n)
    theta = jnp.linalg.solve(HtH_reg, Hty)

    # 7. Separate components and recover plastic sinogram
    theta_p, theta_m = theta[:3], theta[3:]
    H_m = theta_m[0]*H[3] + theta_m[1]*H[4] + theta_m[2]*H[5]

    denom = theta_p[0] + theta_p[1]*m_norm + theta_p[2]*m_norm**2
    denom = jnp.where(jnp.abs(denom) > 1e-6, denom, 1e-6)

    p_hat = p_scaled * (y - H_m) / denom

    # 8. Recombine and reshape
    corrected_flat = p_hat + m_scaled
    corrected_sino = corrected_flat.reshape(measured_sino.shape)

    return corrected_sino, metal_mask, plastic_mask