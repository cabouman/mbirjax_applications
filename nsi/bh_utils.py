import jax
import jax.numpy as jnp
import mbirjax


def estimate_metal_sino(ct_model, sino, recon, metal_threshold=None, order=3):
    """
    Estimate the component of the sinogram due to metal via a polynomial beam hardening model.

    Args:
        ct_model: MBIR CT model object with a forward_project method.
        sino: jnp.ndarray of shape (views, rows, cols)
            Input sinogram containing metal.
        recon: jnp.ndarray of shape (slices, rows, cols)
            Initial reconstruction of the image (NOT the sinogram).
        metal_threshold: float
            Threshold to identify metal in the reconstruction. If None, uses Otsu.
        order: int, default=3
            Order of the polynomial beam hardening model.

    Returns:
        bh_metal_sino: jnp.ndarray
            Beam-hardened metal sinogram approximation.
        theta: jnp.ndarray
            Polynomial coefficients of shape (order,)
    """
    if metal_threshold is None:
        print("Metal threshold calculated using Otsu's method.")
        _, metal_threshold = mbirjax.multi_threshold_otsu(recon, classes=3)

    print("metal_threshold =", metal_threshold)

    try:
        mbirjax.slice_viewer(recon, slice_axis=2, slice_label='Recon Slices', title='Recon Slices')
    except Exception as e:
        print("Viewer failed for recon:", e)

    # Segment the metal in the reconstruction
    metal_mask = 1.0 * jnp.where(recon > metal_threshold, 1.0, 0.0).astype(jnp.float32)
    metal_mask = jax.device_put(metal_mask, device=ct_model.main_device)

    # Forward project the metal to sinogram space
    metal_sino = ct_model.forward_project(metal_mask)

    print("metal_mask sum =", jnp.sum(metal_mask))
    print("metal_sino max =", jnp.max(metal_sino))

    try:
        mbirjax.slice_viewer(metal_mask, slice_axis=2, slice_label='Metal Slices', title='Metal Mask Slices')
        mbirjax.slice_viewer(metal_sino, sino, slice_axis=0, slice_label='Metal Sino', slice_label2='Sino', title='Views')
    except Exception as e:
        print("Viewer failed for masks/sinograms:", e)

    # Flatten for least squares over all pixels
    s_flat = 1.0*sino.reshape(-1)
    m_flat = metal_sino.reshape(-1)

    print("s_flat norm:", jnp.linalg.norm(s_flat))
    print("m_flat norm:", jnp.linalg.norm(m_flat))

    # Build design matrix H = [m, m**2, ..., m**order]
    H = jnp.stack([m_flat**i for i in range(1, order + 1)], axis=1)  # shape: (N, order)
    print("H column norms:", jnp.linalg.norm(H, axis=0))

    # Regularization parameter (small to avoid singular matrix)
    lambda_reg = 1e-6

    # Compute normal equations
    HtH = H.T @ H + lambda_reg * jnp.eye(order)
    Hts = H.T @ s_flat

    # Print HᵀH for inspection
    print("HtH =\n", HtH)

    # Solve the linear system: (HᵀH + λI) θ = Hᵀs
    theta = jnp.linalg.solve(HtH, Hts)

    if jnp.isnan(theta).any():
        print("WARNING: NaNs detected in theta. H may be rank-deficient or poorly scaled.")

    # Reconstruct beam-hardened metal sinogram
    bh_metal_sino_flat = H @ theta
    bh_metal_sino = bh_metal_sino_flat.reshape(sino.shape)

    print("theta =", theta)

    return bh_metal_sino, theta