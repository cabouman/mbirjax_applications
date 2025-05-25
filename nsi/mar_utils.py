import jax
import jax.numpy as jnp
import mbirjax as mj


def estimate_metal_sino(ct_model, sino, recon, metal_threshold=None, order=3, verbose=0):
    """
    Estimate the component of the sinogram due to metal using a polynomial beam hardening model,
    optimized to avoid large memory usage by computing HᵀH and Hᵀs incrementally.

    Args:
        ct_model: MBIR CT model object with a `forward_project` method.
        sino: jnp.ndarray of shape (views, rows, cols)
            Input sinogram containing metal artifacts.
        recon: jnp.ndarray of shape (slices, rows, cols)
            Reconstructed image used to segment metal.
        metal_threshold: float, optional
            Intensity threshold to segment metal. If None, computed using 3-class Otsu threshold.
        order: int, default=3
            Order of the polynomial used to model beam hardening effects.
        verbose: int, default=1
            0 => silent, 1 => verbose.

    Returns:
        bh_metal_sino: jnp.ndarray
            Estimated beam-hardened metal sinogram component.
        metal_mask: jnp.ndarray
            Binary mask of metal regions in the reconstruction.
    """
    if metal_threshold is None:
        if verbose == 1:
            print("Metal threshold calculated using Otsu's method.")
        _, metal_threshold = mj.multi_threshold_otsu(recon, classes=3)

    if verbose == 1:
        print("metal_threshold =", metal_threshold)

    metal_mask = jnp.where(recon > metal_threshold, 1.0, 0.0).astype(jnp.float32)
    metal_mask = jax.device_put(metal_mask, device=ct_model.main_device)
    metal_sino = ct_model.forward_project(metal_mask)

    if verbose == 1:
        print("metal_mask sum =", jnp.sum(metal_mask))
        print("metal_sino max =", jnp.max(metal_sino))

    s_flat = sino.reshape(-1)
    m_flat = metal_sino.reshape(-1)

    # Normalize m_flat to be in the range [0,1]
    m_flat = m_flat / jnp.max(jnp.abs(m_flat))

    # Incremental computation of HᵀH and Hᵀs
    HtH = jnp.zeros((order, order), dtype=jnp.float32)
    Hts = jnp.zeros((order,), dtype=jnp.float32)

    for i in range(order):
        for j in range(i, order):
            val = jnp.sum(m_flat ** (i + 1 + j + 1))
            HtH = HtH.at[i, j].set(val)
            HtH = HtH.at[j, i].set(val)
        val = jnp.sum(m_flat ** (i +1) * s_flat)
        Hts = Hts.at[i].set(val)

    # Add regularization
    epsilon = 1e-7
    HtH += epsilon * jnp.linalg.norm(HtH) * jnp.eye(order)

    # compute theta
    theta = jnp.linalg.solve(HtH, Hts)

    # Reconstruct beam-hardened metal sinogram
    bh_metal_sino_flat = sum(theta[i] * m_flat ** (i + 1) for i in range(order))
    bh_metal_sino = bh_metal_sino_flat.reshape(sino.shape)

    if verbose == 1:
        print("HtH =\n", HtH)
        print("theta =", theta)

    return bh_metal_sino, metal_mask


def make_gaussian_kernel(sigma, size=None):
    if size is None:
        size = int(2 * 3 * sigma + 1)  # Cover approximately +/- 3 sigma
    coords = jnp.arange(size) - (size - 1) / 2
    gauss_1d = jnp.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss_1d /= gauss_1d.sum()
    kernel_2d = jnp.outer(gauss_1d, gauss_1d)
    kernel_2d /= jnp.sum(kernel_2d)  # Normalize to sum to 1.0
    return kernel_2d


def gaussian_blur(image, sigma):
    kernel = make_gaussian_kernel(sigma)
    kernel = kernel[:, :, None, None]  # Shape (H, W, in_channels=1, out_channels=1)
    image = image[None, :, :, None]    # Shape (batch=1, H, W, channels=1)
    blurred = lax.conv_general_dilated(
        image,
        kernel,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    return blurred[0, :, :, 0]  # Remove batch and channel dims


def scatter_correction(sino, alpha, beta, sigma, batch_size=16, atten_factor=4):
    """
    Apply scatter correction to the input sinogram.

    Args:
        sino: jnp.ndarray of shape (views, rows, cols)
        alpha: float, beam hardening correction parameter
        beta: float, scatter correction parameter
        sigma: float, standard deviation for gaussian blur
        batch_size: int, number of views to process at a time
        atten_factor: float, factor to divide min attenuation, default is 4

    Returns:
        corrected_sino: jnp.ndarray of shape (views, rows, cols)
    """
    views, rows, cols = sino.shape

    # Step 1: Compute global min attenuation
    max_sino = jnp.max(sino)
    min_attenuation = jnp.exp(-max_sino) / atten_factor

    corrected = []

    for i in range(0, views, batch_size):
        sino_batch = sino[i:i+batch_size]

        # Step 2: Scatter correction
        attenuation = jnp.exp(-sino_batch)

        one_minus_attenuation = 1.0 - attenuation
        blurred = jax.vmap(lambda img: gaussian_blur(img, sigma))(one_minus_attenuation)

        scatter = beta * blurred

        corrected_attenuation = attenuation - scatter

        # Clip corrected attenuation to minimum attenuation value
        corrected_attenuation = jnp.maximum(corrected_attenuation, min_attenuation)

        corrected_batch = -jnp.log(corrected_attenuation)

        corrected.append(corrected_batch)

    corrected_sino = jnp.concatenate(corrected, axis=0)

    return corrected_sino

def estimate_metal_sino_multi_material_cross(ct_model, sino, recon, verbose=0):
    """
    Estimate the beam-hardened sinogram components using a polynomial model
    with interaction terms and memory-efficient computation of HᵀH and Hᵀs.
    Modified to use 5-class Otsu.

    Returns:
        bh_metal_sino, bh_combined_sino, metal_mask, plastic_mask
    """

    # --- Thresholding ---

    if verbose == 1:
        print("Thresholds calculated using Otsu's method.")
    thresholds = mj.multi_threshold_otsu(recon, classes=5)
    mr_low_threshold = thresholds[0]
    plastic_threshold = thresholds[1]
    mr_high_threshold = thresholds[2]
    metal_threshold = thresholds[3]


    # Create masks based on the 5-class segmentation
    metal_mask = jnp.where(recon > metal_threshold, 1.0, 0.0)
    plastic_mask = jnp.where((recon > plastic_threshold) & (recon <= mr_high_threshold), 1.0, 0.0)
    artifact_mask_low = jnp.where((recon > mr_low_threshold) & (recon <= plastic_threshold), 1.0, 0.0)
    artifact_mask_high = jnp.where((recon > mr_high_threshold) & (recon <= metal_threshold), 1.0, 0.0)

    # Combine artifact masks
    artifact_mask = artifact_mask_low + artifact_mask_high

    metal_mask = jax.device_put(metal_mask, device=ct_model.main_device)
    plastic_mask = jax.device_put(plastic_mask, device=ct_model.main_device)

    m = ct_model.forward_project(metal_mask).reshape(-1)
    p = ct_model.forward_project(plastic_mask).reshape(-1)
    s = sino.reshape(-1)

    # Normalize m, p, and a
    m = m / jnp.max(jnp.abs(m)) if jnp.max(jnp.abs(m)) > 0 else m
    p = p / jnp.max(jnp.abs(p)) if jnp.max(jnp.abs(p)) > 0 else p

    # Construct H matrix: [p, p^2, p*m, p*m^2, m, m^2, m^3]
    H_cols = [p, p**2, p*m, p*(m**2), m, m**2, m**3]
    n_cols = len(H_cols)

    # --- Memory-efficient HᵀH and Hᵀs computation ---
    HtH = jnp.zeros((n_cols, n_cols))
    Hts = jnp.zeros(n_cols)
    for i in range(n_cols):
        col_i = H_cols[i]
        Hts = Hts.at[i].set(jnp.sum(col_i * s))
        for j in range(n_cols):
            col_j = H_cols[j]
            HtH = HtH.at[i, j].set(jnp.sum(col_i * col_j))

    # --- Regularization and Solve ---
    lambda_reg = 1e-7
    HtH += lambda_reg * jnp.linalg.norm(HtH) * jnp.eye(n_cols)
    theta_star = jnp.linalg.solve(HtH, Hts)

    theta_p = jnp.array([theta_star[0], theta_star[1], 0.0, 0.0, 0.0, 0.0, 0.0])

    # --- Compute Hθ and Hθ_p without forming H ---
    s_pm_hat = sum(theta_star[k] * H_cols[k] for k in range(n_cols))
    s_p_hat = sum(theta_p[k] * H_cols[k] for k in range(n_cols))

    # Final metal-corrected sinogram
    bh_combined_sino = s_pm_hat.reshape(sino.shape)
    bh_metal_sino = (s_pm_hat - s_p_hat).reshape(sino.shape)

    if verbose:
        print("theta =", theta_star)
        print("Plastic mask sum:", jnp.sum(plastic_mask))
        print("Metal mask sum:", jnp.sum(metal_mask))

    return bh_metal_sino, bh_combined_sino, metal_mask, plastic_mask, artifact_mask