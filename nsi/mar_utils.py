import jax
import jax.numpy as jnp
import mbirjax


def gen_ghuber_weights(weights, sino_error, T=1.0, delta=1.0, epsilon=1e-6):
    """
    Generate generalized Huber weights.

    This function computes generalized Huber weights based on the method described in the referenced notes.
    It adds robustness by treating any element where |sino_error / weights| > T as an outlier,
    down-weighting it according to the generalized Huber function.

    The function returns new `ghuber_weights`.

    Typically, to obtain the final robust weights, the `ghuber_weights` should be multiplied by the original `weights`:

        final_weights = weights * ghuber_weights

    Args:
        weights: jnp.ndarray of shape (views, rows, cols)
            Initial weights, typically derived from inverse variance estimates.
        sino_error: jnp.ndarray of shape (views, rows, cols)
            Sinogram error array representing deviations from the model.
        T: float, optional (default=1.0)
            Threshold parameter; values greater than T are treated as outliers.
        delta: float, optional (default=1.0)
            Controls the strength of the generalized Huber function (delta=1 corresponds to the conventional Huber).
        epsilon: float, optional (default=1e-6)
            Small number to avoid division by zero.

    Returns:
        ghuber_weights: jnp.ndarray of shape (views, rows, cols)
            The computed generalized Huber weights.

    Notes:
        The generalized Huber function used in this function is based on:
        Venkatakrishnan, S. V., Drummy, L. F., Jackson, M., De Graef, M., Simmons, J. P., and Bouman, C. A.,
        "Model-Based Iterative Reconstruction for Bright-Field Electron Tomography,"
        IEEE Transactions on Computational Imaging, vol. 1, no. 1, pp. 1–15, 2015. DOI: 10.1109/TCI.2014.2371751

    Example:
        >>> from mar_utils import gen_ghuber_weights
        >>> ghuber_weights = gen_ghuber_weights(weights, sino_error, T=1.0)
        >>> final_weights = weights * ghuber_weights
    """
    if not (0.0 <= delta <= 1.0):
        raise ValueError("delta must be between 0 and 1.")

    weights = jnp.asarray(weights)
    sino_error = jnp.asarray(sino_error)

    # Compute std and global alpha
    std = 1.0 / jnp.maximum(jnp.sqrt(weights), epsilon)
    alpha = jnp.linalg.norm(sino_error) / (jnp.linalg.norm(std) + epsilon)
    std_norm = alpha * std

    # Compute normalized error
    normalized_error = sino_error / std_norm
    abs_norm_error = jnp.abs(normalized_error)

    # Apply generalized Huber function
    ghuber_weights = jnp.where(abs_norm_error <= T, 1.0, (delta * T) / (abs_norm_error + epsilon))

    return ghuber_weights


def beam_hardening_correction(sino, alpha, batch_size=16):
    """
    Apply a polynomial beam hardening correction to a sinogram.

    This function applies a polynomial correction to each view of the sinogram
    by evaluating powers of the sinogram values and weighting them by the coefficients in `alpha`,
    while also including the original linear term (the sinogram itself).

    The corrected sinogram is computed as:

        corrected_sino = sino + alpha[0] * sino**2 + alpha[1] * sino**3 + ...

    It processes the sinogram in batches of views for memory efficiency.

    Args:
        sino: jnp.ndarray or np.ndarray of shape (views, rows, cols)
            Input sinogram to correct.
        alpha: list or array of floats
            Coefficients for the polynomial correction. The k-th term corresponds to sino^(k+1).
        batch_size: int, optional (default=16)
            Number of views to process in a single batch.

    Returns:
        corrected_sino: jnp.ndarray of shape (views, rows, cols)
            Beam hardening corrected sinogram.

    Example:
        >>> from mar_utils import beam_hardening_correction
        >>> alpha = [1.0, 0.2, 0.1]  # Correction: sino + 0.2 * sino^2 + 0.1 * sino^3
        >>> corrected_sino = beam_hardening_correction(sino, alpha)
    """
    # Ensure inputs are JAX arrays
    sino = jnp.asarray(sino)
    alpha = jnp.asarray(alpha)

    views, rows, cols = sino.shape
    corrected = []

    for i in range(0, views, batch_size):
        sino_batch = sino[i:i+batch_size]

        # Initialize corrected batch to the linear term (sino_batch)
        corrected_batch = jnp.array(sino_batch)

        # Apply polynomial terms
        for k in range(len(alpha)):
            corrected_batch += alpha[k] * jnp.power(sino_batch, k + 1)

        corrected.append(corrected_batch)

    corrected_sino = jnp.concatenate(corrected, axis=0)

    return corrected_sino

def estimate_metal_sino(ct_model, sino, recon, metal_threshold=None, order=3, verbose=False):
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
        verbose: bool, default=False
            If True, displays diagnostics and prints internal quantities.

    Returns:
        bh_metal_sino: jnp.ndarray
            Estimated beam-hardened metal sinogram component.
        metal_mask: jnp.ndarray
            Binary mask of metal regions in the reconstruction.
        theta: jnp.ndarray
            Polynomial coefficients in physical (unnormalized) units.
    """
    if metal_threshold is None:
        if verbose:
            print("Metal threshold calculated using Otsu's method.")
        _, metal_threshold = mbirjax.multi_threshold_otsu(recon, classes=3)

    if verbose:
        print("metal_threshold =", metal_threshold)

    metal_mask = jnp.where(recon > metal_threshold, 1.0, 0.0).astype(jnp.float32)
    metal_mask = jax.device_put(metal_mask, device=ct_model.main_device)
    metal_sino = ct_model.forward_project(metal_mask)

    if verbose:
        print("metal_mask sum =", jnp.sum(metal_mask))
        print("metal_sino max =", jnp.max(metal_sino))

    s_flat = sino.reshape(-1)
    m_flat = metal_sino.reshape(-1)

    s_norm = jnp.linalg.norm(s_flat)
    m_norm = jnp.linalg.norm(m_flat)
    s_flat = s_flat / s_norm
    m_flat = m_flat / m_norm

    # Precompute powers of metal sinogram
    powers = [jnp.ones_like(m_flat)]
    for i in range(1, 2 * order + 1):
        powers.append(powers[-1] * m_flat)

    # Incremental computation of HᵀH and Hᵀs
    HtH = jnp.zeros((order, order), dtype=jnp.float32)
    Hts = jnp.zeros((order,), dtype=jnp.float32)

    for i in range(order):
        for j in range(i, order):
            val = jnp.sum(powers[i + 1] * powers[j + 1])
            HtH = HtH.at[i, j].set(val)
            HtH = HtH.at[j, i].set(val)
        Hts = Hts.at[i].set(jnp.sum(powers[i + 1] * s_flat))

    # Add regularization
    epsilon = 1e-7
    HtH += epsilon * jnp.linalg.norm(HtH) * jnp.eye(order)

    theta_scaled = jnp.linalg.solve(HtH, Hts)

    # Reconstruct beam-hardened metal sinogram
    bh_metal_sino_flat = sum(theta_scaled[i] * powers[i + 1] for i in range(order)) * s_norm
    bh_metal_sino = bh_metal_sino_flat.reshape(sino.shape)

    # De-normalize theta to make physically interpretable
    theta = jnp.array([
        theta_scaled[i] / (m_norm ** (i + 1)) * s_norm
        for i in range(order)
    ])

    if verbose:
        print("s_flat norm:", s_norm)
        print("m_flat norm:", m_norm)
        print("HtH =\n", HtH)
        print("theta =", theta)

    return bh_metal_sino, metal_mask, theta


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
