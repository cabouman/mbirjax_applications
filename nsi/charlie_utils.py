import jax
import jax.numpy as jnp


def gen_ghuber_weights(weights, sino_error, T=1.0, delta=1.0, batch_size=16, epsilon=1e-6):
    """
    Generate generalized Huber weights.

    Args:
        weights: jnp.ndarray of shape (views, rows, cols)
        sino_error: jnp.ndarray of shape (views, rows, cols)
        T: float, threshold parameter where values > T are treated as outliers (default 1.0)
        delta: float, controls strength of generalized Huber function (delta=1 corresponds to conventional Huber) (default 1.0, must be between 0 and 1)
        batch_size: int, batch size for memory efficiency
        epsilon: float, small number to avoid division by zero

    Returns:
        ghuber_weights: jnp.ndarray of shape (views, rows, cols)

    Notes:
        The generalized Huber function used in this function is based on:
        Venkatakrishnan, S. V., Drummy, L. F., Jackson, M., De Graef, M., Simmons, J. P., and Bouman, C. A.,
        "Model-Based Iterative Reconstruction for Bright-Field Electron Tomography,"
        IEEE Transactions on Computational Imaging, vol. 1, no. 1, pp. 1–15, 2015. DOI: 10.1109/TCI.2014.2371751
    """
    if not (0.0 <= delta <= 1.0):
        raise ValueError("delta must be between 0 and 1.")

    # Ensure inputs are JAX arrays
    weights = jnp.asarray(weights)
    sino_error = jnp.asarray(sino_error)

    views, rows, cols = weights.shape

    def process_single_view(weight_slice, error_slice):
        # Compute standard deviation
        std = 1.0 / jnp.maximum(jnp.sqrt(weight_slice), epsilon)

        # Normalize the standard deviation
        numerator = jnp.sum(std * error_slice, axis=(0, 1), keepdims=True)
        denominator = jnp.linalg.norm(error_slice, axis=(0, 1), keepdims=True) + epsilon
        alpha = numerator / denominator
        std = std / alpha

        # Compute generalized Huber weights
        normalized_error = error_slice / std
        abs_normalized_error = jnp.abs(normalized_error)
        ghuber = jnp.where(abs_normalized_error <= T, 1.0, (delta * T) / abs_normalized_error)

        return ghuber

    ghuber_list = []

    for i in range(0, views, batch_size):
        weights_batch = weights[i:i+batch_size]
        error_batch = sino_error[i:i+batch_size]

        ghuber_batch = jax.vmap(process_single_view)(weights_batch, error_batch)

        ghuber_list.append(ghuber_batch)

    ghuber_weights = jnp.concatenate(ghuber_list, axis=0)

    return ghuber_weights


# Basic test to verify functionality
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    views, rows, cols = 4, 8, 8

    weights = jax.random.uniform(key, (views, rows, cols)) + 0.1
    key, subkey = jax.random.split(key)
    sino_error = 0.1 * jax.random.normal(subkey, (views, rows, cols))

    ghuber_weights = gen_ghuber_weights(weights, sino_error)

    assert ghuber_weights.shape == (views, rows, cols), "Output shape mismatch!"
    assert jnp.all(ghuber_weights > 0), "Found non-positive weights!"

    print("Sample ghuber_weights slice [view 0]:")
    print(ghuber_weights[0])
    print("\nTest passed! ✅")
