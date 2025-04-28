import jax
import jax.numpy as jnp
import numpy as np

import mar_utils

def test_gen_ghuber_weights_basic():
    key = jax.random.PRNGKey(0)
    views, rows, cols = 4, 8, 8

    weights = jax.random.uniform(key, (views, rows, cols)) + 0.1
    key, subkey = jax.random.split(key)
    sino_error = 0.1 * jax.random.normal(subkey, (views, rows, cols))

    ghuber_weights = mar_utils.gen_ghuber_weights(weights, sino_error)

    assert ghuber_weights.shape == (views, rows, cols), "Output shape mismatch!"
    assert jnp.all(ghuber_weights > 0), "Found non-positive weights!"

    print("test_gen_ghuber_weights_basic passed! ✅")

def test_gen_ghuber_weights_outliers():
    key = jax.random.PRNGKey(42)
    views, rows, cols = 4, 8, 8

    weights = jax.random.uniform(key, (views, rows, cols)) + 0.1
    key, subkey = jax.random.split(key)
    sino_error = 0.1 * jax.random.normal(subkey, (views, rows, cols))

    outlier_indices = (jax.random.randint(subkey, (5,), 0, views),
                       jax.random.randint(subkey, (5,), 0, rows),
                       jax.random.randint(subkey, (5,), 0, cols))
    sino_error = sino_error.at[outlier_indices].set(5.0)

    ghuber_weights = mar_utils.gen_ghuber_weights(weights, sino_error)

    outlier_weights = ghuber_weights[outlier_indices]

    assert jnp.all(outlier_weights < 1.0), "Outlier weights should be clipped below 1.0!"

    print("test_gen_ghuber_weights_outliers passed! ✅")

def test_beam_hardening_correction_basic():
    key = jax.random.PRNGKey(1)
    views, rows, cols = 2, 4, 4

    sino = jax.random.uniform(key, (views, rows, cols))
    alpha = [0.2, 0.1]  # 0.2 * sino^2 + 0.1 * sino^3

    corrected_sino = mar_utils.beam_hardening_correction(sino, alpha)

    assert corrected_sino.shape == (views, rows, cols), "Corrected sinogram shape mismatch!"

    # Check that corrected_sino is non-negative if sino is non-negative
    assert jnp.all(corrected_sino >= 0), "Corrected sinogram should be non-negative for non-negative input!"

    print("test_beam_hardening_correction_basic passed! ✅")

if __name__ == "__main__":
    test_gen_ghuber_weights_basic()
    test_gen_ghuber_weights_outliers()
    test_beam_hardening_correction_basic()
