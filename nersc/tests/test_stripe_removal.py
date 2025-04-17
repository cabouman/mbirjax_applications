import unittest
import numpy as np
import jax.numpy as jnp
import mbirjax
from nersc import stripe_removal
from nersc import stripe_removal_jax

class TestStripeRemoval(unittest.TestCase):
    """
    Unite test for stripe removal functions
    Tests the consistency of stripe removal results between the JAX and NumPy implementations.
    """

    @staticmethod
    def add_vertical_stripes(sinogram, num_stripes=300, strength=15.0, strength_variation=(0.5, 1.5)):
        """
        Add vertical stripe artifacts to a 2D sinogram

        Args:
            sinogram (numpy array): a 2D slice of the sinogram data with shape (num_views, num_detector_channels)
            num_stripes (int): number of stripes to add
            strength (float): strength of the stripe
            strength_variation (tuple): A tuple (min_factor, max_factor) to randomly scale the strength for each stripe

        Returns:
            artifact_sinogram (numpy array): a 2D sinogram with added vertical stripes

        """

        artifact_sinogram = np.array(sinogram).copy()
        views, num_det_channels = sinogram.shape

        stripe_indices = np.random.choice(num_det_channels, num_stripes, replace=False)

        for idx in stripe_indices:
            current_strength = strength
            if strength_variation is not None:
                min_factor, max_factor = strength_variation
                scale = np.random.uniform(min_factor, max_factor)
                current_strength *= scale

            artifact_sinogram[:, idx] += current_strength

        return artifact_sinogram

    def setUp(self):
        """ Set up parameters and intialize models before each test."""
        # Sinogram parameters
        self.num_views = 640
        self.num_det_rows = 10
        self.num_det_channels = 1280
        self.sinogram_shape = (self.num_views, self.num_det_rows, self.num_det_channels)

        # Geometry parameters
        start_angle = -jnp.pi
        end_angle = jnp.pi
        self.angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False)

        # Initialize models
        self.parallel_model = mbirjax.ParallelBeamModel(self.sinogram_shape, self.angles)

        # Generate 3D Shepp-Logan phantom and sinogram
        self.phantom = self.parallel_model.gen_modified_3d_sl_phantom()
        self.sino = self.parallel_model.forward_project(self.phantom)

        # Set the tolerances for the difference
        self.remove_all_stripe_tolerance = {'difference': 0.001 * (jnp.max(self.sino) - jnp.min(self.sino))}
        self.remove_stripe_fw_tolerance = {'difference': 0.001 * (jnp.max(self.sino) - jnp.min(self.sino))}

    def test_remove_all_stripe(self):
        """Test the JAX version remove_all_stripe method against Numpy version."""

        # Generate sinogram with added vertical stripes
        artifact_sinogram = np.ones_like(self.sino)
        for m in range(self.sino.shape[1]):
            sino_slice = self.sino[:, m, :]
            artifact_sino_slice = self.add_vertical_stripes(sino_slice)
            artifact_sinogram[:, m, :] = artifact_sino_slice

        artifact_sinogram_jax = jnp.array(artifact_sinogram)
        # Remove the stripe using both JAX version and NumPy version
        corrected_sino_numpy = stripe_removal.remove_all_stripe(artifact_sinogram)
        corrected_sino_jax = stripe_removal_jax.remove_all_stripe_jax(artifact_sinogram_jax)

        # Compute the statistics
        mean_diff = np.mean(np.abs(corrected_sino_jax - corrected_sino_numpy))

        # Verify that the computed stats are within tolerances
        self.assertTrue(mean_diff < self.remove_all_stripe_tolerance['difference'], f"Mean difference too high: {mean_diff}")

    def test_remove_stripe_fw(self):
        """Test the JAX version remove_stripe_fw method against Numpy version."""

        # Generate sinogram with added vertical stripes
        artifact_sinogram = np.ones_like(self.sino)
        for m in range(self.sino.shape[1]):
            sino_slice = self.sino[:, m, :]
            artifact_sino_slice = self.add_vertical_stripes(sino_slice)
            artifact_sinogram[:, m, :] = artifact_sino_slice

        artifact_sinogram_jax = jnp.array(artifact_sinogram)
        # Remove the stripe using both JAX version and NumPy version
        corrected_sino_numpy = stripe_removal.remove_stripe_fw(artifact_sinogram) # Numpy version
        corrected_sino_jax = stripe_removal_jax.remove_stripe_fw_jax(artifact_sinogram_jax) # jax version

        # Compute the statistics
        mean_diff = np.mean(np.abs(corrected_sino_jax - corrected_sino_numpy))

        # Verify that the computed stats are within tolerances
        self.assertTrue(mean_diff < self.remove_stripe_fw_tolerance['difference'],f"Mean difference too high: {mean_diff}")


if __name__ == '__main__':
    unittest.main()

