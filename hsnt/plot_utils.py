import numpy as np
import matplotlib.pyplot as plt


def plot_hyper_recons(recons, display_wave_idx, display_slices, title='Reconstructed hyperspectral images',
                      vmin=None, vmax=None, cb_orientation='vertical'):
    """Function to display reconstructed hyperspectral images corresponding to different wavelengths at different slices.

    Args:
        recons(ndarray): reconstructed images (height x width x depth x wavelengths)
        display_wave_idx(list): wavelength indices to be displayed
        display_slices(list): reconstructed slices to be displayed
        title(str,optional): title of the set of images
        vmin(float,optional): value mapped to black
        vmax(float,optional): value mapped to white
        cb_orientation(str,optional): orientation of the color bar (must be 'horizontal' or 'vertical')
        """
    num_disp_wave = len(display_wave_idx)
    num_disp_slices = len(display_slices)
    display_recons = recons[:, :, display_slices, :]
    display_recons = display_recons[:, :, :, display_wave_idx]

    if vmin is None:
        vmin = np.percentile(display_recons, 1)

    if vmax is None:
        vmax = np.percentile(display_recons, 99)

    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc('font', size=20)
    fig = plt.figure(figsize=(8 * num_disp_slices, 8 * num_disp_wave))
    fig.suptitle(title, size=30)
    fig.supylabel('Wavelength indices', size=30)
    fig.supxlabel('Display slices', size=30)

    for wave in range(num_disp_wave):
        for disp_slice in range(num_disp_slices):
            fig_temp = fig.add_subplot(num_disp_wave, num_disp_slices, wave * num_disp_slices + disp_slice + 1)
            img_temp = fig_temp.imshow(display_recons[:, :, disp_slice, wave], vmin=vmin, vmax=vmax, cmap='gray')
            fig.colorbar(img_temp, ax=fig_temp, orientation=cb_orientation)

    plt.show()
