import jax
import jax.numpy as jnp
import pywt
from scipy import interpolate
from scipy.ndimage import uniform_filter1d, median_filter, binary_dilation
from concurrent.futures import ThreadPoolExecutor


def create_mat_index(nrow, ncol):
    """
    Create a 2D array of indexes used for the sorting technique.

    This function is based on the method described in:
    [Nghia T. Vo et al., 2018] - "Superior techniques for eliminating ring artifacts in x-ray micro-tomography"

    It is also the code from the Tomopy library:
    https://github.com/tomopy/tomopy.git

    References:
    [1] Vo N, and Atwood RC, and Drakopoulos M. Superior techniques for eliminating ring artifacts in x-ray micro-tomography. Optics Express, 26(22):28396–28412, 2018.
    [2] Tomopy library https://github.com/tomopy/tomopy.git

    Args:
        nrow (int): number of detector rows in the sinogram
        ncol (int): number of detector channels in the sinogram

    Returns:
        a 2D jax array of indexes
    """
    list_index = jnp.arange(0.0, ncol, 1.0)
    mat_index = jnp.tile(list_index, (nrow, 1))
    return mat_index


def rs_sort(sino, size, mat_index):
    """
    Remove small-to-medium partial and fulll stripes using the sorting technique.

    This function is based on the method described in:
    [Nghia T. Vo et al., 2018] - "Superior techniques for eliminating ring artifacts in x-ray micro-tomography"

    It also adapts the code from the Tomopy library:
    https://github.com/tomopy/tomopy.git

    References:
    [1] Vo N, and Atwood RC, and Drakopoulos M. Superior techniques for eliminating ring artifacts in x-ray micro-tomography. Optics Express, 26(22):28396–28412, 2018.
    [2] Tomopy library https://github.com/tomopy/tomopy.git

    Args:
        sino (jax array): a 2D slice of the sinogram data with shape (num_views, num_det_channels)
        size (int): window size of the median filter
        mat_index (jax array): a 2D array of indexes used for the sorting technique
        dim (optional, {1, 2}): Dimension of the window

    Return:
        sino (jax array): corrected 2D slice of the sinogram data after stripes removal
    """

    # Sort each column of the sinogram by its grayscale values
    sino = jnp.transpose(sino)
    mat_stack = jnp.stack([mat_index, sino], axis=2)
    sort_indices = jnp.argsort(mat_stack[:, :, 1], axis=1)
    sort_indices_expanded = sort_indices[:, :, None]
    mat_sort = jnp.take_along_axis(mat_stack, sort_indices_expanded, axis=1)

    # Apply the median filter on teh sorted sinogram along each row
    mat_sort = mat_sort.at[:, :, 1].set(median_filter(mat_sort[:, :, 1], (size, 1)))

    # Re-sort the smoothed image columns to the original rows to get the corrected sinogram
    sort_indices = jnp.argsort(mat_sort[:, :, 0], axis=1)
    sort_indices_expanded = sort_indices[:, :, None]
    mat_sort_back = jnp.take_along_axis(mat_sort, sort_indices_expanded, axis=1)
    sino_corrected = mat_sort_back[:, :, 1]
    sino_corrected = jnp.transpose(sino_corrected)

    return sino_corrected

def detect_stripe(listdata, snr):
    """
    Used to locate stripes.
    A segmentation algorithm to separate the extremely positive and negative defects from the normal values in the
    sinogram.

    Args:
        listdata (jax array): a normalized 1D array
        snr (float): a ratio between the defective value and the background value. a reasonable choice of snr should be around 3.0 or above

    Returns:
        listmask (jax array): a 2D binary array denoting the stripes detected
    """

    numdata = listdata.shape[0]

    # Sort the 1D array
    listsorted = jnp.sort(listdata)[::-1]
    xlist = jnp.arange(0, numdata, 1.0)

    # Apply a linear fit to values around the middle of the sorted array
    # Calculating the noise level to avoid false positives caused by minor background variations
    ndrop = jnp.int16(0.25 * numdata)
    (_slope, _intercept) = jnp.polyfit(xlist[ndrop:-ndrop - 1], listsorted[ndrop:-ndrop - 1], 1)
    numt1 = _intercept + _slope * xlist[-1]
    noiselevel = jnp.abs(numt1 - _intercept)
    noiselevel = jnp.clip(noiselevel, 1e-6, None)
    val1 = jnp.abs(listsorted[0] - _intercept) / noiselevel
    val2 = jnp.abs(listsorted[-1] - numt1) / noiselevel

    # Calculate the upper threshold and the lower threshold
    # Binarize the array by replacing all values between lower and upper threshold with 0 and others with 1.
    listmask = jnp.zeros_like(listdata)
    if (val1 >= snr):
        upper_thresh = _intercept + noiselevel * snr * 0.5
        listmask = jnp.where(listdata > upper_thresh, 1.0, listmask)
    if (val2 >= snr):
        lower_thresh = numt1 - noiselevel * snr * 0.5
        listmask = jnp.where(listdata <= lower_thresh, 1.0, listmask)

    return listmask


def rs_large(sino, snr, size, mat_index, drop_ratio=0.1):
    """
    Remove large partial and full stripes using the sorting technique.

    This function is based on the method described in:
    [Nghia T. Vo et al., 2018] - "Superior techniques for eliminating ring artifacts in x-ray micro-tomography"

    It also adapts the code from the Tomopy library:
    https://github.com/tomopy/tomopy.git

    References:
    [1] Vo N, and Atwood RC, and Drakopoulos M. Superior techniques for eliminating ring artifacts in x-ray micro-tomography. Optics Express, 26(22):28396–28412, 2018.
    [2] Tomopy library https://github.com/tomopy/tomopy.git

    Args:
        sino (jax array): a 2D slice of the sinogram data with shape (num_views, num_det_channels)
        snr (float): a ratio between the defective value and the background value
        size (int): window size of the median filter
        mat_index (jnp array): a 2D array of indexes used for the sorting technique
        drop_ratio (float, optional): ratio of pixels at the top and the bottom of the sinogram to be removed. Defaults to 0.1.

    Returns:
        sino (jax array): corrected 2D slice of the sinogram data after stripes removal
    """

    drop_ratio = jnp.clip(drop_ratio, 0.0, 0.8)
    (nrow, ncol) = sino.shape
    ndrop = jnp.int16(0.5 * drop_ratio * nrow)

    # Sorting the columns of the sinogram and apply the median filter on the sorted image along each row
    sino_sort = jnp.sort(sino, axis=0)
    sino_smooth = jnp.array(median_filter(sino_sort, (1, size)))

    # Compute the column-wise average of the sorted and smoothed sinogram
    # Compute the normalized 1D array
    list_raw = jnp.mean(sino_sort[ndrop:nrow - ndrop], axis=0)
    list_smooth = jnp.mean(sino_smooth[ndrop:nrow - ndrop], axis=0)
    list_normal = jnp.where(
        list_smooth != 0,
        jnp.divide(list_raw, list_smooth),
        jnp.ones_like(list_raw)
    )

    # Locate the large stripes
    list_mask = detect_stripe(list_normal, snr)
    list_mask = jnp.array(binary_dilation(list_mask, iterations=1).astype(list_mask.dtype))
    mat_normal_factor = jnp.tile(list_normal, (nrow, 1))

    # Apply pre-correction to the original sinogram
    sino = sino / mat_normal_factor

    # Apply the sorting-based algorithm again to get the corrected columns
    sino_T = jnp.transpose(sino)
    mat_stack = jnp.stack([mat_index, sino_T], axis=2)

    sort_indices = jnp.argsort(mat_stack[:, :, 1], axis=1)
    sort_indices_expanded = sort_indices[:, :, None]
    mat_sort = jnp.take_along_axis(mat_stack, sort_indices_expanded, axis=1)

    mat_sort = mat_sort.at[:, :, 1].set(jnp.transpose(sino_smooth))

    sort_indices = jnp.argsort(mat_sort[:, :, 0], axis=1)
    sort_indices_expanded = sort_indices[:, :, None]
    mat_sort_back = jnp.take_along_axis(mat_sort, sort_indices_expanded, axis=1)

    sino_corrected = jnp.transpose(mat_sort_back[:, :, 1])

    list_x_miss = jnp.where(list_mask > 0.0)[0]

    # Selective Replacement of Defective Columns with corrected columns
    sino = sino.at[:, list_x_miss].set(sino_corrected[:, list_x_miss])
    return sino


def rs_dead(sino, snr, size, mat_index):
    """
    Remove unresponsive and fluctuating stripes using the interpolation technique.
    Sorting approach does not work here because the rankings of the grayscales are significantly different between
    pixels inside the stripes and outside the stripes. Instead, interpolation is an appropriate choice.

    This function is based on the method described in:
    [Nghia T. Vo et al., 2018] - "Superior techniques for eliminating ring artifacts in x-ray micro-tomography"

    It also adapts the code from the Tomopy library:
    https://github.com/tomopy/tomopy.git

    References:
    [1] Vo N, and Atwood RC, and Drakopoulos M. Superior techniques for eliminating ring artifacts in x-ray micro-tomography. Optics Express, 26(22):28396–28412, 2018.
    [2] Tomopy library https://github.com/tomopy/tomopy.git

    Args:
        sino (jax array): a 2D slice of the sinogram data with shape (num_views, num_det_channels)
        snr (float): a ratio between the defective value and the background value
        size (int): window size of the median filter
        mat_index (jax array): a 2D array of indexes used for the sorting technique
        norm (boolean): Remove residual stripes if True. Default is True.

    Returns:
        sino (jax array): corrected 2D slice of the sinogram data after stripes removal

    """
    nrow = sino.shape[0]

    # Compute the column-wise absolute difference between the original sinogram and the smoothed sinogram
    # Help detect the unresponsive and fluctuating stripes (large diff -> fluctuating stripes; small diff -> unresponsive stripes)
    sino_smooth = jnp.array(uniform_filter1d(sino, 10, axis=0))
    list_diff = jnp.sum(jnp.abs(sino - sino_smooth), axis=0)

    # Compute normalized 1D array where the large value correspond to defective pixels
    list_diff_filtered = jnp.array(median_filter(list_diff, size=size))
    list_normal = jnp.where(
        list_diff_filtered != 0,
        jnp.divide(list_diff, list_diff_filtered),
        jnp.ones_like(list_diff)
    )

    # Generate binary mask
    list_mask = detect_stripe(list_normal, snr)
    list_mask = jnp.array(binary_dilation(list_mask, iterations=1).astype(list_mask.dtype))
    list_mask = list_mask.at[0:2].set(0.0)
    list_mask = list_mask.at[-2:].set(0.0)

    # Interpolation
    list_x = jnp.where(list_mask < 1.0)[0]
    list_y = jnp.arange(nrow)
    mat_z = sino[:, list_x]
    fit_function = interpolate.RectBivariateSpline(list_y, list_x, mat_z, kx=1, ky=1)

    # Apply interpolation to defective columns
    list_x_miss = jnp.where(list_mask > 0.0)[0]
    if len(list_x_miss) > 0:
        mat_x_miss, mat_y = jnp.meshgrid(list_x_miss, list_y)
        estimate_output = fit_function.ev(jnp.ravel(mat_y), jnp.ravel(mat_x_miss))
        sino = sino.at[:, list_x_miss].set(jnp.reshape(estimate_output, mat_x_miss.shape))

    # Remove residual large stripes
    sino = rs_large(sino, snr, size, mat_index)

    return sino

def remove_all_stripe_jax(sino, snr=3, la_size=61, sm_size=21):
    """
    To remove all types of stripes from the sinogram, including partial, full, fluctuating, and unresponsive stripes, this approach builds on the tomopy.remove_all_stripes method. It combines three algorithms:
    1.    A sorting-based algorithm for small to medium partial and full stripes.
    2.    A sorting-based algorithm for large partial and full stripes.
    3.    An interpolation-based algorithm for unresponsive and fluctuating stripes.
    These algorithms are applied in the order: (3, 2, 1).

    This function is based on the method described in:
    [Nghia T. Vo et al., 2018] - "Superior techniques for eliminating ring artifacts in x-ray micro-tomography"

    It also adapts the code from the Tomopy library:
    https://github.com/tomopy/tomopy.git

    References:
    [1] Vo N, and Atwood RC, and Drakopoulos M. Superior techniques for eliminating ring artifacts in x-ray micro-tomography. Optics Express, 26(22):28396–28412, 2018.
    [2] Tomopy library https://github.com/tomopy/tomopy.git

    Args:
        sino (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels)
        snr (float, optional): a ratio between the defective value and the background value. a reasonable choice of snr should be around 3.0 or above
        la_size (int, optional): Window size of the median filter to remove large stripes
        sm_size (int, optional): Window size of the median filter to remove small-to-medium stripes

    Returns:
        sino (jax array): corrected sinogram data after removing all stripes
    """
    mat_index = create_mat_index(sino.shape[2], sino.shape[0])
    mat_index_cpu = jax.device_put(mat_index, device=jax.devices("cpu")[0])

    sino_cpu = jax.device_put(sino, device=jax.devices("cpu")[0])

    result = jnp.zeros_like(sino)

    def process_slice(m):
        sino_slice = sino_cpu[:, m, :]
        sino_slice = rs_dead(sino_slice, snr, la_size, mat_index_cpu)
        sino_slice = rs_sort(sino_slice, sm_size, mat_index_cpu)
        return sino_slice

    with ThreadPoolExecutor() as executor:
        processed_slices = list(executor.map(process_slice, range(sino.shape[1])))

    for m, processed_slice in enumerate(processed_slices):
        result = result.at[:, m, :].set(processed_slice)

    return jax.device_put(result)


def remove_stripe_fw_jax(sino, wname="db5", sigma=2):
    """
    Remove vertical stripes from the sinogram by combining 2D Discret Wavelet Transform and 2D Fourier Transform.
    This approach builds based on tomopy.remove_stripe_fw().

    This function is based on the method described in:
    [Beat Munch et al. 2009] - "Stripe and ring artifact removal with combined wavelet — Fourier filtering"

    It also adapts the code from the Tomopy library:
    https://github.com/tomopy/tomopy.git

    Args:
        sino (jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels)
        wname (str, optional): wavelet filter type.
        sigma (float, optional): Damping parameter in Fourier space

    Returns:
        sino (jax array): corrected sinogram data after removing all stripes

    """
    # Determine decomposition level L
    level = int(jnp.ceil(jnp.log2(jnp.max(jnp.array(sino.shape)))))
    views, num_rows, num_columns = sino.shape
    padded_views = views + views // 8
    shift_val = views // 16

    for m in range(sino.shape[1]):
        sino_slice = jnp.zeros((padded_views, num_columns), dtype=jnp.float32)
        sino_slice = sino_slice.at[shift_val:views + shift_val].set(sino[:, m, :])

        # 2D Discrete Wavelte Transform
        cH, cV, cD = {}, {}, {}
        for n in range(level):
            sino_slice, (cH[n], cV[n], cD[n]) = pywt.dwt2(sino_slice, wname)

        # FFT transform of horizontal frequency bands
        for n in range(level):
            # FFT
            fcV = jnp.fft.fftshift(jnp.fft.fft(cV[n], axis=0))
            my, mx = fcV.shape

            # Damping of vertical stripe information
            y_hat = (jnp.arange(-my, my, 2, dtype='float32') + 1) / 2
            damp = -jnp.expm1(-jnp.square(y_hat) / (2 * jnp.square(sigma)))
            fcV *= jnp.transpose(jnp.tile(damp, (mx, 1)))

            # Inverse FFT
            cV[n] = jnp.real(jnp.fft.ifft(jnp.fft.ifftshift(fcV), axis=0))

        # 2D inverse discrete wavelet transform
        for n in range(level)[::-1]:
            sino_slice = sino_slice[0:cH[n].shape[0], 0:cH[n].shape[1]]
            sino_slice = pywt.idwt2((sino_slice, (cH[n], cV[n], cD[n])), wname)

        sino = sino.at[:, m, :].set(sino_slice[shift_val:views + shift_val, 0:num_columns])

    return sino