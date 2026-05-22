There are two scripts in this folder:

1. hsnt_prep_utils.py
----------------------

This script is a library of functions for hyperspectral neutron data preprocessing.
It contains the following functions:

load_data:
    Loads open-beam or projection TIFF images from a given directory.

replace_zero:
    Replaces zeros in the images with the median value of the 8 closest
    neighbors. If any zeros remain, they are replaced with epsilon = 1e-8.

smooth_open_beam:
    Smooths the open-beam counts. This is useful because open beams do not
    contain sharp details, so smoothing reduces noise without losing important
    information.

compute_transmission:
    Computes the ratio of projection counts to open-beam counts.

calibrate_background_ORNL_SNAP:
    Corrects the gain difference between open beams and projections for
    attenuation data. These corrections are specific to ORNL SNAP beamline
    datasets.

hyper_data_preprocessing:
    Performs all the above steps to convert raw neutron count data into
    attenuation or transmission data.


2. demo_Ni_single_view.py
--------------------------

This script preprocesses the Ni single-view datasets.

There are 10 datasets in total, corresponding to 5 different proton charges
and 2 different Ni samples, and the script works for all of them.

The script calls the hyper_data_preprocessing function from
hsnt_prep_utils.py to process the data and then saves the processed results
as HDF5 files.


Note
-----

All necessary libraries are already dependencies of MBIRJAX.
Therefore, simply activate your MBIRJAX environment and run:

    python demo_Ni_single_view.py