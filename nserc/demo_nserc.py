import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import mbirjax
import dxchange

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    # ###################### User defined params. Change the parameters below for your own use case.
    output_path = './output/nserc_demo_sand/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    dataset_path = "./20240425_164409_nist-sand-30-200-mix_27keV_z8mm_n657.h5"    
    
    # #### recon parameters
    sharpness = 0.0
    # ###################### End of parameters

    print("\n*******************************************************",
          "\n***************** Read NSERC dataset ******************",
          "\n*******************************************************")
    numslices = int(dxchange.read_hdf5(dataset_path, "/measurement/instrument/detector/dimension_y")[0])
    numrays = int(dxchange.read_hdf5(dataset_path, "/measurement/instrument/detector/dimension_x")[0])
    pxsize = dxchange.read_hdf5(dataset_path, "/measurement/instrument/detector/pixel_size")[0] / 10.0  # /10 to convert units from mm to cm
    numangles = int(dxchange.read_hdf5(dataset_path, "/process/acquisition/rotation/num_angles")[0])
    propagation_dist = dxchange.read_hdf5(dataset_path, "/measurement/instrument/camera_motor_stack/setup/camera_distance")[1]
    kev = dxchange.read_hdf5(dataset_path, "/measurement/instrument/monochromator/energy")[0] / 1000
    angularrange = dxchange.read_hdf5(dataset_path, "/process/acquisition/rotation/range")[0]

    print(f"{dataset_path}: \
            \n\t * slices: {numslices}, rays: {numrays}, angles: {numangles}, angularrange: {angularrange},\
            \n\t * pxsize: {pxsize*10000:.3f} um, distance: {propagation_dist:.3f} mm. energy: {kev} keV")
    if kev>100:
        print('white light mode detected; energy is set to 30 kev for the phase retrieval function')

