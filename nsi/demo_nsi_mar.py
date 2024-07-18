import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import scipy
import mbirjax
import mbirjax.plot_utils as pu
import demo_utils
import pprint
pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is a demonstration of the metal artifact reduction (MAR) functionality using MAR sinogram weight.\
    \n Demo functionality includes:\
    \n\t * downloading NSI dataset from specified urls;\
    \n\t * Computing sinogram data;\
    \n\t * Computing two sets of sinogram weights, one with type "transmission_root" and the other with type "MAR";\
    \n\t * Computing two sets of MBIR reconstructions with each sinogram weight respectively;\
    \n\t * Displaying the results.\n')
    # ###################### User defined params. Change the parameters below for your own use case.
    output_path = './output/nsi_demo_mar/' # path to store output recon images
    os.makedirs(output_path, exist_ok=True) # mkdir if directory does not exist

    # ##### params for dataset downloading. User may change these parameters for their own datasets.
    # An example NSI dataset (tarball) will be downloaded from `dataset_url`, and saved to `download_dir`.
    # url to NSI dataset.
    dataset_url = 'https://engineering.purdue.edu/~bouman/data_repository/data/mar_demo_data.tgz'
    # destination path to download and extract the NSI data and metadata.
    download_dir = './demo_data/'
    # Path to NSI scan directory.
    _, dataset_dir = demo_utils.download_and_extract_tar(dataset_url, download_dir)
    # for testing user prompt in NSI preprocessing function
    # dataset_dir = "/depot/bouman/data/share_conebeam_data/Autoinjection-Full-LowRes/Vertical-0.5mmTin"
  
    # #### preprocessing parameters
    downsample_factor = [4, 4] # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 1 # view subsample factor.
    
    # #### recon parameters
    sharpness=0.0
    # ###################### End of parameters

    print("\n*******************************************************",
          "\n************** NSI dataset preprocessing **************",
          "\n*******************************************************")
    sino, cone_beam_params, optional_params = \
        mbirjax.preprocess.NSI.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor)

    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    # ConeBeamModel constructor
    ct_model = mbirjax.ConeBeamModel(**cone_beam_params)
    
    # Set additional geometry arguments
    ct_model.set_params(**optional_params)

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1, positivity_flag=True)
    
    # Print out model parameters
    ct_model.print_params()
     
    print("\n*******************************************************",
          "\n***** Calculate transmission_root sinogram weights ****",
          "\n*******************************************************")
    weights_trans = ct_model.gen_weights(sino, weight_type='transmission_root')
    
    print("\n*******************************************************",
          "\n**** Perform recon with transmission_root weights. ****",
          "\n*******************************************************")
    print("This recon will be used to identify metal voxels and compute the MAR sinogram weight.")
    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()

    init_recon, recon_params = ct_model.recon(sino, weights=weights_trans)

    init_recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for initial recon is {:.3f} seconds'.format(elapsed))
    # ##########################
    
    print("\n*******************************************************",
          "\n************ Calculate MAR sinogram weights ***********",
          "\n*******************************************************")
    weights_mar = ct_model.gen_weights_mar(sino, init_recon=init_recon,
                                           beta=1.0, gamma=3.0)
    
    print("\n*******************************************************",
          "\n*********** Perform recon with MAR weights. ***********",
          "\n*******************************************************")
    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()

    recon_mar, recon_params = ct_model.recon(sino, weights=weights_mar, init_recon=init_recon, num_iterations=10)

    recon_mar.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon with MAR weight is {:.3f} seconds'.format(elapsed))
    # ##########################
 
    # change the image data shape to (slices, rows, cols), so that the rotation axis points up when viewing the coronal/sagittal slices with slice_viewer
    init_recon = np.transpose(init_recon, (2,1,0))
    init_recon = init_recon[:,:,::-1]
    
    recon_mar = np.transpose(recon_mar, (2,1,0))
    recon_mar = recon_mar[:,:,::-1]
  
    # rotate the recon images to an upright pose for display purpose
    rot_angle = 17.165 # rotate angle in the plane defined by axes [0,2].
    init_recon = scipy.ndimage.rotate(init_recon, rot_angle, [0,2], reshape=False, order=3)
    recon_mar = scipy.ndimage.rotate(recon_mar, rot_angle, [0,2], reshape=False, order=3) 
    
    # Display results
    vmin = 0
    vmax = downsample_factor[0]*0.008

    pu.slice_viewer(init_recon, recon_mar, vmin=0, vmax=vmax, slice_axis=0, slice_label='Axial Slice', title='recon with transmission_root weight (left) VS recon with MAR weight (right)')
    pu.slice_viewer(init_recon, recon_mar, vmin=0, vmax=vmax, slice_axis=1, slice_label='Coronal Slice', title='recon with transmission_root weight (left) VS recon with MAR weight (right)')
    pu.slice_viewer(init_recon, recon_mar, vmin=0, vmax=vmax, slice_axis=2, slice_label='Sagittal Slice', title='recon with transmission_root weight (left) VS recon with MAR weight (right)')
