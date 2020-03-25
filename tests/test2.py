import sys

try:
    import glob
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import wirc_drp.wirc_object as wo
    import wirc_drp.constants as constants
    from wirc_drp.utils import calibration, spec_utils as su, image_utils as iu
    from wirc_drp.masks import *

    #First we'll set up all the directories and filenames: 
    wircpol_dir = os.environ['WIRC_DRP'] # Your WIRCPOL_DRP directory (with a "/" at the end!)
    tutorial_dir = wircpol_dir + "Tutorial/sample_data/"

    #background sky frames
    sky_frames = [tutorial_dir+'wirc0141.fits',
                    tutorial_dir+'wirc0142.fits',
                    tutorial_dir+'wirc0158.fits',
                    tutorial_dir+'wirc0159.fits',
                    tutorial_dir+'wirc0160.fits']

    image = wo.wirc_data(tutorial_dir+'wirc0140.fits')

    #source finding
    print('testing source finding')
    image.find_sources_v2(sigma_threshold=0, show_plots=False)

    #background subtraction
    print('testing background subtraction methods')
    image.generate_bkg(method='shift_and_subtract', bkg_fns=sky_frames, shift_dir='diagonal', bkg_sub_shift_size=31,
                                    filter_bkg_size=None, same_HWP=False)

    image = wo.wirc_data(tutorial_dir+'wirc0140.fits')                           
    image.generate_bkg(method='PCA', num_PCA_modes=3, ref_lib=sky_frames, same_HWP=False)

    image = wo.wirc_data(tutorial_dir+'wirc0140.fits')
    image.generate_bkg(method='median_ref', ref_lib=sky_frames, same_HWP=False)

    image = wo.wirc_data(tutorial_dir+'wirc0140.fits')
    image.generate_bkg(method='scaled_bkg', bkg_fns=sky_frames, bkg_by_quadrants=False, same_HWP=False)

    image = wo.wirc_data(tutorial_dir+'wirc0140.fits')
    image.generate_bkg(method='simple_median', ref_lib=sky_frames, same_HWP=False)

    #source masking
    print('testing source masking')
    image = wo.wirc_data(tutorial_dir+'wirc0140.fits')
    image.mask_sources(show_plot=False)

    print('End of test script 2.')

except Exception as e:
    print("Failed Test 2 with exception {}".format(e))
    print(e)
    sys.exit(1)