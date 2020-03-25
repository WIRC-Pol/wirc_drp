import sys

try:
    import wirc_drp
    import wirc_drp.wirc_object as wo
    from wirc_drp.utils import calibration
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import numpy as np
    import astropy.io.ascii as asci
    import os

    import matplotlib
    matplotlib.interactive(True)


    #First we'll set up all the directories and filenames: 
    wircpol_dir = os.environ['WIRC_DRP'] # Your WIRCPOL_DRP directory (with a "/" at the end!)
    tutorial_dir = wircpol_dir + "Tutorial/sample_data/Single_File_Tutorial/"

    raw_fn =tutorial_dir+"wirc1586.fits"
    flat_fn = tutorial_dir+"wirc2012_master_flat.fits"
    dark_fn = tutorial_dir+"wirc0141_master_dark.fits"
    bp_fn = tutorial_dir+"wirc2012_bp_map.fits"

    #Now we'll create the wirc_data object, passing in the filenames for the master dark, flat and bad pixel maps
    raw_data = wo.wirc_data(raw_filename=raw_fn, flat_fn = flat_fn, dark_fn = dark_fn, bp_fn = bp_fn,verbose=True,
                            update_bjd=False)

    #calibrate the data object
    raw_data.calibrate(mask_bad_pixels=False,verbose=True)

    #save newly calibrated file
    raw_data.save_wirc_object("calibrated.fits",verbose=True)

    #First we'll create a new data object, mostly just to demonstrate how to read in an existing wirc_data object. 
    calibrated_data = wo.wirc_data(wirc_object_filename="calibrated.fits",verbose=True)

    #generate background image using one of several different options
    calibrated_data.generate_bkg()

    #This specific object will be for a souce in the slit (with coordinates [1063,1027]). 
    wp_source = wo.wircpol_source([1063,1027],'1',0) #The second argument indicates that this source is in the middle slit

    #We'll now get the trace cutouts for this source
    wp_source.get_cutouts(calibrated_data.full_image, calibrated_data.DQ_image, 'J')
    #wp_source.plot_cutouts(figsize=(10,6))

    #Now let's add this source to the calibrated_data source list.
    calibrated_data.source_list.append(wp_source)
    calibrated_data.n_sources += 1

    #We'll now manually add another source in the field, that isn't in the slit
    calibrated_data.source_list.append(wo.wircpol_source([1035,640],'slitless',calibrated_data.n_sources+1))
    calibrated_data.n_sources += 1
    calibrated_data.source_list[1].get_cutouts(calibrated_data.full_image, calibrated_data.DQ_image,'J',
                                            calibrated_data.bkg_image)

    #plot cutouts
    #calibrated_data.source_list[1].plot_cutouts(figsize=(10,6),plot_bkg_sub=True,vmin=-200,vmax=400)

    #extract the spectra from each thumbnail
    calibrated_data.source_list[0].extract_spectra(verbose=False)
    calibrated_data.source_list[0].plot_trace_spectra(figsize=(12,8))

    ## Now we'll do a rough wavelength calibration and plot the data again
    calibrated_data.source_list[0].rough_lambda_calibration(method=2) 
    calibrated_data.source_list[0].plot_trace_spectra(figsize=(10,6),xlow=1.16,xhigh=1.34)

    #We now combine the 4 spectra to calculate Stokes Q and U
    calibrated_data.source_list[0].compute_polarization(cutmin=20, cutmax=150)
    calibrated_data.source_list[0].plot_Q_and_U(figsize=(8,3),xlow=1.17,xhigh=1.32,ylow=-0.1,yhigh=0.1)

    #We now save the new information and tables to a new file Note how it initiates the table columns even if the data hasn't been computed (for example, we haven't computed source_list[1] yet)
    calibrated_data.save_wirc_object("calibrated.fits")
    print('End of test script 1.')

    plt.close("all")
except Exception as e:
    print("Failed Test 1 with exception {}".format(e))
    print(e)
    sys.exit(1)