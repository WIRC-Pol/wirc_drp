import numpy as np
import glob
import warnings
import os
import wirc_drp.wirc_object as wo
from wirc_drp.utils import calibration, spec_utils as su, image_utils as iu

def reduce_dataset(filelist, source_pos, bkg_fnames = None, output_path = "./",verbose=True, 
bkg_methods = ["shift_and_subtract","PCA","median_ref","scaled_bkg","simple_median",
"slit_background","cutout_median"],n_pca=[1,3,5,10,15,20,40], in_slit=False,less_verbose=True,):
    '''
    A function that reduces a dataset given a list of calibrated science and background files

    It will first extract spectra using all of the bkg_methods then calculate the polarization of each one. 


    Inputs:
        filelist    -   A python list of filepaths
        background_list -   A python list of background files (this can be one file)
    '''

    #Set up all the directories and filenames: 
    # fnames = np.sort(glob.glob(filelist))
    if len(filelist) < 1:
        raise ValueError("I couldn't find any files!")
    if verbose or less_verbose:
        print("Found {:d} science files".format(len(filelist)))
    
    if bkg_fnames is not None:
        # bkg_fnames = np.sort(glob.glob(bkg_fnames))
        if len(bkg_fnames) < 1:
            warnings.warn("I couldn't find any background files! But I'm continuing anyway")
        if verbose or less_verbose:
            print("Found {:d} background files".format(len(bkg_fnames)))
    else:
        warnings.warn("You didn't provide any background files. I hope this was on purpose. Continuing.")

    for bkg_method in bkg_methods:
        if bkg_method == "slit_background" and ~in_slit:
            warnings.warn("You requested 'slit_background' subtraction, but didn't indicate that the source was in the slit. Not doing slit_background, but continuing")
            continue

        if verbose or less_verbose:
            print("\nStarting on background subtraction method {} \n".format(bkg_method))
        #PCA needs to iterate over 
        if bkg_method == "PCA":
            for npca in n_pca:
                if verbose or less_verbose:
                    print("\nUsing {} PCA modes".format(npca))
                #Make a new directory for this bkg_method
                outdir = output_path+bkg_method+str(npca)+"/"
                
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                outdir2 = outdir+"bkg_cutout_update/"
                if not os.path.exists(outdir2):
                    os.makedirs(outdir2)
                
                for i,fname in enumerate(filelist):
                    if verbose or less_verbose:
                        print("File {} of {} with bkg_method = {}{}: {}".format(i+1,len(filelist),bkg_method,npca,fname))
                    extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir,verbose=verbose,bkg_method=bkg_method,
                    num_PCA_modes=npca)
                    extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir2,verbose=verbose,bkg_method=bkg_method,
                    num_PCA_modes=npca,update_cutout_backgrounds=True)
        
        else:

            #Make a new directory for this bkg_method
            outdir = output_path+bkg_method + "/"
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            outdir2 = outdir+"bkg_cutout_update/"
            if not os.path.exists(outdir2):
                os.makedirs(outdir2)

            for i,fname in enumerate(filelist):
                if verbose or less_verbose:
                    print("File {} of {} with bkg_method = {}: {}".format(i+1,len(filelist),bkg_method,fname))
                extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir,verbose=verbose,bkg_method=bkg_method)
                if bkg_method != "cutout_median":
                    extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir2,verbose=verbose,bkg_method=bkg_method,
                    update_cutout_backgrounds=True)
            
        

def extract_single_file(filename,source_pos, bkg_fnames,output_path = "./",output_suffix="",verbose=True,
bkg_method=None,num_PCA_modes=None,update_cutout_backgrounds=False):
    '''
    Opens a file, generates a background image, extracts the source spectra and then saves them to the output path

    Inputs:
    filename    -   The filename of a calibrated wirc object file
    source_pos  -   The source position in [x,y] format
    '''
    tmp_data = wo.wirc_data(wirc_object_filename=filename,verbose=verbose)
    tmp_data.source_list = []
    tmp_data.n_sources = 0
    tmp_data.add_source(source_pos[0],source_pos[1],update_w_chi2_shift=True)
    
    if bkg_method is not None and bkg_method != "cutout_median":
        tmp_data.generate_bkg(method=bkg_method,verbose=verbose,
                            bkg_by_quadrants=True,
                            bkg_fns=bkg_fnames,num_PCA_modes=num_PCA_modes)
    wp_source = tmp_data.source_list[0]
    wp_source.get_cutouts(tmp_data.full_image,tmp_data.DQ_image,'J',
                            replace_bad_pixels=True,method='interpolate',
                            bkg_image = tmp_data.bkg_image)
    
    if bkg_method == "cutout_median":
        wp_source.generate_cutout_backgrounds(update=False)
    if update_cutout_backgrounds:
        wp_source.generate_cutout_backgrounds(update=True)

    wp_source.extract_spectra(verbose=verbose,
                            plot_findTrace=False,plot_optimal_extraction=False,
                            spatial_sigma=3,diag_mask=True)

    tmp_data.source_list.append(wp_source)
    tmp_data.n_sources += 1

    # import pdb; pdb.set_trace()
    output_fname = output_path+filename.rsplit(".fits")[0].split("/")[-1]+output_suffix+".fits"
    tmp_data.save_wirc_object(output_fname)

def reduce_ABAB_dataset(filelist, source_pos, output_path = "./",verbose=True):
    '''
    A function that reduces a dataset given a list of calibrated science files, assuming you observed in an ABAB dither pattern.
    It uses each position as backgrond for the other. 

    Inputs:
        filelist    -   A python list of filepaths
        background_list -   A python list of background files (this can be one file)
    '''

    #Set up all the directories and filenames: 
    fnames = np.sort(glob.glob(filelist))
    if len(fnames) < 1:
        raise ValueError("I couldn't find any files!")
    if verbose:
        print("Found {:d} science files".format(len(fnames)))
