import numpy as np
import glob
import warnings
import os
import wirc_drp.wirc_object as wo
from wirc_drp.utils import calibration, spec_utils as su, image_utils as iu
from wirc_drp.utils import source_utils as so
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from astropy import stats
from astropy.timeseries import LombScargle
import copy
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import warnings
warnings.filterwarnings('ignore')


def reduce_dataset(filelist, source_pos, bkg_fnames = None, output_path = "./",verbose=True, 
bkg_methods = ["shift_and_subtract","PCA","median_ref","scaled_bkg","simple_median",
"slit_background","cutout_median"],n_pca=[1,3,5,10,15,20,40], in_slit=False,less_verbose=True,
parallel=False,n_processes=None,nclosest=None,same_HWP=False,sub_bar=False,method='optimal_extraction',
fixed_width = None):
    """A function that reduces a dataset given a list of calibrated science and background files

    It will first extract spectra using all of the bkg_methods then calculate the polarization of each one (though the polarization calculation is not yet implemented)

    Args:
        filelist ([str]): A list of input filepaths
        source_pos ([float]): The source positions [x,y]
        bkg_fnames ([str], optional): An optional list of background files. Most of the most successful background subtraction methods need this. Defaults to None.
        output_path (str, optional): Where do you want to put all the extracted wirc_object files? I recommend elsewhere. Defaults to "./".
        verbose (bool, optional): How many details do you want?. Defaults to True.
        bkg_methods (list, optional): A list of all the background subtraction methods that you want to use. Defaults to ["shift_and_subtract","PCA","median_ref","scaled_bkg","simple_median", "slit_background","cutout_median"].
        n_pca (list, optional): A list of PCA modes you want to use for PCA bkg subtraction. Defaults to [1,3,5,10,15,20,40].
        in_slit (bool, optional): Is the source in the slit? If not, then some of the bkg_methods won't be used. Defaults to False.
        less_verbose (bool, optional): Somewhere between 'verbose' and silence. Defaults to True.
        parallel (bool, optional): Do you want to run things in parallel? Probably, but sometimes it breaks. Defaults to False.
        n_processes ([type], optional): If you're parallel how many processes do you want to run? Defaults to None.
        nclosest ([type], optional): When building a library of background files to use (from bkg_fnames) for each file you can choose to only use the nclosests files in time. Defaults to None (i.e. use all).
        same_HWP (bool, optional): When choosing bkg files, do you want to use those only with the same HWP angle? Defaults to False.
        sub_bar (bool, optional): Do you want to subtract the back of doom? If you have no background files and you're relying on cutout_median, then maybe. Defaults to False.
        method (str, optional): The method you want to use to extract the spectrum. Defaults to 'optimal_extraction'. Another option is 'sum_across_trace'
        fixed_width ([type], optional): You can manually set the width of the extraction region rather than finding it automatically. This can sometimes be more stable. You can pass either a single value or a list of 4 - one for each trace. Defaults to None.

    Raises:
        ValueError: Raised in your filelist is empty
    """

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
        # try: 
        if bkg_method == "slit_background" and not in_slit:
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

                if not in_slit:
                    outdir2 = outdir+"bkg_cutout_update/"
                    if not os.path.exists(outdir2):
                        os.makedirs(outdir2)


                #Let's reduce the files in parallel!
                if parallel: 
                    import multiprocessing as mp

                    #If the user doesn't provide the number of processes,
                    #Then pick the maximum minus 1
                    if n_processes is None:
                        n_processes = mp.cpu_count() - 1

                    #Make the pool
                    with mp.Pool(processes=n_processes) as pool:
                    # pool = mp.Pool(processes=n_processes)
                        
                        #Package up the arguments: 
                        args = [(fname,source_pos,bkg_fnames,
                        outdir,"",verbose,bkg_method,npca,False,False,nclosest,same_HWP,sub_bar,
                        method,fixed_width) for fname in filelist]
                        outputs = pool.map(extract_single_file_parallel_helper,args) 

                        if not in_slit:
                        #Now with the update cutout backgrounds
                            args = [(fname,source_pos,bkg_fnames,
                            outdir2,"",verbose,bkg_method,npca,True,False,nclosest,same_HWP,sub_bar,
                            method,fixed_width) for fname in filelist]
                            outputs = pool.map(extract_single_file_parallel_helper,args) 
                #Or not parallel
                else: 
                
                    for i,fname in enumerate(filelist):
                        if verbose or less_verbose:
                            print("File {} of {} with bkg_method = {}{}: {}".format(i+1,len(filelist),bkg_method,npca,fname))
                        
                        extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir,verbose=verbose,bkg_method=bkg_method,
                        num_PCA_modes=npca,sub_bar=sub_bar,nclosest=nclosest,same_HWP=same_HWP,method=method,fixed_width=fixed_width)

                        if not in_slit:
                            extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir2,verbose=verbose,bkg_method=bkg_method,
                            num_PCA_modes=npca,update_cutout_backgrounds=True,sub_bar=sub_bar,nclosest=nclosest,same_HWP=same_HWP,
                            method=method,fixed_width=fixed_width)
        
        else: #not PCA

            #Make a new directory for this bkg_method
            outdir = output_path+bkg_method + "/"
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            outdir2 = outdir+"bkg_cutout_update/"
            if not os.path.exists(outdir2):
                os.makedirs(outdir2)

            if parallel:

                import multiprocessing as mp

                #If the user doesn't provide the number of processes,
                #Then pick the maximum minus 1
                if n_processes is None:
                    n_processes = mp.cpu_count() - 1

                #Make the pool
                with mp.Pool(processes=n_processes) as pool:
                # pool = mp.Pool(processes=n_processes)
                
                    #Package up the arguments: 
                    args = [(fname,source_pos,bkg_fnames,
                    outdir,"",verbose,bkg_method,None,False,False,nclosest,same_HWP,sub_bar,
                    method,fixed_width) for fname in filelist]
                    outputs = pool.map(extract_single_file_parallel_helper,args) 

                    if bkg_method != "cutout_median" and not in_slit:
                        #This is to update cutout backgrounds, don't have to do if method is already "cutout_median"
                        #Now with the update cutout backgrounds
                        args = [(fname,source_pos,bkg_fnames,
                        outdir2,"",verbose,bkg_method,None,True,False,nclosest,same_HWP,sub_bar,
                        method,fixed_width) for fname in filelist]
                        outputs = pool.map(extract_single_file_parallel_helper,args) 

            else: #not parallel

                for i,fname in enumerate(filelist):
                    if verbose or less_verbose:
                        print("File {} of {} with bkg_method = {}: {}".format(i+1,len(filelist),bkg_method,fname))
                    extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir,verbose=verbose,
                    bkg_method=bkg_method,nclosest=nclosest,sub_bar=sub_bar,same_HWP=same_HWP,
                    method=method,fixed_width=fixed_width)
                    if bkg_method != "cutout_median" and not in_slit:
                        #This is to update cutout backgrounds, don't have to do if method is already "cutout_median"
                        extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir2,verbose=verbose,bkg_method=bkg_method,
                        update_cutout_backgrounds=True,nclosest=nclosest,sub_bar=sub_bar,same_HWP=same_HWP,
                        method=method,fixed_width=fixed_width)
        # except:
            # print("Couldn't finish reduction on file {} for bkg_method {}".format(fname,bkg_method))
            # pass

        # plot_dataset_extraction_summary(outdir,save=True,)
            
def extract_single_file(filename,source_pos, bkg_fnames,output_path = "./",output_suffix="",verbose=True,
bkg_method=None,num_PCA_modes=None,update_cutout_backgrounds=False,save_full_image = False,sub_bar = False,
nclosest=None,same_HWP=False,method='optimal_extraction',fixed_width=None):
    """Opens a file, generates a background image, extracts the source spectra and then saves them to the output path

    Args:
        filename ([str]): The filename of a calibrated wirc object file
        source_pos ([float]): The source positions [x,y]
        bkg_fnames ([str], optional): An optional list of background files. Most of the most successful background subtraction methods need this. Defaults to None.
        output_path (str, optional): Where do you want to put all the extracted wirc_object files? I recommend elsewhere. Defaults to "./".
        output_suffix (str, optional): If you'd like you could append a new suffix to all the output filenames. Defaults to "".
        verbose (bool, optional): How many details do you want?. Defaults to True.
        bkg_method (str, optional): Which background method are we using?. Defaults to None.
        num_PCA_modes (list of ints, optional): A list of PCA modes you want to use for PCA bkg subtraction. Defaults to None.
        update_cutout_backgrounds (bool, optional): After you perform background subtraction, do you want to top it up with a median subtraction of the cutout? Defaults to False.
        save_full_image (bool, optional): When you save the wirc_object, do you want to save the full image and save some space? Defaults to False.
        sub_bar (bool, optional): Do you want to subtract the back of doom? If you have no background files and you're relying on cutout_median, then maybe. Defaults to False.
        nclosest ([type], optional): When building a library of background files to use (from bkg_fnames) for each file you can choose to only use the nclosests files in time. Defaults to None (i.e. use all).
        same_HWP (bool, optional): When choosing bkg files, do you want to use those only with the same HWP angle? Defaults to False.
        method (str, optional): The method you want to use to extract the spectrum. Defaults to 'optimal_extraction'. Another option is 'sum_across_trace'
        fixed_width ([type], optional): You can manually set the width of the extraction region rather than finding it automatically. This can sometimes be more stable. You can pass either a single value or a list of 4 - one for each trace. Defaults to None.
    """
    try:
        tmp_data = wo.wirc_data(wirc_object_filename=filename,verbose=verbose)
        tmp_data.source_list = []
        tmp_data.n_sources = 0

        if bkg_method is not None and bkg_method != "cutout_median":
            tmp_data.generate_bkg(method=bkg_method,verbose=verbose,
                                bkg_by_quadrants=True,
                                bkg_fns=bkg_fnames,num_PCA_modes=num_PCA_modes,
                                nclosest=nclosest,
                                same_HWP=same_HWP,
                                destripe=True)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(tmp_data.full_image)
        # plt.show()
        band = tmp_data.header['AFT']
        if 'J' in band:
            band = 'J'
        else:
            band = 'H'

        tmp_data.add_source(source_pos[0],source_pos[1],update_w_chi2_shift=True,sub_bkg=True,max_offset=40)

        wp_source = tmp_data.source_list[0]
        wp_source.get_cutouts(tmp_data.full_image,tmp_data.DQ_image,band,
                                replace_bad_pixels=True,method='interpolate',
                                bkg_image = tmp_data.bkg_image,sub_bar=sub_bar)
        
        if bkg_method == "cutout_median":
            wp_source.generate_cutout_backgrounds(update=False)
        if update_cutout_backgrounds:
            wp_source.generate_cutout_backgrounds(update=True)

        wp_source.extract_spectra(method=method,verbose=verbose,
                                plot_findTrace=False,plot_optimal_extraction=False,
                                spatial_sigma=3,diag_mask=True,
                                diag_mask_width = 50,fixed_width = fixed_width)

        tmp_data.source_list.append(wp_source)
        tmp_data.n_sources += 1

        output_fname = output_path+filename.rsplit(".fits")[0].split("/")[-1]+output_suffix+".fits"
        tmp_data.save_wirc_object(output_fname,save_full_image = save_full_image)

        so.plot_source_summary(tmp_data,save=True,verbose=True)

    except Exception as e:
        print("Some error with file {}".format(filename))
        print(str(e))

def extract_single_file_parallel_helper(args):
    """This is here to help the parallelization of extract_single_file by unpacking the arguments

    Args:
        args ([list]): a big list of all the arguments to be passed to extract_single_file
    """    
   
    try: 
        extract_single_file(args[0],args[1],args[2],output_path = args[3],output_suffix = args[4],
        verbose = args[5],bkg_method = args[6],num_PCA_modes = args[7],
        update_cutout_backgrounds = args[8], save_full_image=args[9],nclosest=args[10],same_HWP=args[11],
        sub_bar=args[12],method=args[13],fixed_width=args[14])
    except Exception as e:
        print("Some error with file {}".format(args[0]))
        print(e)

def reduce_ABAB_dataset(filelist, source_posA, source_posB,output_path = "./",verbose=False, less_verbose=True,
bkg_methods = ["shift_and_subtract","PCA","median_ref","scaled_bkg","simple_median",
"slit_background","cutout_median"],n_pca=[1,3,5,10,15,20,40], in_slit=False,parallel=False,
n_processes=None,n_per_group=4,nclosest=None,same_HWP=False,sub_bar=False,groupA=None,
groupB=None,method='optimal_extraction',fixed_width=None):
    """A function that reduces a dataset given a list of calibrated science files, assuming you observed in an ABAB dither pattern.
    It uses each position as backgrond for the other. 
    
    It assumes by default that you have 4 images per dither position, but you can change this with the
    n_per_group keyword, or you can manually input groupA and groupB indices. 

    Args:
        filelist ([str]): A list of input filepaths
        source_posA ([float]): The source positions [x,y] of in the A position
        source_posB ([float]): The source positions [x,y] of in the B position
        output_path (str, optional): Where do you want to put all the extracted wirc_object files? I recommend elsewhere. Defaults to "./".
        verbose (bool, optional): How many details do you want?. Defaults to True.
        less_verbose (bool, optional): Somewhere between 'verbose' and silence. Defaults to True.
        bkg_methods (list, optional): A list of all the background subtraction methods that you want to use. Defaults to ["shift_and_subtract","PCA","median_ref","scaled_bkg","simple_median", "slit_background","cutout_median"].
        n_pca (list, optional): A list of PCA modes you want to use for PCA bkg subtraction. Defaults to [1,3,5,10,15,20,40].
        in_slit (bool, optional): Is the source in the slit? If not, then some of the bkg_methods won't be used. Defaults to False.
        parallel (bool, optional): Do you want to run things in parallel? Probably, but sometimes it breaks. Defaults to False.
        n_processes ([type], optional): If you're parallel how many processes do you want to run? Defaults to None.
        n_per_group (int, optional): The number of files per dither group (e.g. at the A position). Defaults to 4.
        nclosest ([type], optional): When building a library of background files to use (from bkg_fnames) for each file you can choose to only use the nclosests files in time. Defaults to None (i.e. use all).
        same_HWP (bool, optional): When choosing bkg files, do you want to use those only with the same HWP angle? Defaults to False.
        sub_bar (bool, optional): Do you want to subtract the back of doom? If you have no background files and you're relying on cutout_median, then maybe. Defaults to False.
        groupA ([type], optional): You can optionally provide the indices in the filelist that correspond to group A. You also need to provide groupB. Defaults to None.
        groupB ([type], optional): You can optionally provide the indices in the filelist that correspond to group A. You also need to provide groupA. Defaults to None.
        method (str, optional): The method you want to use to extract the spectrum. Defaults to 'optimal_extraction'. Another option is 'sum_across_trace'
        fixed_width ([type], optional): You can manually set the width of the extraction region rather than finding it automatically. This can sometimes be more stable. You can pass either a single value or a list of 4 - one for each trace. Defaults to None.
    """
    nfiles = np.size(filelist)
    ngroups = nfiles//(2*n_per_group)

    if groupA is None or groupB is None: 
        groupA = []
        groupB = []
        for i in range(ngroups):
            groupA.append(np.arange(i*n_per_group*2,(2*i+1)*n_per_group))
            groupB.append(np.arange((2*i+1)*n_per_group,(2*i+2)*n_per_group))

        groupA=np.hstack(groupA)
        groupB=np.hstack(groupB)
    
    # Reduce them all with all the possible background subtraction methods - Dither position 1
    reduce_dataset(filelist[groupA], source_posA, bkg_fnames = filelist[groupB], 
                            bkg_methods = bkg_methods,
                            output_path = output_path,
                            verbose=verbose,
                            less_verbose=less_verbose,
                            n_pca=n_pca, in_slit=in_slit,
                            parallel=parallel,n_processes=n_processes,
                            nclosest=nclosest,same_HWP=same_HWP,
                            sub_bar=sub_bar,method=method,fixed_width=fixed_width)

    #Reduce them all with all the possible background subtraction methods - Dither position 2
    reduce_dataset(filelist[groupB], source_posB, bkg_fnames = filelist[groupA], 
                            bkg_methods = bkg_methods,
                            output_path = output_path,
                            verbose=verbose,
                            less_verbose=less_verbose,
                            n_pca=n_pca, in_slit=in_slit,
                            parallel=parallel,n_processes=n_processes,
                            nclosest=nclosest,same_HWP=same_HWP,
                            sub_bar=sub_bar,method=method,fixed_width=fixed_width)

def reduce_dataset_distance(filelist, source_pos, output_path = "./",verbose=False, less_verbose=True,
bkg_methods = ["shift_and_subtract","PCA","median_ref","scaled_bkg","simple_median",
"slit_background","cutout_median"],n_pca=[1,3,5,10,15,20,40], in_slit=False,delta_ra=15,
parallel=False,n_processes=None,parallel_finding=True,fixed_width=None):
    """A version of reduce_dataset that has an ra distance cutoff to use in determining what background to use. 

    This one works best when you have a bright-ish source that can be automatically found by the source finder. 
    
    This one is also a bit out of date compared to reduce_dataset and reduced_ABAB_dataset

    Args:
        filelist ([str]): A list of input filepaths
        source_pos ([float]): The source positions [x,y]
        bkg_fnames ([str], optional): An optional list of background files. Most of the most successful background subtraction methods need this. Defaults to None.
        output_path (str, optional): Where do you want to put all the extracted wirc_object files? I recommend elsewhere. Defaults to "./".
        verbose (bool, optional): How many details do you want?. Defaults to True.
        less_verbose (bool, optional): Somewhere between 'verbose' and silence. Defaults to True.
        bkg_methods (list, optional): A list of all the background subtraction methods that you want to use. Defaults to ["shift_and_subtract","PCA","median_ref","scaled_bkg","simple_median", "slit_background","cutout_median"].
        n_pca (list, optional): A list of PCA modes you want to use for PCA bkg subtraction. Defaults to [1,3,5,10,15,20,40].
        in_slit (bool, optional): Is the source in the slit? If not, then some of the bkg_methods won't be used. Defaults to False.
        delta_ra (int, optional): The minimum distance in pixels you want before you consider a file a good background. Defaults to 15.
        parallel (bool, optional): Do you want to run things in parallel? Probably, but sometimes it breaks. Defaults to False.
        n_processes ([type], optional): If you're parallel how many processes do you want to run? Defaults to None.
        parallel_finding (bool, optional): Do you want to find the sources in parallel as well? Defaults to True.
        fixed_width ([type], optional): You can manually set the width of the extraction region rather than finding it automatically. This can sometimes be more stable. You can pass either a single value or a list of 4 - one for each trace. Defaults to None.

    Raises:
        ValueError: [description]
    """

    ### Get all the RAS
    ## Old Version
    # list_of_headers = []
    # for i in filelist:
    #     list_of_headers += [fits.getheader(i)]
    # #closest in time, some distance away, same HWP
    # all_hdr = list_of_headers
    # #get some useful quantities
    # coords = np.array([ SkyCoord(x['RA'], x['DEC'], unit = (u.hourangle, u.deg)) for x in all_hdr ])
    # ras = np.array([x.ra.degree for x in coords]) #In units of degrees

    if less_verbose:
        print("Finding Source Positions")
    sources = find_dataset_sources(filelist,output_dir=output_path,parallel=parallel_finding,no_save=True)
    source_positions = []
    for fnames,sources in sources:
        source_distances = [np.sqrt((source[0]-source_pos[1])**2+(source[1]-source_pos[0])**2) for source in sources]
        source_positions.append(sources[np.where(source_distances == np.min(source_distances))[0][0]])
    source_positions = np.array(source_positions)
    ras = source_positions[:,1] #Now in units of pixels
    if less_verbose:
        print("")

    #Set up all the directories and filenames: 
    # fnames = np.sort(glob.glob(filelist))
    if len(filelist) < 1:
        raise ValueError("I couldn't find any files!")
    if verbose or less_verbose:
        print("Found {:d} science files".format(len(filelist)))

    for bkg_method in bkg_methods:
        # try: 
        if bkg_method == "slit_background" and not in_slit:
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

                #Let's reduce the files in parallel!
                if parallel: 
                    import multiprocessing as mp

                    #If the user doesn't provide the number of processes,
                    #Then pick the maximum minus 1
                    if n_processes is None:
                        n_processes = mp.cpu_count() - 1

                    #Make the pool
                    # with mp.Pool(processes=n_processes) as pool:
                    # pool = mp.Pool(processes=n_processes)
                    
                        #Package up the arguments (getting the background list looks complicated, 
                        # but is really just a compact version of the non-parallel version): 
                        # args = [(fname,source_pos,filelist[np.where(np.abs(ras-ras[i]) > delta_ra)],
                        # outdir,"",verbose,bkg_method,npca,False,False) for i,fname in enumerate(filelist)]
                        # outputs = pool.map(extract_single_file_parallel_helper,args) 
                        
                        # #Now with the update cutout backgrounds
                        # args = [(fname,source_pos,filelist[np.where(np.abs(ras-ras[i]) > delta_ra)],
                        # outdir2,"",verbose,bkg_method,npca,True,False) for i,fname in enumerate(filelist)]
                        # outputs = pool.map(extract_single_file_parallel_helper,args) 

                    do_parallel_extraction(source_pos,filelist,ras,delta_ra,outdir,verbose,bkg_method,npca,False,False,n_processes)

                    do_parallel_extraction(source_pos,filelist,ras,delta_ra,outdir2,verbose,bkg_method,npca,True,False,n_processes)

                        
                #Or not
                else: 
                
                    for i,fname in enumerate(filelist):
                        ra_dist = np.abs(ras-ras[i])
                        good_bkgs = np.where(ra_dist > delta_ra)

                        if verbose or less_verbose:
                            print("File {} of {} with bkg_method = {}{}: {}".format(i+1,len(filelist),bkg_method,npca,fname))
                        try: 
                            extract_single_file(fname,source_pos, filelist[good_bkgs],output_path=outdir,verbose=verbose,bkg_method=bkg_method,
                            num_PCA_modes=npca)
                        except: 
                            print("Something went wrong with file {}".format(fname))

                        try: 
                            extract_single_file(fname,source_pos, filelist[good_bkgs],output_path=outdir2,verbose=verbose,bkg_method=bkg_method,
                            num_PCA_modes=npca,update_cutout_backgrounds=True)
                        except: 
                            print("Something went wrong with background cutouts with file {} ".format(fname))
        
        else:

            #Make a new directory for this bkg_method
            outdir = output_path+bkg_method + "/"
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            outdir2 = outdir+"bkg_cutout_update/"
            if not os.path.exists(outdir2):
                os.makedirs(outdir2)

            if parallel:

                import multiprocessing as mp

                #If the user doesn't provide the number of processes,
                #Then pick the maximum minus 1
                if n_processes is None:
                    n_processes = mp.cpu_count() - 1

                npca = None
                
                do_parallel_extraction(source_pos,filelist,ras,delta_ra,outdir,verbose,bkg_method,npca,False,False,n_processes,method)
                
                if bkg_method != "cutout_median":
                    do_parallel_extraction(source_pos,filelist,ras,delta_ra,outdir2,verbose,bkg_method,npca,True,False,n_processes,method)

                # #Make the pool
                # with mp.Pool(processes=n_processes) as pool:
                # # pool = mp.Pool(processes=n_processes)
                
                #     #Package up the arguments (getting the background list looks complicated, 
                #     # but is really just a compact version of the non-parallel version): 
                #     args = [(fname,source_pos,filelist[np.where(np.abs(ras-ras[i]) > delta_ra)],
                #     outdir,"",verbose,bkg_method,None,False,False) for i,fname in enumerate(filelist)]
                #     outputs = pool.map(extract_single_file_parallel_helper,args) 

                #     # import pdb; pdb.set_trace()
                #     if bkg_method != "cutout_median":
                #         #Now with the update cutout backgrounds
                #         args = [(fname,source_pos,filelist[np.where(np.abs(ras-ras[i]) > delta_ra)],
                #         outdir2,"",verbose,bkg_method,None,True,False) for i,fname in enumerate(filelist)]
                #         outputs = pool.map(extract_single_file_parallel_helper,args) 

            else:

                for i,fname in enumerate(filelist):
                    ra_dist = np.abs(ras-ras[i])
                    good_bkgs = np.where(ra_dist > delta_ra)

                    if verbose or less_verbose:
                        print("File {} of {} with bkg_method = {}: {}".format(i+1,len(filelist),bkg_method,fname))
                    try: 
                        extract_single_file(fname,source_pos, filelist[good_bkgs],output_path=outdir,verbose=verbose,bkg_method=bkg_method,
                                            method=method)
                    except: 
                        print("Something went wrong with file {}".format(fname))
                    if bkg_method != "cutout_median":
                        try: 
                            extract_single_file(fname,source_pos, filelist[good_bkgs],output_path=outdir2,verbose=verbose,bkg_method=bkg_method,
                            update_cutout_backgrounds=True,method=method)
                        except:
                            print("Something went wrong with bkg_cutouts with file {}".format(fname))

def do_parallel_extraction(source_pos,filelist,ras,delta_ra,outdir,verbose,bkg_method,npca,update_bkg_cutouts,save,n_processes,method):
    '''
    A helper function for reduce_data_distance
    '''
    import multiprocessing as mp
    #Make the pool
    with mp.Pool(processes=n_processes) as pool:
        # pool = mp.Pool(processes=n_processes)
        
        #Package up the arguments (getting the background list looks complicated, 
        # but is really just a compact version of the non-parallel version): 
        args = [(fname,source_pos,filelist[np.where(np.abs(ras-ras[i]) > delta_ra)],
        outdir,"",verbose,bkg_method,npca,update_bkg_cutouts,save,method) for i,fname in enumerate(filelist)]
        outputs = pool.map(extract_single_file_parallel_helper,args)
    

def find_dataset_sources(filelist,output_dir=None,n_processes=None,parallel=True,no_save=False):
    '''
    A function that finds all the sources in a given filelist and then re-saves the file. 

    Returns a list showing the filenames and found sources
    '''

    if output_dir is None:
        print("You MUST provide an output directory")
        return

    output_filenames = [(fname, output_dir+os.path.basename(fname),no_save) for fname in filelist]

    if parallel: 
        import multiprocessing as mp

        #If the user doesn't provide the number of processes,
        #Then pick the maximum minus 1
        if n_processes is None:
            n_processes = mp.cpu_count() - 1

        #Make the pool
        with mp.Pool(processes=n_processes) as pool:
        # pool = mp.Pool(processes=n_processes)
        
            #Package up the arguments: 
            outputs = pool.map(_find_source_and_save,output_filenames) 
    else: 
        outputs = []
        for file_names in output_filenames: 
            outputs.append(_find_source_and_save(file_names))
    
    return outputs

def _find_source_and_save(paths):
    '''
    A utility function that 

    Input: 
    paths   -   A tuple of the form (original_path,output_path)
    '''

    tmp_data = wo.wirc_data(wirc_object_filename=paths[0],verbose=False)
    tmp_data.find_sources_v2(show_plots=False,verbose=False)

    #If not no_save
    if not paths[2]:
        tmp_data.save_wirc_object(paths[1],verbose=False)

    source_poses = [source.pos for source in tmp_data.source_list]

    return paths[0],source_poses

def plot_dataset_extraction_summary(directory,BD_all_spec_cube = None,verbose=True,
                                    save=False,yhigh=None,target_name = "Unknown",prefix = "image"):
    '''
    Produces a plot that summarizes the extractions and polarization from a whole dataset. 
    '''
    
    #The user can either provide a directory to look in, or the BD_all_spec_cube
    if BD_all_spec_cube is None:
        #### Load up the files ####
        filelist = sorted(glob.glob(directory+"/"+prefix+"*.fits"))
        nfiles = len(filelist)
        if nfiles == 0:
            filelist = sorted(glob.glob(directory+"/wirc*.fits"))
            nfiles = len(filelist)
        if nfiles == 0:
            filelist = sorted(glob.glob(directory+"/image*.fits"))
            nfiles = len(filelist)

        # print(nfiles)
        # print("Found {} files".format(nfiles))

        time = np.zeros([nfiles])

        BD_all_spec_cube = []
        hwp_ang = np.zeros([nfiles])
        for im in np.arange(nfiles):
            wirc_object = wo.wirc_data(wirc_object_filename=filelist[im],verbose=False)
            band = wirc_object.header['AFT']
            if 'J' in band:
                band = 'J'
                wirc_object.source_list[0].rough_lambda_calibration(filter_name="J", 
                                                                    method=3, lowcut=0, highcut=-1)
            else:
                band = 'H'
                wirc_object.source_list[0].rough_lambda_calibration(filter_name="H", 
                                                                    method=2, lowcut=0, highcut=-1)
            BD_all_spec_cube.append(wirc_object.source_list[0].trace_spectra)
            hwp_ang[im] = wirc_object.header['HWP_ANG']
            time[im] = wirc_object.header['BJD']

        BD_all_spec_cube = np.array(BD_all_spec_cube)
        
        #Read in the reference spectrum
    # cal_spec = np.load("/scr/data/Other_Files/Tinyanont_2018_Fig6/HD109055_set2.npy")[:,:,84:-60]
    # wvs = su.rough_wavelength_calibration_v2(cal_spec[0,1], 'J')
    # wvs = np.pad(wvs,2,mode='edge')
    # BD_all_spec_cube = su.align_spectral_cube(BD_all_spec_cube,ref_trace=cal_spec[0,1])

    if band == 'J':
        cal_spec = np.load("/scr/data/Other_Files/Tinyanont_2018_Fig6/HD109055_set2.npy")[:,:,84:-60]
        wvs = su.rough_wavelength_calibration_v2(cal_spec[0,1], 'J')
        wvs = np.pad(wvs,2,mode='edge')
        BD_all_spec_cube = su.align_spectral_cube(BD_all_spec_cube,ref_trace=cal_spec[0,1])
    elif band == "H":
        wvs = BD_all_spec_cube[0,0,0]
        BD_all_spec_cube = su.align_spectral_cube(BD_all_spec_cube)
    

    fig = plt.figure(figsize=(20,10),constrained_layout=True)
    # fig.suptitle(wirc_object.header["FN"])
    gs = fig.add_gridspec(4, 9)
    
    ax1 = fig.add_subplot(gs[0,6:])
    ax2 = fig.add_subplot(gs[1,6:])
    ax3 = fig.add_subplot(gs[2,6:])
    ax4 = fig.add_subplot(gs[3,6:])
    
    nspectra = BD_all_spec_cube.shape[0]
    for i in range(nspectra):
        ax1.plot(wvs,BD_all_spec_cube[i,0,1],color="C0",alpha=0.1)
        ax2.plot(wvs,BD_all_spec_cube[i,1,1],color="C1",alpha=0.1)
        ax3.plot(wvs,BD_all_spec_cube[i,2,1],color="C2",alpha=0.1)
        ax4.plot(wvs,BD_all_spec_cube[i,3,1],color="C3",alpha=0.1)

    if band == "J":
        ax1.set_xlim(1.14,1.35)
        ax2.set_xlim(1.14,1.35)
        ax3.set_xlim(1.14,1.35)
        ax4.set_xlim(1.14,1.35)
    if band =="H":
        ax1.set_xlim(1.4,1.85)
        ax2.set_xlim(1.4,1.85)
        ax3.set_xlim(1.4,1.85)
        ax4.set_xlim(1.4,1.85)

    if yhigh is None:
        yhigh = 1.5*np.nanpercentile(BD_all_spec_cube[i,0,1],[99.9])

    ax1.set_ylim(-100,yhigh)
    ax2.set_ylim(-100,yhigh)
    ax3.set_ylim(-100,yhigh)
    ax4.set_ylim(-100,yhigh)
    # else:
    ax1.text(1.15,yhigh*0.85, "Top-Left",fontsize=20)
    ax2.text(1.15,yhigh*0.85, "Bottom-Right",fontsize=20)
    ax3.text(1.15,yhigh*0.85, "Top-Right",fontsize=20)
    ax4.text(1.15,yhigh*0.85, "Bottom-Left",fontsize=20)
    
    ax5 = fig.add_subplot(gs[:,:6])
    
    ax5.plot(wvs,np.median(BD_all_spec_cube[:,0,1],axis=0),color='C0',label="Top-Left")
    ax5.plot(wvs,np.median(BD_all_spec_cube[:,1,1],axis=0),color='C1',label="Bottom-Right")
    ax5.plot(wvs,np.median(BD_all_spec_cube[:,2,1],axis=0),color='C2',label="Top-Right")
    ax5.plot(wvs,np.median(BD_all_spec_cube[:,3,1],axis=0),color='C3',label="Bottom-Left")
    if band =="J":
        ax5.set_xlim(1.14,1.35)
    if band =="H":
        ax5.set_xlim(1.4,1.85)

    ax5.legend(fontsize=20,loc="lower center")
    ax4.set_xlabel(r"Wavelength ($\mu m$)",fontsize=20)
    ax5.set_xlabel(r"Wavelength ($\mu m$)",fontsize=20)
    ax5.set_ylabel("Counts")
    
    # if nfiles is None:
    nfiles = nspectra
    # ax5.text(1.15,yhigh*0.9,"Target Name = {}".format(target_name),fontsize=20)
    # ax5.text(1.15,yhigh*0.86,"Number of Files = {}".format(nfiles),fontsize=20)
    
    ax5.set_title("Target Name = {}, Number of Files = {}".format(target_name,nfiles),fontsize=20)
    
    if save:

        output_dir = directory
            
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                    
        png_filename = output_dir + "/"+target_name+"_extraction_summary.png"
        if verbose:
            print("Saving a summary plot to {}".format(png_filename))
        plt.savefig(png_filename,dpi=200,bbox_inches="tight")

def plot_dataset_variability_summary(directory,BD_all_spec_cube = None, hwp_ang = None,
                                    save=False,png_filename=None,prefix="image",verbose=True,
                                   target_name = "Unknown",nfiles=None,max_files=-1,binsize=4,
                                     all_theta = None,ylow=-0.025,yhigh=0.025,theta_low=0,theta_high=180,
                                    known_period=None,n_time_bins=5,cmap=None,theta_wrap=0,mode="median",
                                    n_time_bins2 = 10, n_wv_bins_2= 5,cmap2="jet"):
    '''
    Produces a plot that summarizes the extractions and polarization from a whole dataset. 
    '''
    
    #### Load up the files ####
    filelist = sorted(glob.glob(directory+"/"+prefix+"*.fits"))
    filelist = filelist[:max_files]
    nfiles = len(filelist)

    time = np.zeros([nfiles])

    BD_all_spec_cube = []
    hwp_ang = np.zeros([nfiles])
    for im in np.arange(nfiles):
        wirc_object = wo.wirc_data(wirc_object_filename=filelist[im],verbose=False)
        wirc_object.source_list[0].rough_lambda_calibration(filter_name="J", 
                                                            method=3, lowcut=0, highcut=-1)
        BD_all_spec_cube.append(wirc_object.source_list[0].trace_spectra)
        hwp_ang[im] = wirc_object.header['HWP_ANG']
        time[im] = wirc_object.header['BJD']

    dt = (time-time[0])*24
    
    if len(dt)%2 ==0:
        dt_qu = 0.5*(dt[::2]+dt[1::2])
    else:
        dt_qu = 0.5*(dt[:-1:2]+dt[1::2])
        
    BD_all_spec_cube = np.array(BD_all_spec_cube)

    #Read in the reference spectrum
    cal_spec = np.load("/scr/data/Other_Files/Tinyanont_2018_Fig6/HD109055_set2.npy")[:,:,84:-60]
    wvs = su.rough_wavelength_calibration_v2(cal_spec[0,1], 'J')
    wvs = np.pad(wvs,2,mode='edge')
    BD_all_spec_cube = su.align_spectral_cube(BD_all_spec_cube,ref_trace=cal_spec[0,1])

    q,u,qerr,uerr,qinds,uinds = so.compute_qu_for_obs_sequence(BD_all_spec_cube,
                                                                   hwp_ang,run_alignment=False)        

    spec = np.nanmean(BD_all_spec_cube[:,:,1,:],axis=(0,1))
    
    
    fig,axes = plt.subplots(4,2,figsize=(20,24))
    
    good_wvs = np.where((wvs>1.16) & (wvs < 1.32))[0]
    
    bb_stokesI = np.sum(BD_all_spec_cube[:,:,1,good_wvs],axis=(1,2))
    
    norm_I = bb_stokesI/np.median(bb_stokesI)
                            
    axes[0,0].plot((time-time[0])*24,bb_stokesI/np.median(bb_stokesI),'o')
    
    axes[0,0].set_xlabel("Time from start (h)",fontsize=16)
    axes[0,0].set_ylabel("Normalized Trace Sums",fontsize=16)
    axes[0,0].set_title("Sum of all traces normalized by the median",fontsize=20)
                        

    good_indss = np.isfinite(dt) & np.isfinite(norm_I)
    ls = LombScargle(dt[good_indss], norm_I[good_indss])
    frequency, power = ls.autopower()
    #The false alarm levels
    probabilities = [0.1, 0.05, 0.01]
    ls_fal = ls.false_alarm_level(probabilities)  
    
    axes[0,1].plot(1/frequency,power)
    axes[0,1].set_xlabel("Period (h)",fontsize=16)
    axes[0,1].set_ylabel("Power",fontsize=16)
    axes[0,1].set_title("Stokes I Lomb-Scargle Periodogram",fontsize=20)
    axes[0,1].set_xlim(0,np.max(dt)*2)
    
    axes[0,1].axhline(ls_fal[0],color='k',label="10% False Alarm Probability")
    axes[0,1].axhline(ls_fal[1],color='k',label="5% False Alarm Probability",linestyle="--")
    axes[0,1].axhline(ls_fal[2],color='k',label="1% False Alarm Probability",linestyle="-.")
    
    if known_period is not None:
        axes[0,1].axvline(known_period,color='r',linestyle='-.',label="Known Period")
    
    axes[0,1].legend(fontsize=16)
     
    ### Get the broadband q and u
    bb_cube = np.zeros(BD_all_spec_cube.shape[:3])
    low = np.max(np.where(wvs<1.16))
    high = np.min(np.where(wvs>1.32))

    for i in range(bb_cube.shape[0]):
        bb_cube[i,:,0] = np.sum(BD_all_spec_cube[i,:,1,low:high]*BD_all_spec_cube[i,:,0,low:high],axis=1)/np.sum(BD_all_spec_cube[i,:,1,low:high],axis=1)
        bb_cube[i,:,1] = np.sum(BD_all_spec_cube[i,:,1,low:high],axis=1)
        bb_cube[i,:,2] = np.sqrt(np.sum(BD_all_spec_cube[i,:,2,low:high]**2,axis=1))
        
    q_bb,u_bb,qerr_bb,uerr_bb,qind_bb,uind_bb = so.compute_qu_for_obs_sequence(bb_cube[:,:,:,None],
                                                                       hwp_ang,run_alignment=False,
                                                                       method="double_difference")
    #Sometimes compute_qu ditches some of the later times steps if we've missing HWP angles. 
    dt_qu = dt_qu[:q_bb.shape[0]]

    axes[1,0].errorbar(dt_qu,q_bb[:,0],yerr=qerr_bb[:,0],label="q",marker='o',linestyle="")
    axes[1,0].errorbar(dt_qu,u_bb[:,0],yerr=uerr_bb[:,0],label="u",marker='o',linestyle="")
    axes[1,0].axhline(0,color='r',linestyle='-.')
    
    # bb_ylow = np.percentile(np.append(q_bb[:,0],u_bb[:,0]),[5,99])
    
    # axes[1,0].set_ylim(2*bb_ylow[0],2*bb_ylow[1])
    axes[1,0].set_ylim(ylow,yhigh)
    axes[1,0].set_xlabel("Time from start (h)",fontsize=16)
    axes[1,0].set_ylabel("q or u",fontsize=16)
    axes[1,0].set_title("Broadband q and u",fontsize=20)
    axes[1,0].legend(fontsize=16)
        
    qls = LombScargle(dt_qu, q_bb[:,0],qerr_bb[:,0])
    qfrequency, qpower = qls.autopower()

    uls = LombScargle(dt_qu, u_bb[:,0],uerr_bb[:,0])
    ufrequency, upower = uls.autopower()

    axes[1,1].plot(1/qfrequency,qpower,label="q")
    axes[1,1].plot(1/ufrequency,upower,label="u")
    axes[1,1].set_xlim(0,np.max(dt_qu)*2)
    
    
    #False Alarm Probabilities
    qls_fal = qls.false_alarm_level(probabilities)  
    uls_fal = uls.false_alarm_level(probabilities)  
    axes[1,1].axhline(qls_fal[0],color='C0',label="10% False Alarm Probability",linestyle='dotted',alpha=0.5)
    axes[1,1].axhline(qls_fal[1],color='C0',label="5% False Alarm Probability",linestyle="--",alpha=0.5)
    axes[1,1].axhline(qls_fal[2],color='C0',label="1% False Alarm Probability",linestyle="-.",alpha=0.5)
    axes[1,1].axhline(uls_fal[0],color='C1',label="10% False Alarm Probability",linestyle='dotted',alpha=0.5)
    axes[1,1].axhline(uls_fal[1],color='C1',label="5% False Alarm Probability",linestyle="--",alpha=0.5)
    axes[1,1].axhline(uls_fal[2],color='C1',label="1% False Alarm Probability",linestyle="-.",alpha=0.5)

    if known_period is not None:
        axes[1,1].axvline(known_period,color='r',linestyle='-.',label="Known Period")

    
    axes[1,1].legend(fontsize=16)

    
    axes[1,1].set_xlabel("Period (h)",fontsize=16)
    axes[1,1].set_ylabel("Power",fontsize=16)
    axes[1,1].set_title("Broadband q and u Lomb-Scargle Periodograms",fontsize=20)
    
    
    ### Get the mean and std broadband values: 
    from astropy import stats
    mn_qbb,_,std_qbb = stats.sigma_clipped_stats(q_bb, sigma=3, maxiters=5)
    mn_ubb,_,std_ubb = stats.sigma_clipped_stats(u_bb, sigma=3, maxiters=5)
    
    axes[1,0].text(0,0.88*yhigh,"Mean q = {:.2f} +/- {:.2f}%".format(mn_qbb*100,std_qbb*100/np.sqrt(len(q_bb))),fontsize=15)
    axes[1,0].text(0,0.75*yhigh,"Mean u = {:.2f} +/- {:.2f}%".format(mn_ubb*100,std_ubb*100/np.sqrt(len(u_bb))),fontsize=15)
    
    
    
    #####################################################################
    ### Now Look for variability in the polarized spectrum of q and u ###
    if known_period is not None:
        phase = (dt % known_period)/known_period
    else:
        phase = np.linspace(0,1.0,n_time_bins,endpoint=False)
    
    #Cycle through the time bins
    time_snip = dt.shape[0] % (n_time_bins*4)
    time_bin_size = (dt.shape[0]-time_snip)//n_time_bins

    good_wvs = (wvs > 1.15) & (wvs < 1.34)
    spec_cube = copy.deepcopy(BD_all_spec_cube[:,:,:,good_wvs])
    master_wvs = copy.deepcopy(wvs[good_wvs])
    master_spec = copy.deepcopy(spec[good_wvs])


    if time_snip != 0:
        print(dt.shape)
        print(time_snip)
        dt_bins = np.mean(np.reshape(dt[:-time_snip],(-1,time_bin_size)),axis=1)
    else:
        print(dt.shape)
        dt_bins = np.mean(np.reshape(dt,(-1,time_bin_size)),axis=1)

    if known_period is None:
        known_period = np.max(dt)

    phase = (dt_bins % known_period)/known_period

    time_inds = np.arange(dt.shape[0])
    
    ## Setup a colormap
    if cmap is None:
        colormapp = mpl.colors.LinearSegmentedColormap.from_list("", ["k","firebrick","crimson","darkorchid","blueviolet","mediumblue","navy","k"])
    else:
        colormapp = plt.get_cmap(cmap)
    # print(phase)
    colors = colormapp(phase)

    colors2 = plt.get_cmap(cmap2)((np.arange(n_wv_bins_2)+0.5)/n_wv_bins_2)
        
    
    ## Cycle through the different time steps
    for k in range(n_time_bins):
        # print("time_bin_size = {}".format(time_bin_size)) 
        good_inds = time_inds[np.where((time_inds >= k*time_bin_size) & (time_inds < (k+1)*time_bin_size))]

        # import pdb; pdb.set_trace()
        # print("Using inds {}".format(good_inds))
        q,u,qerr,uerr,qind,uind = so.compute_qu_for_obs_sequence(spec_cube[good_inds],hwp_ang[good_inds],run_alignment=False)

        #First calculate the double_difference values
        p = np.sqrt(q**2+u**2)
        theta = 0.5*np.degrees(np.arctan2(u,q))
        theta[theta < 0] +=180
        
        #Now calculate the mean or median

        q_mean = np.zeros([q.shape[1]]) #We name this mean, though it could either be Mean or Median
        q_std = np.zeros([q.shape[1]])
        u_mean = np.zeros([u.shape[1]])
        u_std = np.zeros([u.shape[1]])

        for i in range(q.shape[1]):
            # import pdb; pdb.set_trace()
            mn,md,std = stats.sigma_clipped_stats(q[:,i], sigma=3, maxiters=5)
            if mode == 'median':
                q_mean[i] = md
            else:
                q_mean[i] = mn
            q_std[i] = std/np.sqrt(q.shape[0])
            mn,md,std = stats.sigma_clipped_stats(u[:,i], sigma=3, maxiters=5)
            u_std[i] = std/np.sqrt(q.shape[0])
            if mode == 'median':
                u_mean[i] = md
            else:
                u_mean[i] = mn
            
            p_mean = np.sqrt(q_mean**2+u_mean**2)
            theta_mean = 0.5*np.degrees(np.arctan2(u_mean,q_mean))
            theta_mean[theta_mean < theta_wrap] +=180

            q_mean_err = np.sqrt(np.nansum(qerr**2,axis=0))/qerr.shape[0]
            u_mean_err = np.sqrt(np.nansum(uerr**2,axis=0))/uerr.shape[0]
            p_mean_err = np.sqrt(q_mean**2*q_mean_err**2+u_mean**2*u_mean_err**2)/p_mean
            p_std = np.sqrt(q_mean**2*q_std**2+u_mean**2*u_std**2)/p_mean
            theta_mean_err = 0.5*np.degrees( np.sqrt( (u_mean**2*q_mean_err**2+q_mean**2*u_mean_err**2)/(q_mean**2+u_mean**2)**2))

        ### Implement Binning
        if binsize != 1:
            snip = q_mean.shape[0] % binsize
            # print("Snip = {}".format(snip))
            if snip != 0:
                # print(q_mean.shape)
                q_mean = np.mean(q_mean[:-snip].reshape(-1,binsize),axis=1)
                u_mean = np.mean(u_mean[:-snip].reshape(-1,binsize),axis=1)
                q_mean_err = np.sqrt(np.sum(q_mean_err[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
                q_std = np.sqrt(np.sum(q_std[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
                u_mean_err = np.sqrt(np.sum(u_mean_err[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
                u_std = np.sqrt(np.sum(u_std[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
                wvs_bin = np.mean(wvs[good_wvs][:-snip].reshape(-1,binsize),axis=1)

            else: 
                q_mean = np.mean(q_mean.reshape(-1,binsize),axis=1)
                u_mean = np.mean(u_mean.reshape(-1,binsize),axis=1)
                q_mean_err = np.sqrt(np.sum(q_mean_err.reshape(-1,binsize)**2,axis=1))/binsize
                q_std = np.sqrt(np.sum(q_std.reshape(-1,binsize)**2,axis=1))/binsize
                u_mean_err = np.sqrt(np.sum(u_mean_err.reshape(-1,binsize)**2,axis=1))/binsize
                u_std = np.sqrt(np.sum(u_std.reshape(-1,binsize)**2,axis=1))/binsize
                # dt_bin2 = np.mean(dt.reshape(-1,binsize),axis=1)
                wvs_bin = np.mean(wvs[good_wvs].reshape(-1,binsize),axis=1)

        ##### Plot Q, U, P and theta ######
        #The mean values
        if mode == "median":
            axes[2,0].plot(wvs_bin,q_mean,color=colors[k],label="Median")
        else:
            axes[2,0].plot(wvs_bin,q_mean,color=colors[k],label="Mean")    
        axes[2,1].plot(wvs_bin,u_mean,color=colors[k])           

        #Make a line at zero
        axes[2,0].axhline(0.,color='k',linestyle='--')
        axes[2,1].axhline(0.,color='k',linestyle='--')

        #Fill in photon/ron error ranges
        axes[2,0].fill_between(wvs_bin,q_mean+q_std,q_mean-q_std,color=colors[k],alpha=0.1)
        axes[2,1].fill_between(wvs_bin,u_mean+u_std,u_mean-u_std,color=colors[k],alpha=0.1)

    ####Add the colormaps to show the phase colors
    #Top left
    cbaxes = inset_axes(axes[2,0], width="60%", height="7%", loc=9) 
    cbar = mpl.colorbar.ColorbarBase(cbaxes, cmap = colormapp,norm = mpl.colors.Normalize(vmin=0,vmax=1.),ticks=[0.,0.5,1], orientation='horizontal')
    if known_period == np.max(dt):
        cbar.set_label("Phase (Period = ??)",fontsize=16)
    else:
        cbar.set_label("Phase (Period = {}h)".format(known_period),fontsize=16)
    cbaxes.scatter(phase,phase*0.,marker="^",color='white',s=900)

    for i,p in enumerate(phase):
        cbaxes.text(p-0.015,0.075,"{:d}".format(i),weight="bold")

    #Top Right
    cbaxes = inset_axes(axes[2,1], width="60%", height="7%", loc=9) 
    cbar = mpl.colorbar.ColorbarBase(cbaxes, cmap = colormapp,norm = mpl.colors.Normalize(vmin=0,vmax=1.),ticks=[0.,0.5,1], orientation='horizontal')
    if known_period == np.max(dt):
        cbar.set_label("Phase (Period = ??)",fontsize=16)
    else:
        cbar.set_label("Phase (Period = {}h)".format(known_period),fontsize=16)
    cbaxes.scatter(phase,phase*0.,marker="^",color='white',s=900)
    for i,p in enumerate(phase):
        cbaxes.text(p-0.015,0.075,"{:d}".format(i),weight="bold")

    axes[2,0].set_xlim(1.15,1.34)
    axes[2,1].set_xlim(1.15,1.34)
    axes[2,0].set_ylim(ylow,yhigh)
    axes[2,1].set_ylim(ylow,yhigh)
    axes[2,0].set_xlabel(r"Wavelength ($\mu m$)",fontsize=16)
    axes[2,1].set_xlabel(r"Wavelength ($\mu m$)",fontsize=16)
    axes[2,0].set_ylabel(r"q")
    axes[2,1].set_ylabel(r"q")
    
    # axes[2,0].legend()
    
    axes[2,0].fill_between([0,1.17],ylow,yhigh,hatch='x',color='k',alpha=0.05)
    axes[2,1].fill_between([0,1.17],ylow,yhigh,hatch='x',color='k',alpha=0.05)
    
    axes[2,0].fill_between([1.32,1.34],ylow,yhigh,hatch='x',color='k',alpha=0.05)
    axes[2,1].fill_between([1.32,1.34],ylow,yhigh,hatch='x',color='k',alpha=0.05)

    axes[2,0].set_title(r"q spectrum variability +/- $\sigma_{SOE}$",fontsize=20)
    axes[2,1].set_title(r"u spectrum variability +/- $\sigma_{SOE}$",fontsize=20)
    
    fig.suptitle(directory,fontsize=22)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    #################################################################
    ###### Ok, now let's bin the wavelengths down into 5 bins #######
    #################################################################
    
    ####Add the colormaps to show the phase colors
    #Top left
    wv_range = np.max(wvs[good_wvs])-np.min(wvs[good_wvs])
    cbaxes1 = inset_axes(axes[3,0], width="60%", height="7%", loc=9) 
    cbar = mpl.colorbar.ColorbarBase(cbaxes1, cmap = cmap2,norm = mpl.colors.Normalize(vmin=0,vmax=1.),ticks=[0.,0.25,0.5,0.75,1], orientation='horizontal')
    cbar.set_ticklabels(["{:.2f}".format(np.min(wvs[good_wvs])),
                         "{:.2f}".format(np.min(wvs[good_wvs])+0.25*wv_range),
                         "{:.2f}".format(np.min(wvs[good_wvs])+0.5*wv_range),
                         "{:.2f}".format(np.min(wvs[good_wvs])+0.75*wv_range),
                         "{:.2f}".format(np.max(wvs[good_wvs]))])
    cbar.set_label(r"Wavelength ($\mu m$)",fontsize=16)


    # for i,p in enumerate(wvs_bin):
        # cbaxes.text((mn_wvs-1.15)/wv_range-0.015,0.075,"{:d}".format(i),weight="bold")

    #Top Right
    cbaxes2 = inset_axes(axes[3,1], width="60%", height="7%", loc=9) 
    cbar2 = mpl.colorbar.ColorbarBase(cbaxes2, cmap = cmap2,norm = mpl.colors.Normalize(vmin=0,vmax=1.),ticks=[0.,0.25,0.5,0.75,1], orientation='horizontal')
    cbar2.set_ticklabels(["{:.2f}".format(np.min(wvs[good_wvs])),
                 "{:.2f}".format(np.min(wvs[good_wvs])+0.25*(np.max(wvs[good_wvs])-np.min(wvs[good_wvs]))),
                 "{:.2f}".format(np.min(wvs[good_wvs])+0.5*(np.max(wvs[good_wvs])-np.min(wvs[good_wvs]))),
                 "{:.2f}".format(np.min(wvs[good_wvs])+0.75*(np.max(wvs[good_wvs])-np.min(wvs[good_wvs]))),
                 "{:.2f}".format(np.max(wvs[good_wvs]))])

    cbar2.set_label(r"Wavelength ($\mu m$)",fontsize=16)
        # for i,p in enumerate(phase):
            # cbaxes.text(p-0.015,0.075,"{:d}".format(i),weight="bold")

    axes[3,0].set_ylim(ylow,yhigh)
    axes[3,1].set_ylim(ylow,yhigh)
    
    axes[3,0].set_xlabel(r"Time from start (h)")
    axes[3,1].set_xlabel(r"Time from start (h)")

    axes[3,0].set_ylabel(r"q",fontsize=16)
    axes[3,1].set_ylabel(r"u",fontsize=16)
    
    n_good_wvs = len(np.where(good_wvs)[0])
    wvs_bin_length = n_good_wvs//n_wv_bins_2
    wvs_snip = n_good_wvs % n_wv_bins_2
    
    
    #For each wavelength bin
    for k in range(n_wv_bins_2):
        
        #Pick the wavelengths for this bin
        these_wvs = np.arange(n_good_wvs)[k*wvs_bin_length:(k+1)*wvs_bin_length]
        mn_wvs = np.nanmean(wvs[good_wvs][these_wvs])
    
        # print(spec_cube.shape)
        q,u,qerr,uerr,qind,uind = so.compute_qu_for_obs_sequence(spec_cube[:,:,:,these_wvs],hwp_ang,run_alignment=False)

        #Now calculate the mean or median over "these_wvs"
        q_mean = np.zeros([q.shape[0]]) #We name this mean, though it could either be Mean or Median
        q_std = np.zeros([q.shape[0]])
        u_mean = np.zeros([u.shape[0]])
        u_std = np.zeros([u.shape[0]])

        #Cycle through the time steps and get the mean or median at each time step
        for i in range(q.shape[0]):
            # import pdb; pdb.set_trace()
            mn,md,std = stats.sigma_clipped_stats(q[i,:], sigma=3, maxiters=5)
            if mode == 'median':
                q_mean[i] = md
            else:
                q_mean[i] = mn
            q_std[i] = std/np.sqrt(q.shape[1])
            mn,md,std = stats.sigma_clipped_stats(u[i,:], sigma=3, maxiters=5)
            u_std[i] = std/np.sqrt(u.shape[1])

            if mode == 'median':
                u_mean[i] = md
            else:
                u_mean[i] = mn
            q_mean_err = np.sqrt(np.nansum(qerr**2,axis=1))/qerr.shape[0]
            u_mean_err = np.sqrt(np.nansum(uerr**2,axis=1))/uerr.shape[0]

        n_obs = q.shape[0]
        time_bin_size=q.shape[0]//n_time_bins2

        ### Implement the Time Binning
        snip = q_mean.shape[0] % time_bin_size
        if snip != 0:
            q_mean = np.nanmedian(q_mean[:-snip].reshape(-1,time_bin_size),axis=1)
            u_mean = np.nanmedian(u_mean[:-snip].reshape(-1,time_bin_size),axis=1)
            q_mean_err = np.sqrt(np.nansum(q_mean_err[:-snip].reshape(-1,time_bin_size)**2,axis=1))/time_bin_size
            q_std = np.sqrt(np.nansum(q_std[:-snip].reshape(-1,time_bin_size)**2,axis=1))/time_bin_size
            u_mean_err = np.sqrt(np.sum(u_mean_err[:-snip].reshape(-1,time_bin_size)**2,axis=1))/time_bin_size
            u_std = np.sqrt(np.nansum(u_std[:-snip].reshape(-1,time_bin_size)**2,axis=1))/time_bin_size
            dt_bin = np.nanmean(dt_qu[:-snip].reshape(-1,time_bin_size),axis=1)
        else: 
            q_mean = np.nanmedian(q_mean.reshape(-1,time_bin_size),axis=1)
            u_mean = np.nanmedian(u_mean.reshape(-1,time_bin_size),axis=1)
            wvs_bin = np.nanmean(wvs[good_wvs].reshape(-1,time_bin_size),axis=1)
            q_mean_err = np.sqrt(np.nansum(q_mean_err.reshape(-1,time_bin_size)**2,axis=1))/time_bin_size
            q_std = np.sqrt(np.nansum(q_std.reshape(-1,time_bin_size)**2,axis=1))/time_bin_size
            u_mean_err = np.sqrt(np.nansum(u_mean_err.reshape(-1,time_bin_size)**2,axis=1))/time_bin_size
            u_std = np.sqrt(np.nansum(u_std.reshape(-1,time_bin_size)**2,axis=1))/time_bin_size
            dt_bin = np.nanmean(dt_qu.reshape(-1,time_bin_size),axis=1)


        ##### Plot Q, U, P and theta ######
        #The mean values
        if mode == "median":
            axes[3,0].plot(dt_bin,q_mean,color=colors2[k],label="Median")
        else:
            axes[3,0].plot(dt_bin,q_mean,color=colors2[k],label="Mean")    
        axes[3,1].plot(dt_bin,u_mean,color=colors2[k])           

        #Make a line at zero
        axes[3,0].axhline(0.,color='k',linestyle='--')
        axes[3,1].axhline(0.,color='k',linestyle='--')

        #Fill in photon/ron error ranges
        axes[3,0].fill_between(dt_bin,q_mean+q_std,q_mean-q_std,color=colors2[k],alpha=0.1)
        axes[3,1].fill_between(dt_bin,u_mean+u_std,u_mean-u_std,color=colors2[k],alpha=0.1)
        
        # axes[3,0].plot(dt_bin,q_mean+q_std,linestyle='--',color=colors2[k])
        # axes[3,0].plot(dt_bin,q_mean-q_std,linestyle='--',color=colors2[k])
        # axes[3,1].plot(dt_bin,u_mean+u_std,linestyle='--',color=colors2[k])
        # axes[3,1].plot(dt_bin,u_mean-u_std,linestyle='--',color=colors2[k])

        cbaxes1.scatter((mn_wvs-np.min(wvs[good_wvs]))/wv_range,mn_wvs*0.,marker="^",color='white',s=900)
        cbaxes2.scatter((mn_wvs-np.min(wvs[good_wvs]))/wv_range,mn_wvs*0.,marker="^",color='white',s=900)
        
        axes[3,0].set_title(r"q spectrum variability +/- $\sigma_{SOE}$",fontsize=20)
        axes[3,1].set_title(r"u spectrum variability +/- $\sigma_{SOE}$",fontsize=20)

    if save:
        output_dir = directory

        if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        png_filename = output_dir + "/"+target_name+"_variability_summary.png"
        if verbose:
            print("Saving a summary plot to {}".format(png_filename))
        plt.savefig(png_filename,dpi=200,bbox_inches="tight")
                            
def plot_dataset_broadband_summary(directory,save=False,png_filename=None,prefix="image",verbose=True,
                                max_files=None,ylow=-0.025,yhigh=0.025,start_file =0,
                                known_period=None,target_name="Unknown",calibrate=True):

    #### Load up the files ####
    filelist = sorted(glob.glob(directory+"/"+prefix+"*.fits"))
    filelist=filelist[start_file:]
    if max_files is not None:
        filelist = filelist[:max_files]
    nfiles = len(filelist)

    time = np.zeros([nfiles])

    BD_all_spec_cube = []
    hwp_ang = np.zeros([nfiles])
    for im in np.arange(nfiles):
        wirc_object = wo.wirc_data(wirc_object_filename=filelist[im],verbose=False)
        band = wirc_object.header['AFT']
        if 'J' in band:
            band = 'J'
            wirc_object.source_list[0].rough_lambda_calibration(filter_name="J", 
                                                                method=3, lowcut=0, highcut=-1)
        else:
            band = 'H'
            wirc_object.source_list[0].rough_lambda_calibration(filter_name="H", 
                                                                method=2, lowcut=0, highcut=-1)
        
        BD_all_spec_cube.append(wirc_object.source_list[0].trace_spectra)
        hwp_ang[im] = wirc_object.header['HWP_ANG']
        time[im] = wirc_object.header['BJD']
        

    dt = (time-time[0])*24
    
    if len(dt)%2 ==0:
        dt_qu = 0.5*(dt[::2]+dt[1::2])
    else:
        dt_qu = 0.5*(dt[:-1:2]+dt[1::2])
        
    BD_all_spec_cube = np.array(BD_all_spec_cube)
    
    #Read in the reference spectrum
    
    spec = np.nanmean(BD_all_spec_cube[:,:,1,:],axis=(0,1))

    if band == 'J':
        cal_spec = np.load("/scr/data/Other_Files/Tinyanont_2018_Fig6/HD109055_set2.npy")[:,:,84:-60]
        wvs = su.rough_wavelength_calibration_v2(cal_spec[0,1], 'J')
        wvs = np.pad(wvs,2,mode='edge')
        BD_all_spec_cube = su.align_spectral_cube(BD_all_spec_cube,ref_trace=cal_spec[0,1])
        lows = [0,np.max(np.where(wvs<1.15)),np.max(np.where(wvs<1.16)),np.max(np.where(wvs<1.17))]
        highs = [-1,np.min(np.where(wvs>1.34)),np.min(np.where(wvs>1.33)),np.min(np.where(wvs>1.32))]
    elif band == "H":
        trace_md_spec = np.median(BD_all_spec_cube,axis=0)
        wvs = su.rough_wavelength_calibration_v2(trace_md_spec[0,1,:], band)
        BD_all_spec_cube = su.align_spectral_cube(BD_all_spec_cube,ref_trace=trace_md_spec[0,1])
        lows = [0,np.max(np.where(wvs<1.45)),np.max(np.where(wvs<1.49)),np.max(np.where(wvs<1.52))]
        highs = [-1,np.min(np.where(wvs>1.82)),np.min(np.where(wvs>1.8)),np.min(np.where(wvs>1.78))]
    

    n_plots = len(lows)
    
    #Plot Setup
    fig = plt.figure(figsize=(20,n_plots*4))
    gs = fig.add_gridspec(n_plots+1, 3)
    ax1 = fig.add_subplot(gs[0,:])
    
    ax1.plot(wvs,spec,label="Source Extracted Spectrum")
    
    
    for i in range(len(lows)):
        
        low = lows[i]
        high = highs[i]
        
        ax1.axvline(wvs[low],color='C{:d}'.format(i+1),label="Broadband Range {:d}".format(i+1))
        if i == 0:
            ax1.axvline(np.nanmax(wvs),color='C{:d}'.format(i+1))
        else:
            ax1.axvline(wvs[high],color='C{:d}'.format(i+1))
        
        axn1 = fig.add_subplot(gs[i+1,0])
        axn2 = fig.add_subplot(gs[i+1,1])
        axn3 = fig.add_subplot(gs[i+1,2])
        
        ### Get the broadband q and u
        bb_cube = np.zeros(BD_all_spec_cube.shape[:3])

        for j in range(bb_cube.shape[0]):
            bb_cube[j,:,0] = np.sum(BD_all_spec_cube[j,:,1,low:high]*BD_all_spec_cube[j,:,0,low:high],axis=1)/np.sum(BD_all_spec_cube[j,:,1,low:high],axis=1)
            bb_cube[j,:,1] = np.sum(BD_all_spec_cube[j,:,1,low:high],axis=1)
            bb_cube[j,:,2] = np.sqrt(np.sum(BD_all_spec_cube[j,:,2,low:high]**2,axis=1))

        q_bb,u_bb,qerr_bb,uerr_bb,qind_bb,uind_bb = so.compute_qu_for_obs_sequence(bb_cube[:,:,:,None],
                                                                           hwp_ang,run_alignment=False)
        
    
    ## Now let's calibrate!
        # print(q_bb) 
        if calibrate:
            for i in range(len(q_bb[qind_bb==0])):
                ind = np.where(qind_bb==0)[0][i]

                qc,uc,qc_err,uc_err = calibration.calibrate_qu(bb_cube[0,0,0],q_bb[ind],u_bb[ind],qerr_bb[ind],uerr_bb[ind],trace_pair=0)
                q_bb[ind] = qc
                u_bb[ind] = uc
                qerr_bb[ind] = qc_err
                uerr_bb[ind] = uc_err

            for i in range(len(q_bb[qind_bb==1])):
                ind = np.where(qind_bb==1)[0][i]
                qc,uc,qc_err,uc_err = calibration.calibrate_qu(bb_cube[0,0,0],q_bb[ind],u_bb[ind],qerr_bb[ind],uerr_bb[ind],trace_pair=1)
                q_bb[ind] = qc
                u_bb[ind] = uc
                qerr_bb[ind] = qc_err
                uerr_bb[ind] = uc_err

        #Soemtimes compute_qu dithces some of the last values to match HWPs
        dt_qu = dt_qu[:q_bb.shape[0]]

        axn1.errorbar(dt_qu,q_bb[:,0],yerr=qerr_bb[:,0],label="q",marker='o',linestyle="")
        axn1.errorbar(dt_qu,u_bb[:,0],yerr=uerr_bb[:,0],label="u",marker='o',linestyle="")
        axn1.axhline(0,color='r',linestyle='-.')

        # bb_ylow = np.percentile(np.append(q_bb[:,0],u_bb[:,0]),[5,99])

        # axes[1,0].set_ylim(2*bb_ylow[0],2*bb_ylow[1])
        axn1.set_ylim(ylow,yhigh)
        axn1.set_xlabel("Time from start (h)",fontsize=16)
        axn1.set_ylabel("q or u for range {:d}".format(i+1),fontsize=16)
        if i == 0:
            axn1.set_title("Broadband q and u",fontsize=20)
        axn1.legend(fontsize=16)
        
        from astropy.timeseries import LombScargle
        probabilities = [0.1, 0.05, 0.01]
        qls = LombScargle(dt_qu, q_bb[:,0],qerr_bb[:,0])
        qfrequency, qpower = qls.autopower()

        uls = LombScargle(dt_qu, u_bb[:,0],uerr_bb[:,0])
        ufrequency, upower = uls.autopower()

        axn2.plot(1/qfrequency,qpower,label="q")
        axn2.plot(1/ufrequency,upower,label="u")
        axn2.set_xlim(0,np.max(dt_qu)*2)


        #False Alarm Probabilities
        qls_fal = qls.false_alarm_level(probabilities)  
        uls_fal = uls.false_alarm_level(probabilities)  
        axn2.axhline(qls_fal[0],color='C0',label="10% False Alarm Probability",linestyle='dotted',alpha=0.5)
        axn2.axhline(qls_fal[1],color='C0',label="5% False Alarm Probability",linestyle="--",alpha=0.5)
        axn2.axhline(qls_fal[2],color='C0',label="1% False Alarm Probability",linestyle="-.",alpha=0.5)
        axn2.axhline(uls_fal[0],color='C1',label="10% False Alarm Probability",linestyle='dotted',alpha=0.5)
        axn2.axhline(uls_fal[1],color='C1',label="5% False Alarm Probability",linestyle="--",alpha=0.5)
        axn2.axhline(uls_fal[2],color='C1',label="1% False Alarm Probability",linestyle="-.",alpha=0.5)

        if known_period is not None:
            axn2.axvline(known_period,color='r',linestyle='-.',label="Known Period")


        axn2.legend(fontsize=8)

        axn2.set_xlabel("Period (h)",fontsize=16)
        axn2.set_ylabel("Power",fontsize=16)
        if i == 0:
            axn2.set_title("Broadband q and u Lomb-Scargle Periodograms",fontsize=20)
            axn3.set_title("q and u histograms")


        ### Get the mean and std broadband values: 
        from astropy import stats
        mn_qbb,md_qbb,std_qbb = stats.sigma_clipped_stats(q_bb, sigma=3, maxiters=5)
        mn_ubb,md_ubb,std_ubb = stats.sigma_clipped_stats(u_bb, sigma=3, maxiters=5)

        #### Now histograms ###
        q_bb_percentiles  = np.nanpercentile(q_bb,[3,97])
        u_bb_percentiles  = np.nanpercentile(u_bb,[3,97])
        h1 = axn3.hist(q_bb,alpha=0.7,range=q_bb_percentiles,density=True)
        h2 = axn3.hist(u_bb,alpha=0.7,range=u_bb_percentiles,density=True)
        
        if np.max(h1[0]) < np.max(h2[0]):
            spot = np.max(h2[0])
        else:
            spot = np.max(h1[0])

        #Now calibrate! 
        
        axn3.text(q_bb_percentiles[0],0.9*spot,"Median q = {:.2f} +/- {:.2f}%".format(md_qbb*100,std_qbb*100/np.sqrt(len(q_bb))),fontsize=12)
        axn3.text(q_bb_percentiles[0],0.75*spot,"Median u = {:.2f} +/- {:.2f}%".format(md_ubb*100,std_ubb*100/np.sqrt(len(u_bb))),fontsize=12)

        axn3.set_xlabel("q or u")
        axn3.set_ylabel("# of datapoints")
        
    ax1.set_title("Broadband Extraction Ranges",fontsize=16)
    ax1.set_ylabel("Mean Counts")
    ax1.set_xlabel(r"Wavelength ($\mu m$)")
    ax1.legend()
    fig.suptitle(directory,fontsize=22)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if save:

        output_dir = directory

        if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        png_filename = output_dir + "/"+target_name+"_broadband_summary.png"
        if verbose:
            print("Saving a summary plot to {}".format(png_filename))
        plt.savefig(png_filename,dpi=200,bbox_inches="tight")

def plot_bkg_method_comparison(home_directory,save=False,prefix="image",verbose=True,
                                   max_files=-1,ylow=-0.025,yhigh=0.025,
                                    known_period=None,target_name="Unknown",
                               exclude_methods=["PCA1","PCA3","PCA15","PCA20"],
                              mode="median", extra_methods = []):
    
    #Get all the methods
    list_subfolders_with_paths = [f.path.rsplit("/",1)[-1] for f in os.scandir(home_directory) if f.is_dir()]

    #Exclude the ones we don't want
    good_methods = []
    for method in list_subfolders_with_paths:
        if method in exclude_methods:
            if verbose: 
                print("Skipping {} because it was in the 'exclude_methods' kwarg".format(method))
            continue
        good_methods.append(method)    
    
    if len(extra_methods) > 0:
        good_methods = np.append(good_methods,extra_methods)

    n_methods = len(good_methods)

    #Read in the reference spectrum
    cal_spec = np.load("/scr/data/Other_Files/Tinyanont_2018_Fig6/HD109055_set2.npy")[:,:,84:-60]
    wvs = su.rough_wavelength_calibration_v2(cal_spec[0,1], 'J')
    wvs = np.pad(wvs,2,mode='edge')
    
    #Set up the figure
    fig,axes = plt.subplots(2,2,figsize=(12,8))
    max_std = 0
    
    
    #Cycle through the methods and plot the q, u and std.
    for method in good_methods:
        if method in exclude_methods:
            if verbose: 
                print("Skipping {} because it was in the 'exclude_methods' kwarg".format(method))
            continue

        if os.path.isdir(home_directory+method+"/bkg_cutout_update/"):
            tst = home_directory+method+"/bkg_cutout_update/"
            filelist = sorted(glob.glob(tst+"/"+prefix+"*.fits"))
            nfiles = len(filelist)
            if nfiles > 0:
                method += "/bkg_cutout_update/"

        filelist = sorted(glob.glob(home_directory+method+"/"+prefix+"*.fits"))
        nfiles = len(filelist)
        print("Found {} files for method {}".format(nfiles,method))

        time = np.zeros([nfiles])

        BD_all_spec_cube = []
        hwp_ang = np.zeros([nfiles])
        for im in np.arange(nfiles):
            wirc_object = wo.wirc_data(wirc_object_filename=filelist[im],verbose=False)
            wirc_object.source_list[0].rough_lambda_calibration(filter_name="J", 
                                                                method=3, lowcut=0, highcut=-1)
            BD_all_spec_cube.append(wirc_object.source_list[0].trace_spectra)
            hwp_ang[im] = wirc_object.header['HWP_ANG']
            time[im] = wirc_object.header['BJD']
        #Three images at each position
        # if i ==0 :
            # wvs = np.array(BD_all_spec_cube)[0,0,0]
        BD_all_spec_cube = np.array(BD_all_spec_cube)
        BD_all_spec_cube = su.align_spectral_cube(BD_all_spec_cube,ref_trace=cal_spec[0,1])


        q,u,qerr,uerr,qind,uind = so.compute_qu_for_obs_sequence(BD_all_spec_cube,
                                                                           hwp_ang,run_alignment=False,
                                                                           method="double_difference")
        
        #q mean OR median depending on mode. 
        q_mean = np.zeros(q.shape[1])
        u_mean = np.zeros(q.shape[1])
        q_std = np.zeros(q.shape[1])
        u_std = np.zeros(u.shape[1])
        
        #Cycle through the time steps and get the mean or median at each time step
        for i in range(q.shape[1]):
            mn,md,std = stats.sigma_clipped_stats(q[:,i], sigma=3, maxiters=5)
            if mode == 'median':
                q_mean[i] = md
            else:
                q_mean[i] = mn
            q_std[i] = std/np.sqrt(q.shape[1])
            
            
            mn,md,std = stats.sigma_clipped_stats(u[:,i], sigma=3, maxiters=5)
            if mode == 'median':
                u_mean[i] = md
            else:
                u_mean[i] = mn
            u_std[i] = std/np.sqrt(u.shape[1])
            
        axes[0,0].plot(wvs,q_mean,label=method)
        axes[0,1].plot(wvs,u_mean,label=method)
        
        pq = axes[1,0].plot(wvs,q_std,label=method+" sigma-clipped")
        pu = axes[1,1].plot(wvs,u_std,label=method+" sigma-clipped")
        # axes[1,0].plot(wvs,np.nanstd(q,axis=0)/np.sqrt(q.shape[0]),
        #                color=pq[-1].get_color(),linestyle='--',label=method)
        # axes[1,1].plot(wvs,np.nanstd(u,axis=0)/np.sqrt(u.shape[0]),
        #                color=pu[-1].get_color(),linestyle='--',label=method)

        max_qstd = np.nanmax(q_std[(wvs > 1.2) & (wvs < 1.3)])
        if max_qstd > max_std:
            max_std = max_qstd
        max_ustd = np.nanmax(u_std[(wvs > 1.2) & (wvs < 1.3)])
        if max_ustd > max_std:
            max_std = max_ustd
        

        
    axes[0,0].set_xlim(1.15,1.34)
    axes[0,1].set_xlim(1.15,1.34)
    axes[1,0].set_xlim(1.15,1.34)
    axes[1,1].set_xlim(1.15,1.34)
        
    axes[0,0].set_ylim(ylow,yhigh)
    axes[0,1].set_ylim(ylow,yhigh)
    axes[1,0].set_ylim(0,1.2*max_std)
    axes[1,1].set_ylim(0,1.2*max_std)
    
    axes[0,0].set_xlabel(r"Wavelength ($\mu m$)")
    axes[0,1].set_xlabel(r"Wavelength ($\mu m$)")
    axes[1,0].set_xlabel(r"Wavelength ($\mu m$)")
    axes[1,1].set_xlabel(r"Wavelength ($\mu m$)")
    
    axes[0,0].set_ylabel(r"mean q")
    axes[0,1].set_ylabel(r"mean u")
    axes[1,0].set_ylabel(r"std q")
    axes[1,1].set_ylabel(r"std u")
    
    axes[1,0].legend()
    axes[0,0].legend()
    
    fig.suptitle(home_directory,fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if save:
        output_dir = home_directory

        if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        png_filename = output_dir + "/"+target_name+"_bkg_method_comparison.png"
        if verbose:
            print("Saving a summary plot to {}".format(png_filename))
        plt.savefig(png_filename,dpi=200,bbox_inches="tight")

def plot_dataset_polarization_summary(directory,BD_all_spec_cube = None, hwp_ang = None,
                                    save=False,png_filename=None,prefix="image",verbose=True,
                                   target_name = "Unknown",nfiles=None,max_files=-1,binsize=4,
                                     all_theta = None,ylow=-0.02,yhigh=0.02,theta_low=0,theta_high=180,
                                     calibrate=False):
    '''
    Produces a plot that summarizes the extractions and polarization from a whole dataset. 
    '''
    
    fig, axes = plt.subplots(3,2,figsize=(20,24))
    
    if BD_all_spec_cube is None or hwp_ang is None:
          #### Load up the files ####
        filelist = sorted(glob.glob(directory+"/"+prefix+"*.fits"))
        filelist = filelist[:max_files]
        nfiles = len(filelist)

        time = np.zeros([nfiles])

        BD_all_spec_cube = []
        hwp_ang = np.zeros([nfiles])
        for im in np.arange(nfiles):
            wirc_object = wo.wirc_data(wirc_object_filename=filelist[im],verbose=False)
            band = wirc_object.header['AFT']
            if 'J' in band:
                band = 'J'
                wirc_object.source_list[0].rough_lambda_calibration(filter_name="J", 
                                                                    method=3, lowcut=0, highcut=-1)
            else:
                band = 'H'
                wirc_object.source_list[0].rough_lambda_calibration(filter_name="H", 
                                                                    method=2, lowcut=0, highcut=-1)
            BD_all_spec_cube.append(wirc_object.source_list[0].trace_spectra)
            hwp_ang[im] = wirc_object.header['HWP_ANG']
            time[im] = wirc_object.header['BJD']

        BD_all_spec_cube = np.array(BD_all_spec_cube)
        
    #Read in the reference spectrum
    # cal_spec = np.load("/scr/data/Other_Files/Tinyanont_2018_Fig6/HD109055_set2.npy")[:,:,84:-60]
    # wvs = su.rough_wavelength_calibration_v2(cal_spec[0,1], 'J')
    # wvs = np.pad(wvs,2,mode='edge')
    # BD_all_spec_cube = su.align_spectral_cube(BD_all_spec_cube,ref_trace=cal_spec[0,1])

    if band == 'J':
        cal_spec = np.load("/scr/data/Other_Files/Tinyanont_2018_Fig6/HD109055_set2.npy")[:,:,84:-60]
        wvs = su.rough_wavelength_calibration_v2(cal_spec[0,1], 'J')
        wvs = np.pad(wvs,2,mode='edge')
        BD_all_spec_cube = su.align_spectral_cube(BD_all_spec_cube,ref_trace=cal_spec[0,1])

    elif band == "H":
        trace_md_spec = np.median(BD_all_spec_cube,axis=0)
        wvs = su.rough_wavelength_calibration_v2(trace_md_spec[0,1,:], band)
        BD_all_spec_cube = su.align_spectral_cube(BD_all_spec_cube,ref_trace=trace_md_spec[0,1])
    
    q,u,qerr,uerr,qinds,uinds = so.compute_qu_for_obs_sequence(BD_all_spec_cube,
                                                                   hwp_ang,run_alignment=False)        

    spec = np.nanmean(BD_all_spec_cube[:,:,1,:],axis=(0,1))

    n_qs = q.shape[0]
    n_qs_pair1 = q[qinds==0].shape[0]
    n_qs_pair2 = q[qinds==1].shape[0]

    n_us = u.shape[0]
    n_us_pair1 = u[uinds==0].shape[0]
    n_us_pair2 = u[uinds==1].shape[0]

    q_mean = np.zeros([q.shape[1]])
    u_mean = np.zeros([u.shape[1]])
    q_median = np.zeros([q.shape[1]])
    u_median = np.zeros([u.shape[1]])
    q_std = np.zeros([q.shape[1]])
    u_std = np.zeros([u.shape[1]])

    ## NOTE: This error doesn't take into account the sigma clipping (though it would be nice if it did)
    q_mean_err = np.sqrt(np.nansum(qerr**2,axis=0))/n_qs
    u_mean_err = np.sqrt(np.nansum(uerr**2,axis=0))/n_us

    q_mean_err_pair1 = np.sqrt(np.nansum(qerr[qinds==0]**2,axis=0))/n_qs_pair1
    q_mean_err_pair2 = np.sqrt(np.nansum(qerr[qinds==1]**2,axis=0))/n_qs_pair2

    u_mean_err_pair1 = np.sqrt(np.nansum(uerr[uinds==0]**2,axis=0))/n_us_pair1
    u_mean_err_pair2 = np.sqrt(np.nansum(uerr[uinds==1]**2,axis=0))/n_us_pair2

    #### Split things up by trace pair
    q_mean_pair1 = np.zeros([q.shape[1]])
    q_mean_pair2 = np.zeros([q.shape[1]])
    q_median_pair1 = np.zeros([q.shape[1]])
    q_median_pair2 = np.zeros([q.shape[1]])
    q_std_pair1 = np.zeros([q.shape[1]])
    q_std_pair2 = np.zeros([q.shape[1]])

    u_mean_pair1 = np.zeros([q.shape[1]])
    u_mean_pair2 = np.zeros([q.shape[1]])
    u_median_pair1 = np.zeros([q.shape[1]])
    u_median_pair2 = np.zeros([q.shape[1]])
    u_std_pair1 = np.zeros([q.shape[1]])
    u_std_pair2 = np.zeros([q.shape[1]])


    for i in range(q.shape[1]):

        #All q/u
        mn,md,std = stats.sigma_clipped_stats(q[:,i][np.isfinite(q[:,i])], sigma=3, maxiters=5)
        new_array = stats.sigma_clip(q[:,i][np.isfinite(q[:,i])],sigma=3,maxiters=5)
        n = np.size(new_array)
        q_mean[i] = mn
        q_median[i] = md
        q_std[i] = std/np.sqrt(n)


        mn,md,std = stats.sigma_clipped_stats(u[:,i][np.isfinite(u[:,i])], sigma=3, maxiters=5)
        new_array = stats.sigma_clip(u[:,i][np.isfinite(u[:,i])],sigma=3,maxiters=5)
        n = np.size(new_array)
        u_mean[i] = mn
        u_median[i] = md
        u_std[i] = std/np.sqrt(n)

        #q Pair 1
        mn,md,std = stats.sigma_clipped_stats(q[:,i][qinds==0][np.isfinite(q[:,i][qinds==0])], sigma=3, maxiters=5)
        new_array = stats.sigma_clip(q[:,i][qinds==0][np.isfinite(q[:,i][qinds==0])],sigma=3,maxiters=5)
        n = np.size(new_array)
        q_mean_pair1[i] = mn
        q_median_pair1[i] = md
        q_std_pair1[i] = std/np.sqrt(n)
        #q Pair2
        mn,md,std = stats.sigma_clipped_stats(q[:,i][qinds==1][np.isfinite(q[:,i][qinds==1])], sigma=3, maxiters=5)
        new_array = stats.sigma_clip(q[:,i][qinds==1][np.isfinite(q[:,i][qinds==1])],sigma=3,maxiters=5)
        n = np.size(new_array)
        q_mean_pair2[i] = mn
        q_median_pair2[i] = md
        q_std_pair2[i] = std/np.sqrt(n)

        #u Pair1
        mn,md,std = stats.sigma_clipped_stats(u[:,i][uinds==0][np.isfinite(u[:,i][uinds==0])], sigma=3, maxiters=5)
        new_array = stats.sigma_clip(u[:,i][uinds==0][np.isfinite(u[:,i][uinds==0])],sigma=3,maxiters=5)
        n = np.size(new_array)
        u_mean_pair1[i] = mn
        u_median_pair1[i] = md
        u_std_pair1[i] = std/np.sqrt(n)

        #u Pair2
        mn,md,std = stats.sigma_clipped_stats(u[:,i][uinds==1][np.isfinite(u[:,i][uinds==1])], sigma=3, maxiters=5)
        new_array = stats.sigma_clip(u[:,i][uinds==1][np.isfinite(u[:,i][uinds==1])],sigma=3,maxiters=5)
        n = np.size(new_array)
        u_mean_pair2[i] = mn
        u_median_pair2[i] = md
        u_std_pair2[i] = std/np.sqrt(n)

    #### Subplot 1 - All the Data (it is unbinned)
    for i in range(n_qs):
        axes[0,0].plot(wvs,q[i],color='k',alpha=0.05)
        axes[0,1].plot(wvs,u[i],color='k',alpha=0.05)

    axes[0,0].plot(wvs,q_mean,color='k',label="Mean")
    axes[0,1].plot(wvs,u_mean,color='k')

    axes[0,0].plot(wvs,q_median,color='k',linestyle="-.",label="Median")
    axes[0,1].plot(wvs,u_median,color='k',linestyle="-.")

    if band == "J":
        qymax = np.nanpercentile(q[:,(wvs > 1.2) & (wvs < 1.3)],(5,95))
        uymax = np.nanpercentile(u[:,(wvs > 1.2) & (wvs < 1.3)],(5,95))
    else:
        qymax = np.nanpercentile(q[:,(wvs > 1.55) & (wvs < 1.7)],(5,95))
        uymax = np.nanpercentile(u[:,(wvs > 1.55) & (wvs < 1.7)],(5,95))

    axes[0,0].set_ylim(3*qymax[0],3*qymax[1])
    
    
    axes[0,1].set_ylim(3*uymax[0],3*uymax[1])

    axes[0,0].axhline(0,color='crimson',linestyle='-.')
    axes[0,1].axhline(0,color='crimson',linestyle='-.')

    axes[0,0].set_xlabel(r"Wavelength ($\mu m$)",fontsize=16)
    axes[0,1].set_xlabel(r"Wavelength ($\mu m$)",fontsize=16)

    axes[0,0].set_ylabel(r"q",fontsize=16)
    axes[0,1].set_ylabel(r"u",fontsize=16)

    axes[0,0].legend(fontsize=16)

    axes[0,0].set_title(directory,loc="center",fontsize=20)
    axes[1,1].set_title("Target Name = {}".format(target_name),loc="center",fontsize=20) 
    axes[0,1].set_title("{} Individual Files".format(nfiles),loc="center",fontsize=20) 

    if band == "J":
        axes[0,0].set_xlim(1.15,1.34)
        axes[0,1].set_xlim(1.15,1.34)
    else: 
        axes[0,0].set_xlim(1.45,1.85)
        axes[0,1].set_xlim(1.45,1.85)

    ### Now handle binning

    #Do we have to throw away any datasets to bin evenly? 
    snip = q_mean.shape[0] % binsize

    if snip != 0:
        q_mean = np.nanmean(q_mean[:-snip].reshape(-1,binsize),axis=1)
        u_mean = np.nanmean(u_mean[:-snip].reshape(-1,binsize),axis=1)
        q_median = np.nanmedian(q_median[:-snip].reshape(-1,binsize),axis=1)
        u_median = np.nanmedian(u_median[:-snip].reshape(-1,binsize),axis=1)

        wvs_bin = np.mean(wvs[:-snip].reshape(-1,binsize),axis=1)
        p_mean = np.sqrt(q_mean**2+u_mean**2)
        theta_mean = 0.5*np.degrees(np.arctan2(u_mean,q_mean))
        if spec is not None:
            spec_bin = np.mean(spec[:-snip].reshape(-1,binsize),axis=1) 

        q_mean_pair1 = np.mean(q_mean_pair1[:-snip].reshape(-1,binsize),axis=1)
        q_mean_pair2 = np.mean(q_mean_pair2[:-snip].reshape(-1,binsize),axis=1)
        q_median_pair1 = np.median(q_median_pair1[:-snip].reshape(-1,binsize),axis=1)
        q_median_pair2 = np.median(q_median_pair2[:-snip].reshape(-1,binsize),axis=1)
        q_std_pair1 = np.sqrt(np.sum(q_std_pair1[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
        q_std_pair2 = np.sqrt(np.sum(q_std_pair2[:-snip].reshape(-1,binsize)**2,axis=1))/binsize


        u_mean_pair1 = np.mean(u_mean_pair1[:-snip].reshape(-1,binsize),axis=1)
        u_mean_pair2 = np.mean(u_mean_pair2[:-snip].reshape(-1,binsize),axis=1)
        u_median_pair1 = np.mean(u_median_pair1[:-snip].reshape(-1,binsize),axis=1)
        u_median_pair2 = np.mean(u_median_pair2[:-snip].reshape(-1,binsize),axis=1)

        u_std_pair1 = np.sqrt(np.sum(u_std_pair1[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
        u_std_pair2 = np.sqrt(np.sum(u_std_pair2[:-snip].reshape(-1,binsize)**2,axis=1))/binsize

        q_mean_err = np.sqrt(np.sum(q_mean_err[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
        q_mean_err_pair1 = np.sqrt(np.sum(q_mean_err_pair1[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
        q_mean_err_pair2 = np.sqrt(np.sum(q_mean_err_pair2[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
        q_std = np.sqrt(np.sum(q_std[:-snip].reshape(-1,binsize)**2,axis=1))/binsize 
        u_mean_err = np.sqrt(np.sum(u_mean_err[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
        u_mean_err_pair1 = np.sqrt(np.sum(u_mean_err_pair1[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
        u_mean_err_pair2 = np.sqrt(np.sum(u_mean_err_pair2[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
        u_std = np.sqrt(np.sum(u_std[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
        p_mean_err = np.sqrt(q_mean**2*q_mean_err**2+u_mean**2*u_mean_err**2)/p_mean
        theta_mean_err = 0.5*np.degrees( np.sqrt( (u_mean**2*q_mean_err**2+q_mean**2*u_mean_err**2)/(q_mean**2+u_mean**2)**2))

    else: 
        q_mean = np.mean(q_mean.reshape(-1,binsize),axis=1)
        u_mean = np.mean(u_mean.reshape(-1,binsize),axis=1)
        wvs_bin = np.mean(wvs.reshape(-1,binsize),axis=1)
        p_mean = np.sqrt(q_mean**2+u_mean**2)
        theta_mean = 0.5*np.degrees(np.arctan2(u_mean,q_mean))
        # theta_bin[theta_bin < 0] +=180
        if spec is not None:
            spec_bin = np.mean(spec.reshape(-1,binsize),axis=1)

        q_mean_err = np.sqrt(np.sum(q_mean_err.reshape(-1,binsize)**2,axis=1))/binsize
        q_mean_err_pair1 = np.sqrt(np.sum(q_mean_err_pair1.reshape(-1,binsize)**2,axis=1))/binsize
        q_mean_err_pair2 = np.sqrt(np.sum(q_mean_err_pair1.reshape(-1,binsize)**2,axis=1))/binsize
        q_std = np.sqrt(np.sum(q_std.reshape(-1,binsize)**2,axis=1))/binsize
        u_mean_err = np.sqrt(np.sum(u_mean_err.reshape(-1,binsize)**2,axis=1))/binsize
        u_mean_err_pair1 = np.sqrt(np.sum(u_mean_err_pair1.reshape(-1,binsize)**2,axis=1))/binsize
        u_mean_err_pair2 = np.sqrt(np.sum(u_mean_err_pair2.reshape(-1,binsize)**2,axis=1))/binsize
        u_std = np.sqrt(np.sum(u_std.reshape(-1,binsize)**2,axis=1))/binsize
        p_mean_err = np.sqrt(q_mean**2*q_mean_err**2+u_mean**2*u_mean_err**2)/p_mean
        theta_mean_err = 0.5*np.degrees( np.sqrt( (u_mean**2*q_mean_err**2+q_mean**2*u_mean_err**2)/(q_mean**2+u_mean**2)**2))

    ## Now let's calibrate!
    if calibrate: 
        q_median_pair1, u_median_pair1, q_std_pair1,u_std_pair1 = calibration.calibrate_qu(wvs_bin,q_median_pair1,u_median_pair1,q_std_pair1,u_std_pair1,trace_pair=0)
        q_median_pair2, u_median_pair2, q_std_pair2,u_std_pair2 = calibration.calibrate_qu(wvs_bin,q_median_pair2,u_median_pair2,q_std_pair2,u_std_pair2,trace_pair=1)
    
    q_median = 0.5*(q_median_pair1+q_median_pair2)
    q_std = np.sqrt(q_std_pair1**2+q_std_pair2**2)/np.sqrt(2)
    u_median = 0.5*(u_median_pair1+u_median_pair2)
    u_std = np.sqrt(u_std_pair1**2+u_std_pair2**2)/np.sqrt(2)

    p_std = np.sqrt(q_mean**2*q_std**2+u_mean**2*u_std**2)/p_mean

    p_mean = np.sqrt(q_mean**2+u_mean**2)
    theta_mean = 0.5*np.degrees(np.arctan2(u_mean,q_mean))
    theta_mean = theta_mean %180
    theta_mean[theta_mean < 0] += 180

    p_median = np.sqrt(q_median**2+u_median**2)

    where_3sigma = np.where(p_mean > 3*p_mean_err)


    #### Subplot 2 - All the Data

    #q - pair 1
    axes[1,0].plot(wvs_bin,q_median_pair1,color='tab:blue',linewidth=2,label="Median Pair 1")
    axes[1,0].fill_between(wvs_bin,q_median_pair1+q_std_pair1,q_median_pair1-q_std_pair1,color='tab:blue',alpha=0.1)
    axes[1,0].plot(wvs_bin,q_median_pair1+q_mean_err_pair1,color='tab:blue',linestyle='--',label="$\sigma_{photometric}$ Pair 1",alpha=0.5)
    axes[1,0].plot(wvs_bin,q_median_pair1-q_mean_err_pair1,color='tab:blue',linestyle='--',alpha=0.5)

    #q - pair 2
    axes[1,0].plot(wvs_bin,q_median_pair2,color='tab:orange',linewidth=2,label="Median Pair 2")
    axes[1,0].fill_between(wvs_bin,q_median_pair2+q_std_pair2,q_median_pair2-q_std_pair2,color='tab:orange',alpha=0.1)
    axes[1,0].plot(wvs_bin,q_median_pair2+q_mean_err_pair2,color='tab:orange',linestyle='--',label="$\sigma_{photometric}$ Pair 2")
    axes[1,0].plot(wvs_bin,q_median_pair2-q_mean_err_pair2,color='tab:orange',linestyle='--')

    #u - pair 1
    axes[1,1].fill_between(wvs_bin,u_median_pair1+u_std_pair1,u_median_pair1-u_std_pair1,color='tab:blue',alpha=0.1,label="$\sigma_{std}$ Pair 1")
    axes[1,1].plot(wvs_bin,u_median_pair1+u_mean_err_pair1,color='tab:blue',linestyle='--')
    axes[1,1].plot(wvs_bin,u_median_pair1-u_mean_err_pair1,color='tab:blue',linestyle='--')
    axes[1,1].plot(wvs_bin,u_median_pair1,color='tab:blue',linewidth=2)

    #u - pair 2
    axes[1,1].plot(wvs_bin,u_median_pair2+u_mean_err_pair2,color='tab:orange',linestyle='--')
    axes[1,1].plot(wvs_bin,u_median_pair2-u_mean_err_pair2,color='tab:orange',linestyle='--')
    axes[1,1].fill_between(wvs_bin,u_median_pair2+u_std_pair2,u_median_pair2-u_std_pair2,color='tab:orange',alpha=0.1,label="$\sigma_{std}$ Pair 2")
    axes[1,1].plot(wvs_bin,u_median_pair2,color='tab:orange',linewidth=2)        

    #q combined pair
    axes[1,0].fill_between(wvs_bin,q_median+q_std,q_median-q_std,color='rebeccapurple',alpha=0.2)    
    axes[1,0].plot(wvs_bin,q_median,color='rebeccapurple',linewidth=2,label="Median Combined Traces")
    axes[1,0].plot(wvs_bin,q_median+q_mean_err,color='rebeccapurple',linestyle="--",label="$\sigma_{photometric}$")
    axes[1,0].plot(wvs_bin,q_median-q_mean_err,color='rebeccapurple',linestyle="--")
    # axes[1,0].plot(wvs_bin,q_median,color='rebeccapurple',linewidth=2,label="Median Combined Traces")

    #u combined pair
    axes[1,1].fill_between(wvs_bin,u_median+u_std,u_median-u_std,color='rebeccapurple',alpha=0.2,label="Combined")    
    axes[1,1].plot(wvs_bin,u_median,color='rebeccapurple',linewidth=2)
    axes[1,1].plot(wvs_bin,u_median+u_mean_err,color='rebeccapurple',linestyle="--")
    axes[1,1].plot(wvs_bin,u_median-u_mean_err,color='rebeccapurple',linestyle="--")


    axes[1,0].set_ylim(ylow,yhigh)
    axes[1,1].set_ylim(ylow,yhigh)

    if band == "J":
        axes[1,0].set_xlim(1.15,1.34)
        axes[1,1].set_xlim(1.15,1.34)
    else: 
        axes[1,0].set_xlim(1.45,1.85)
        axes[1,1].set_xlim(1.45,1.85)

    axes[1,0].axhline(0,color='crimson',linestyle='-.')
    axes[1,1].axhline(0,color='crimson',linestyle='-.')

    axes[1,0].set_xlabel(r"Wavelength ($\mu m$)",fontsize=16)
    axes[1,1].set_xlabel(r"Wavelength ($\mu m$)",fontsize=16)

    axes[1,0].set_ylabel(r"q",fontsize=16)
    axes[1,1].set_ylabel(r"u",fontsize=16)

    axes[1,0].legend(fontsize=16)
    axes[1,1].legend(fontsize=16)

    axes[1,0].set_title("Split by Trace Pairs",loc="center",fontsize=20) 


    ### Plot p

    axes[2,0].fill_between(wvs_bin,p_mean-p_mean_err,p_mean+p_mean_err,color='rebeccapurple',alpha=0.2,label=r"$\sigma_{photometric}$")
    axes[2,0].plot(wvs_bin,p_mean,color='rebeccapurple',label= "Mean")

    axes[2,0].set_xlim(1.15,1.34)
    if band == "J":
        p_percentile = np.nanpercentile(p_mean[(wvs_bin > 1.2) & (wvs_bin < 1.3)],(5,95))
    else:
        p_percentile = np.nanpercentile(p_mean[(wvs_bin > 1.55) & (wvs_bin < 1.7)],(5,95))
    axes[2,0].set_ylim(0,p_percentile[1]*2)

    axes[2,0].plot(wvs_bin,3*p_mean_err,color='r',label="3$\sigma_{photometric}$")
    axes[2,0].plot(wvs_bin,3*p_std,color='r',linestyle="--",label="3$\sigma_{std}$")

    axes[2,0].legend(fontsize=16)

    ### Plot theta

    axes[2,1].errorbar(wvs_bin,theta_mean,
                   yerr=theta_mean_err,linestyle="",
                  marker='o',color='k',label="All theta",alpha=0.3)


    axes[2,1].errorbar(wvs_bin[where_3sigma],theta_mean[where_3sigma],
                       yerr=theta_mean_err[where_3sigma],linestyle="",
                      marker='o',color='k',label=r"p > 3$\sigma_{photometric}$")


    ax20_2 = axes[2,0].twinx()
    ax20_2.plot(wvs_bin,spec_bin,label="Stokes I Spectrum")

    ax20_2.tick_params(axis='y', labelcolor='C0')
    ax20_2.set_ylabel("Stokes I Spectrum",fontsize=16,color="C0")

    axes[2,0].set_title("Total Time = {:.2f}h".format((time[-1]-time[0])*24),fontsize=20)

    axes[2,0].set_xlabel(r"Wavelength ($\mu m$)",fontsize=16)
    axes[2,1].set_xlabel(r"Wavelength ($\mu m$)",fontsize=16)

    axes[2,0].set_ylabel(r"p",fontsize=16)
    axes[2,1].set_ylabel(r"$\theta$",fontsize=16)

    axes[2,1].legend(fontsize=16)
    axes[2,1].set_ylim(theta_low,theta_high)


    if band == "J":
        axes[2,0].set_xlim(1.15,1.34)
        axes[2,1].set_xlim(1.15,1.34)
    else: 
        axes[2,0].set_xlim(1.45,1.85)
        axes[2,1].set_xlim(1.45,1.85)

                # axes[0,0].set_ylim(3*qymax[0],3*qymax[1])

    if band =="J":
        axes[0,0].fill_between([0,1.17],3*qymax[0],3*qymax[1],hatch='x',color='k',alpha=0.05,zorder=10)
        axes[0,1].fill_between([0,1.17],3*uymax[0],3*uymax[1],hatch='x',color='k',alpha=0.05,zorder=10)
        axes[1,0].fill_between([0,1.17],ylow,yhigh,hatch='x',color='k',alpha=0.05,zorder=10)
        axes[1,1].fill_between([0,1.17],ylow,yhigh,hatch='x',color='k',alpha=0.05,zorder=10)
        axes[2,0].fill_between([0,1.17],0,p_percentile[1]*2,hatch='x',color='k',alpha=0.05)
        axes[2,1].fill_between([0,1.17],theta_low,theta_high,hatch='x',color='k',alpha=0.05)

        axes[0,0].fill_between([1.32,1.34],3*qymax[0],3*qymax[1],hatch='x',color='k',alpha=0.05,zorder=10)
        axes[0,1].fill_between([1.32,1.34],3*uymax[0],3*uymax[1],hatch='x',color='k',alpha=0.05,zorder=10)
        axes[1,0].fill_between([1.32,1.34],ylow,yhigh,hatch='x',color='k',alpha=0.05,zorder=10)
        axes[1,1].fill_between([1.32,1.34],ylow,yhigh,hatch='x',color='k',alpha=0.05,zorder=10)
        axes[2,0].fill_between([1.32,1.34],0,p_percentile[1]*2,hatch='x',color='k',alpha=0.05)
        axes[2,1].fill_between([1.32,1.34],theta_low,theta_high,hatch='x',color='k',alpha=0.05)

    if band =="H":
        axes[0,0].fill_between([0,1.49],3*qymax[0],3*qymax[1],hatch='x',color='k',alpha=0.05,zorder=10)
        axes[0,1].fill_between([0,1.49],3*uymax[0],3*uymax[1],hatch='x',color='k',alpha=0.05,zorder=10)
        axes[1,0].fill_between([0,1.49],ylow,yhigh,hatch='x',color='k',alpha=0.05,zorder=10)
        axes[1,1].fill_between([0,1.49],ylow,yhigh,hatch='x',color='k',alpha=0.05,zorder=10)
        axes[2,0].fill_between([0,1.49],0,p_percentile[1]*2,hatch='x',color='k',alpha=0.05)
        axes[2,1].fill_between([0,1.49],theta_low,theta_high,hatch='x',color='k',alpha=0.05)

        axes[0,0].fill_between([1.78,1.85],3*qymax[0],3*qymax[1],hatch='x',color='k',alpha=0.05,zorder=10)
        axes[0,1].fill_between([1.78,1.85],3*uymax[0],3*uymax[1],hatch='x',color='k',alpha=0.05,zorder=10)
        axes[1,0].fill_between([1.78,1.85],ylow,yhigh,hatch='x',color='k',alpha=0.05,zorder=10)
        axes[1,1].fill_between([1.78,1.85],ylow,yhigh,hatch='x',color='k',alpha=0.05,zorder=10)
        axes[2,0].fill_between([1.78,1.85],0,p_percentile[1]*2,hatch='x',color='k',alpha=0.05)
        axes[2,1].fill_between([1.78,1.85],theta_low,theta_high,hatch='x',color='k',alpha=0.05)

    plt.tight_layout()

    if save:

        output_dir = directory

        if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        png_filename = output_dir + "/"+target_name+"_polarization_summary.png"
        if verbose:
            print("Saving a summary plot to {}".format(png_filename))
        plt.savefig(png_filename,dpi=200,bbox_inches="tight")