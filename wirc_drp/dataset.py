import numpy as np
import glob
import warnings
import os
import wirc_drp.wirc_object as wo
from wirc_drp.utils import calibration, spec_utils as su, image_utils as iu
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

def reduce_dataset(filelist, source_pos, bkg_fnames = None, output_path = "./",verbose=True, 
bkg_methods = ["shift_and_subtract","PCA","median_ref","scaled_bkg","simple_median",
"slit_background","cutout_median"],n_pca=[1,3,5,10,15,20,40], in_slit=False,less_verbose=True,
parallel=False,n_processes=None,nclosest=None,same_HWP=True,sub_bar=False):
    '''
    A function that reduces a dataset given a list of calibrated science and background files

    It will first extract spectra using all of the bkg_methods then calculate the polarization of each one. 


    Inputs:
        filelist    -   A python list of filepaths
        source_pos  -   The source position in detector [x,y]
        bkg_fnames -   A python list of background files (this can be one file)
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
                        outdir,"",verbose,bkg_method,npca,False,False,nclosest,same_HWP,sub_bar) for fname in filelist]
                        outputs = pool.map(extract_single_file_parallel_helper,args) 

                        if not in_slit:
                        #Now with the update cutout backgrounds
                            args = [(fname,source_pos,bkg_fnames,
                            outdir2,"",verbose,bkg_method,npca,True,False,nclosest,same_HWP,sub_bar) for fname in filelist]
                            outputs = pool.map(extract_single_file_parallel_helper,args) 
                #Or not parallel
                else: 
                
                    for i,fname in enumerate(filelist):
                        if verbose or less_verbose:
                            print("File {} of {} with bkg_method = {}{}: {}".format(i+1,len(filelist),bkg_method,npca,fname))
                        
                        extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir,verbose=verbose,bkg_method=bkg_method,
                        num_PCA_modes=npca,sub_bar=sub_bar,nclosest=nclosest,same_HWP=same_HWP)

                        if not in_slit:
                            extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir2,verbose=verbose,bkg_method=bkg_method,
                            num_PCA_modes=npca,update_cutout_backgrounds=True,sub_bar=sub_bar,nclosest=nclosest,same_HWP=same_HWP)
        
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
                    outdir,"",verbose,bkg_method,None,False,False,nclosest,same_HWP,sub_bar) for fname in filelist]
                    outputs = pool.map(extract_single_file_parallel_helper,args) 

                    if bkg_method != "cutout_median" and not in_slit:
                        #This is to update cutout backgrounds, don't have to do if method is already "cutout_median"
                        #Now with the update cutout backgrounds
                        args = [(fname,source_pos,bkg_fnames,
                        outdir2,"",verbose,bkg_method,None,True,False,nclosest,same_HWP,sub_bar) for fname in filelist]
                        outputs = pool.map(extract_single_file_parallel_helper,args) 

            else: #not parallel

                for i,fname in enumerate(filelist):
                    if verbose or less_verbose:
                        print("File {} of {} with bkg_method = {}: {}".format(i+1,len(filelist),bkg_method,fname))
                    extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir,verbose=verbose,
                    bkg_method=bkg_method,nclosest=nclosest,sub_bar=sub_bar,same_HWP=same_HWP)
                    if bkg_method != "cutout_median" and not in_slit:
                        #This is to update cutout backgrounds, don't have to do if method is already "cutout_median"
                        extract_single_file(fname,source_pos, bkg_fnames,output_path=outdir2,verbose=verbose,bkg_method=bkg_method,
                        update_cutout_backgrounds=True,nclosest=nclosest,sub_bar=sub_bar,same_HWP=same_HWP)
        # except:
            # print("Couldn't finish reduction on file {} for bkg_method {}".format(fname,bkg_method))
            # pass
            
def extract_single_file(filename,source_pos, bkg_fnames,output_path = "./",output_suffix="",verbose=True,
bkg_method=None,num_PCA_modes=None,update_cutout_backgrounds=False,save_full_image = False,sub_bar = False,
nclosest=None,same_HWP=True):
    '''
    Opens a file, generates a background image, extracts the source spectra and then saves them to the output path

    Inputs:
    filename    -   The filename of a calibrated wirc object file
    source_pos  -   The source position in [x,y] format
    '''

    try:
        tmp_data = wo.wirc_data(wirc_object_filename=filename,verbose=verbose)
        tmp_data.source_list = []
        tmp_data.n_sources = 0

        if bkg_method is not None and bkg_method != "cutout_median":
            tmp_data.generate_bkg(method=bkg_method,verbose=verbose,
                                bkg_by_quadrants=True,
                                bkg_fns=bkg_fnames,num_PCA_modes=num_PCA_modes,
                                nclosest=nclosest,
                                same_HWP=same_HWP)

        tmp_data.add_source(source_pos[0],source_pos[1],update_w_chi2_shift=False,sub_bkg=True)

        wp_source = tmp_data.source_list[0]
        wp_source.get_cutouts(tmp_data.full_image,tmp_data.DQ_image,'J',
                                replace_bad_pixels=True,method='interpolate',
                                bkg_image = tmp_data.bkg_image,sub_bar=sub_bar)
        
        if bkg_method == "cutout_median":
            wp_source.generate_cutout_backgrounds(update=False)
        if update_cutout_backgrounds:
            wp_source.generate_cutout_backgrounds(update=True)

        wp_source.extract_spectra(verbose=verbose,
                                plot_findTrace=False,plot_optimal_extraction=False,
                                spatial_sigma=3,diag_mask=True,
                                diag_mask_width = 50)

        tmp_data.source_list.append(wp_source)
        tmp_data.n_sources += 1

        output_fname = output_path+filename.rsplit(".fits")[0].split("/")[-1]+output_suffix+".fits"
        tmp_data.save_wirc_object(output_fname,save_full_image = save_full_image)
    except Exception as e:
        print("Some error with file {}".format(filename))
        print(str(e))

def extract_single_file_parallel_helper(args):
    '''
    This is here to help the parallelization of extract_single_file by unpacking the arguments
    '''
    try: 
        extract_single_file(args[0],args[1],args[2],output_path = args[3],output_suffix = args[4],
        verbose = args[5],bkg_method = args[6],num_PCA_modes = args[7],
        update_cutout_backgrounds = args[8], save_full_image=args[9],nclosest=args[10],same_HWP=args[11],
        sub_bar=args[12])
    except Exception as e:
        print("Some error with file {}".format(args[0]))
        print(e)

def reduce_ABAB_dataset(filelist, source_posA, source_posB,output_path = "./",verbose=False, less_verbose=True,
bkg_methods = ["shift_and_subtract","PCA","median_ref","scaled_bkg","simple_median",
"slit_background","cutout_median"],n_pca=[1,3,5,10,15,20,40], in_slit=False,parallel=False,
n_processes=None,n_per_group=4,nclosest=None,same_HWP=False,sub_bar=False,groupA=None,
groupB=None):
    '''
    A function that reduces a dataset given a list of calibrated science files, assuming you observed in an ABAB dither pattern.
    It uses each position as backgrond for the other. 
    
    It assumes by default that you have 4 images per dither position, but you can change this with the
    n_per_group keyword, or you can manually input groupA and groupB indices. 

    Inputs:
        filelist    -   A python list of filepaths
        background_list -   A python list of background files (this can be one file)
        groupA      - the indices of the files in position A
        groupB      - the indicies of the files in position B. Both groupA and groupB must be set. 
    '''

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
                            nclosest=nclosest,same_HWP=same_HWP,sub_bar=sub_bar)

    #Reduce them all with all the possible background subtraction methods - Dither position 2
    reduce_dataset(filelist[groupB], source_posB, bkg_fnames = filelist[groupA], 
                            bkg_methods = bkg_methods,
                            output_path = output_path,
                            verbose=verbose,
                            less_verbose=less_verbose,
                            n_pca=n_pca, in_slit=in_slit,
                            parallel=parallel,n_processes=n_processes,
                            nclosest=nclosest,same_HWP=same_HWP,sub_bar=sub_bar)

def reduce_dataset_distance(filelist, source_pos, output_path = "./",verbose=False, less_verbose=True,
bkg_methods = ["shift_and_subtract","PCA","median_ref","scaled_bkg","simple_median",
"slit_background","cutout_median"],n_pca=[1,3,5,10,15,20,40], in_slit=False,delta_ra=10,
parallel=False,n_processes=None,parallel_finding=True):
    '''
    A version of reduce_dataset that has an ra distance cutoff to use in determining what background to use. 

    This one works best when you have a bright-ish source that can be automatically found by the source finder. 
    '''

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
                
                do_parallel_extraction(source_pos,filelist,ras,delta_ra,outdir,verbose,bkg_method,npca,False,False,n_processes)
                
                if bkg_method != "cutout_median":
                    do_parallel_extraction(source_pos,filelist,ras,delta_ra,outdir2,verbose,bkg_method,npca,True,False,n_processes)

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
                        extract_single_file(fname,source_pos, filelist[good_bkgs],output_path=outdir,verbose=verbose,bkg_method=bkg_method)
                    except: 
                        print("Something went wrong with file {}".format(fname))
                    if bkg_method != "cutout_median":
                        try: 
                            extract_single_file(fname,source_pos, filelist[good_bkgs],output_path=outdir2,verbose=verbose,bkg_method=bkg_method,
                            update_cutout_backgrounds=True)
                        except:
                            print("Something went wrong with bkg_cutouts with file {}".format(fname))

def do_parallel_extraction(source_pos,filelist,ras,delta_ra,outdir,verbose,bkg_method,npca,update_bkg_cutouts,save,n_processes):
    '''
    Another helping function
    '''
    import multiprocessing as mp
    #Make the pool
    with mp.Pool(processes=n_processes) as pool:
        # pool = mp.Pool(processes=n_processes)
        
        #Package up the arguments (getting the background list looks complicated, 
        # but is really just a compact version of the non-parallel version): 
        args = [(fname,source_pos,filelist[np.where(np.abs(ras-ras[i]) > delta_ra)],
        outdir,"",verbose,bkg_method,npca,update_bkg_cutouts,save) for i,fname in enumerate(filelist)]
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



