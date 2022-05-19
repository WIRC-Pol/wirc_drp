import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib
import warnings

import wirc_drp.utils.image_utils as image_utils
import wirc_drp.utils.spec_utils as spec_utils
import wirc_drp.utils.source_utils as source_utils
import wirc_drp.utils.calibration as calibration
from wirc_drp import constants
from wirc_drp import version # For versioning (requires gitpython 'pip install gitpython')
from wirc_drp.masks import * ### Make sure that the wircpol/DRP/mask_design directory is in your Python Path!
import wirc_drp
from astropy import time as ap_time, coordinates as coord, units as u
from astropy.stats import sigma_clipped_stats

import pdb
import copy 

mask_path = (wirc_drp.__path__)[0]+'/masks/'
non_linearity_path = (wirc_drp.__path__)[0]+'/specification/'

class wirc_data(object):
    """
    A wirc data file, that may include reduced data products

    Args:
        raw_filename: The filename of the raw image [optional]
        wirc_object_filename: The filename

    Attributes:
        full_image: Array of shape (y,x) [probably 2048 x 2048]
        header: An astropy fits header corresponding to the raw data file [including history of data reduction]

        dark_fn: the filename for the dark file used to correct this image
        flat_fn: the filename for the flat file used to correct this image
        bkg_fn: the filename for the bkg file subtracted off this image
        bp_fn: the filename of the bad pixel map

        filelist: If this file is the result of multiple files being combined, this is a string list of all the filenames of length n_files
        dxdy_list: If this file is the result of multiple files being combined, this is an array of [nfiles,dx,dy] that hold the shifts applied to each image before combining

        n_sources: the number of sources in the full_image
        source_list: A list of n_sources wircpol_source objects
        bjd: mid-exposure time in BJD_TDB

    """

    def __init__(self, raw_filename=None, wirc_object_filename=None, load_full_image = True, 
        dark_fn = None, flat_fn = None, bp_fn = None, hp_fn = None, bkg_fn = None, ref_lib = None, 
        cross_correlation_template=None, trace_template=None, verbose = True,clear_sources=False,
        update_bjd=True):
        ## set verbose=False to suppress print outputs
        ## Load in either the raw file, or the wircpol_object file,
        ## If load_full_image is True, load the full array image. This uses a lot of memory if a lot of wric objects are loaded at once.
        ##

        if raw_filename is not None and wirc_object_filename is not None:
            print("Can't open both a raw file and wircpol_object, ignoring the raw file and loading the wirc_object_file ")
            print("Loading a wircpol_data object from file {}".format(wirc_object_filename))
            self.load_wirc_object(wirc_object_filename)

        elif wirc_object_filename is not None:
            if verbose:
                print("Loading a wirc_data object from file {}".format(wirc_object_filename))
            self.load_wirc_object(wirc_object_filename, load_full_image = load_full_image,clear_sources=clear_sources,
                                verbose=verbose)
            

        elif raw_filename is not None:

            if verbose:
                print("Creating a new wirc_data object from file {}".format(raw_filename))
            self.raw_filename = raw_filename

            with fits.open(raw_filename) as hdu:
                self.full_image = hdu[0].data
                self.header = hdu[0].header

            self.header['RAW_FN'] = raw_filename
            self.header["FN"] = raw_filename

            self.filelist = [raw_filename]
            self.dxdy_list = [[0.,0.]]

            #### Get the Filter Info ####
            aft_filter = self.header['AFT']
            filter_name = aft_filter[0]
            if filter_name != 'J' and filter_name != 'H':
                print("The pipeline was expecting either a J- or H-band filter but found {} instead".format(aft_filter))
                print("Returning.\n")
                # break
            else:
                if verbose:
                    print("Found a {}-band filter in the header of file {}".format(filter_name,raw_filename))
            self.filter_name = filter_name

            self.calibrated = False
            self.bkg_subbed = False
            self.already_masked = False

            self.n_sources = 0 
            self.source_list = []
            self.source_positions = []
            self.dark_fn = dark_fn
            self.flat_fn = flat_fn
            self.bkg_fn = bkg_fn
            self.bp_fn = bp_fn
            self.hp_fn = hp_fn
            self.ref_lib = ref_lib
            self.cross_correlation_template = cross_correlation_template
            self.trace_template = trace_template
            self.trace_width = None

            self.trace_fluxes = []

            #A bad flag that indicates this whole file should be disregarded.
            self.bad_flag = False
            self.bad_reason = ""

            #The data quality image
            self.DQ_image = None

            #Background image   
            self.bkg_image = None   

            #Cutout images
            self.UL_cutout = None
            self.LR_cutout = None
            self.UR_cutout = None
            self.LL_cutout = None

            self.UL_bkg = None
            self.LR_bkg = None
            self.UR_bkg = None
            self.LL_bkg = None

            #TODO Determine whether the type is spec or pol
            # self.type =

            #get mid-exposure time in BJD_TDB
            if update_bjd:
                try:
                    date_in=self.header['UTSHUT']
                    target_pos=coord.SkyCoord(self.header['RA'],self.header['DEC'],unit=(u.hourangle,u.deg),frame='icrs')
                    palomar=coord.EarthLocation.of_site('Palomar')
                    time=ap_time.Time(date_in,format='isot',scale='utc',location=palomar)
                    mid_exptime=0.5*self.header['EXPTIME']*self.header['COADDS']/(24*3600) #in units of days
                    ltt_bary=time.light_travel_time(target_pos)
                    time=time.tdb+ltt_bary #convert from UTC to TDB standard, apply barycentric correction
                    self.bjd=time.jd+mid_exptime
                except Exception as e:
                    self.bjd = 0.
                    print("Couldn't update the BJD. Error {}".format(e))
            else: 
                self.bjd = 0.

        else: #for a blank wirc object
            self.calibrated = False
            self.bkg_subbed = False
            self.n_sources = 0
            self.source_list = []
            self.source_positions = []

        self.cross_correlation_template = None
    
    def calibrate(self, clean_bad_pix=True, replace_nans=True, mask_bad_pixels=False, stable_bad_pix_map = 'default', get_bad_pix_from_master_dark = True,
                  destripe_raw = False, destripe=False, verbose=False,  report_median = False,
                  report_bkg_multiplier = False, median_subtract = False, bkg_by_quadrants=False, correct_nonlinearity = False, nonlinearity_array = None, multicomponent_frame = None):
        '''
        Apply dark and flat-field correction

        Dealing with bad pixels
            (1) Use the given hot and bad pixel files created from dark and flat respectively (self.hp_fn and self.bp_fn)
            (2) Use the stable bad pixel map stored in ./hot_pixel_map/bad_pix_list.npy (stable_bad_pix_map == 'default' parameter; in the future take arbitrary file)
            (3) For long exposure, also use pixels with value = 0 in the master dark (get_bad_pix_from_master_dark parameter)
    
        '''
        #TODO Add checks to make sure the flatnames are not none

        if not self.calibrated:
            if correct_nonlinearity:
                n_coadds = self.header["COADDS"]
                if nonlinearity_array is None:
                    #if want to correct non linearity but an array is not provided, use a default file stored in the specification directory.
                    nonlinearity_fname = non_linearity_path+'NL_polyfit_maps-beta.fits'
                    print("Use default non linearity map: "+nonlinearity_fname)
                    with fits.open(nonlinearity_fname) as hdu:
                        nonlinearity_array = hdu[1].data
                #Rn the correction
                self.full_image = calibration.correct_nonlinearity(
                                      self.full_image, n_coadds,
                                      nonlinearity_array)

            if self.dark_fn is not None:
                #Open the master dark
                master_dark_hdu = fits.open(self.dark_fn)
                master_dark = master_dark_hdu[0].data
                dark_shape = np.shape(master_dark)
                if verbose:
                    print(("Subtracting {} from the image".format(self.dark_fn)))
                dark_exp_time = master_dark_hdu[0].header['EXPTIME'] * master_dark_hdu[0].header['COADDS']
                total_exp_time = self.header["EXPTIME"]*self.header["COADDS"]
                #Checking Dark Exposure times and scaling if need be
                if dark_exp_time != total_exp_time:
                    if verbose:
                        print("The master dark file doesn't have the same exposure time as the image being calibrated. We'll scale the dark for now, but this isn't ideal")
                    factor = total_exp_time/dark_exp_time
                else:
                    factor = 1.

                #For long exposure dark and if get_bad_pix_from_master_dark, create a map of pixels with 0
                if master_dark_hdu[0].header['EXPTIME'] >= 30 and get_bad_pix_from_master_dark == True:
                    long_exp_mask = master_dark == 0

                #Subtract the dark
                self.full_image = self.full_image-factor*master_dark

                #Update the header
                self.header['HISTORY'] = "Subtracting {} from each flat file".format(self.dark_fn)
                self.header['DARK_FN'] = self.dark_fn

            else:
                print("No dark filename found, continuing without subtracting a dark")

            if self.flat_fn is not None:
                #Open the master flat
                master_flat_hdu = fits.open(self.flat_fn)
                master_flat = master_flat_hdu[0].data
                if verbose:
                    print(("Dividing the image by {}".format(self.flat_fn)))

                #Divide the flat
                self.full_image = self.full_image/master_flat

                #Update the header
                self.header['HISTORY'] = "Dividing each file by {}".format(self.flat_fn)
                self.header["FLAT_FN"] = self.flat_fn

            else:
                print("No flat filename found, continuing without divinding by a falt")

            if report_median:
                mean, med, std = sigma_clipped_stats(self.full_image.flatten())

            if destripe_raw:
                if verbose:
                    print("Destriping the detector image")
                self.full_image = calibration.destripe_raw_image(self.full_image)


            #If a bad pixel map is provided then correct for bad pixels, taking into account the clean_bad_pix and mask_mad_pixels flags
            bad_pixel_map_bool = np.zeros(self.full_image.shape).astype('bool')
            hot_pixel_map_bool = np.zeros(self.full_image.shape).astype('bool') #start with zeros, if provided then change the values
            if self.bp_fn is not None:
                #Open the bad pixel map
                bp_map_hdu = fits.open(self.bp_fn)
                bad_pixel_map = (bp_map_hdu[0].data).astype('bool')
                bad_pixel_map_bool = np.logical_or(bad_pixel_map_bool , bad_pixel_map)
                if verbose:
                    print(("Using bad pixel map {}".format(self.bp_fn)))

            #If stable bad pixel map is requested, find the file and include it
            if stable_bad_pix_map == 'default':
                try:
                    #read the file
                    stable_bad_pix_list = np.load('bad_pix_mask/bad_pix_list.npy')
                    #make a list with lenght = number of all pixels on H2 array
                    recon_mask = np.zeros(2048*2048) ###ASSUMING 2k * 2k array
                    #set pixels with index listed in the provided mask to 1
                    recon_mask[[stable_bad_pix_list]] = 1
                    #reshape it back to 2D
                    stable_bad_pix = recon_mask.reshape((2048,2048))
                    #logical_or it to the current map
                    bad_pixel_map_bool = np.logical_or(bad_pixel_map_bool, stable_bad_pix)
                except:
                    stable_bad_pix_map = 'failed'

            #This is for long exposure frames
            if master_dark_hdu[0].header['EXPTIME'] >= 30 and get_bad_pix_from_master_dark == True:
                bad_pixel_map_bool = np.logical_or(bad_pixel_map_bool, long_exp_mask) 

            #if hot pixel map is also given
            if self.hp_fn is not None:
                hot_pixel_map = fits.open(self.hp_fn)[0].data
                bad_pixel_map_bool = np.logical_or(bad_pixel_map_bool, hot_pixel_map.astype(bool)) #combine two maps
            else:
                hot_pixel_map = hot_pixel_map_bool
            if clean_bad_pix:
                redux = calibration.cleanBadPix(self.full_image, bad_pixel_map_bool)
                if self.hp_fn is not None:
                    self.header['HISTORY'] = "Cleaned all bad/hot pixels found in {}, {} using a median filter".format(self.bp_fn, self.hp_fn)
                else:
                    self.header['HISTORY'] = "Cleaned all bad pixels found in {} using a median filter".format(self.bp_fn)
                if stable_bad_pix_map == 'default':
                    self.header['HISTORY'] = "Cleaned all stable bad pixels in bad_pix_mask/bad_pix_list.npy using a median filter"
                if master_dark_hdu[0].header['EXPTIME'] >= 30 and get_bad_pix_from_master_dark == True:
                    self.header['HISTORY'] = "Cleaned all pixels with value = 0 in long exposure dark using a median filter"
                self.header['CLEAN_BP'] = "True"
            else:
                redux = self.full_image
                if self.hp_fn is not None:
                    self.header['HISTORY'] = "Bad/hot pixels mask found in {},{} are not cleaned. Do this in cutouts".format(self.bp_fn,self.hp_fn)
                else:
                    self.header['HISTORY'] = "Bad pixels mask found in {} are not cleaned. Do this in cutouts".format(self.bp_fn)
                if stable_bad_pix_map == 'default':
                    self.header['HISTORY'] = "Stable bad pixels in bad_pix_mask/bad_pix_list.npy included, not cleaned"
                if master_dark_hdu[0].header['EXPTIME'] >= 30 and get_bad_pix_from_master_dark == True:
                    self.header['HISTORY'] = "All pixels with value = 0 in long exposure dark included, not cleaned"
                self.header['CLEAN_BP'] = "False"

            #Mask the bad pixels if the flag is set
            if mask_bad_pixels:
                redux = self.full_image*~bad_pixel_map_bool

                #Update the header
                if self.hp_fn is not None:
                    self.header['HISTORY'] = "Masking all bad/hot pixels found in {},{}".format(self.bp_fn, self.hp_fn)
                    self.header['BP_FN'] = self.bp_fn
                    self.header['HP_FN'] = self.hp_fn
                else:
                    self.header['HISTORY'] = "Masking all bad pixels found in {}".format(self.bp_fn)
                    self.header['BP_FN'] = self.bp_fn
                if stable_bad_pix_map == 'default':
                    self.header['HISTORY'] = "Masked all stable bad pixels in bad_pix_mask/bad_pix_list.npy using a median filter"
                if master_dark_hdu[0].header['EXPTIME'] >= 30 and get_bad_pix_from_master_dark == True:
                    self.header['HISTORY'] = "All pixels with value = 0 in long exposure dark masked"

            self.full_image = redux
            #Data quality image: 0 = good, 1 = bad pixel from flat, 2 = hot pixel from dark, 3 in both bad/hot pixel maps

            self.DQ_image = np.ndarray.astype(bad_pixel_map + 2*hot_pixel_map,int)

            # else:
            #     print("No Bad pixel map filename found, continuing without correcting bad pixels")

            #Replace the nans if the flag is set.
            if replace_nans:
                nanmask = np.isnan(self.full_image) #nan = True, just in case this is useful
                self.full_image = np.nan_to_num(self.full_image)


            #Turn on the calibrated flag
            self.calibrated = True
            if report_median:
                return med
            elif report_bkg_multiplier:
                if multicomponent_frame is None:
                    return scale_bkg
                else:
                    return scale_factors
        else:
            print("Data already calibrated")
    
    def generate_bkg(self, method='shift_and_subtract', bkg_fns=None, nclosest = None, same_HWP = True, ref_lib=None, num_PCA_modes=None, \
        source_pos=None, bkg_by_quadrants=False, destripe=False, \
        shift_dir='horizontal', bkg_sub_shift_size = 31, filter_bkg_size=None,verbose=False,**kwargs):
        """
        Generates a model of the background using a variety of possible methods and then saves it to self.bkg_image:

        (1) 'shift_and_subtract': shifts the image by a specified amount and direction and then subtracts (*This is the default method*)
        (2) 'PCA' (Principal component analysis): See Soummer et al. (2012) for a description of the algorithm 
        (3) 'median_ref': Median combines the reference library frames 
        (4) 'scaled_bkg': Takes and scales a manually inputted background frame
        (5) 'simple_median': Calculates the median pixel value of the image and subtracts that off the entire image 
        (6) 'slit_background': Call the image_utils.subtract_slit_background function to fit the backgrond inside slit

        nclosest: If not None, only use n images in bkg_fns exposed closest to the science frame to construct background
        same_HWP: If True, only use bkg_fns with the same HWP angle as the science image
        """
        #put bkg_fns into a list
        # print(bkg_fns)
        
        if type(bkg_fns) == str:
            print('Put background name in list')
            bkg_fns = np.array([bkg_fns])
        if bkg_fns is not None:
            bkg_fns = np.array(bkg_fns)
            #check that ncloset value is valid
            if nclosest is not None:
                nclosest = int(nclosest) #in case somebody put in non integer
                if nclosest > len(bkg_fns): #if you want more 'closest' files than available, use everything
                    nclosest = None
            #if nclosest is not None or same_HWP is True, gather header info on files in bkg_fns
            if (nclosest is not None) or same_HWP == True:
                all_hdr = []
                for i in bkg_fns:
                    all_hdr += [fits.getheader(i)]
                #get some useful quantities
                # coords = np.array([ SkyCoord(x['RA'], x['DEC'], unit = (u.hourangle, u.deg)) for x in all_hdr ])
                # names = np.array([x['RAW_FN'] for x in all_hdr])
                hwps = np.array([x['HWP_ANG'] for x in all_hdr])
                times = np.array([ap_time.Time(x['UTSHUT'], format = 'isot') for x in all_hdr])
                del all_hdr

                time_obs = ap_time.Time(self.header['UTSHUT'], format = 'isot')
                times_diff = np.abs(time_obs - times)

            if same_HWP:
                if verbose: 
                    print("Selecting only background frames with the same HWP angle")
                inds_same_HWP = np.abs(hwps - self.header['HWP_ANG'] ) < 0.1 #some threshold 
                bkg_fns = bkg_fns[inds_same_HWP]
                times_diff = times_diff[inds_same_HWP]

            if nclosest is not None: #Get n numbers of bkg_fns with smallest time difference to 
                inds = times_diff.argsort() #these indices are sorted by absolute time difference
                bkg_fns = (bkg_fns[inds])[0:nclosest] #Then sort bkg_fns based on that, and pick the n closest ones
                if verbose: 
                    print("Selection {} closest background frames".format(nclosest))
            
            if verbose: 
                print("Using background files: {}".format(bkg_fns))

            #Catch an error if there's no background fitting the criteria
            if len(bkg_fns) == 0:
                raise ValueError('No background file matching your criteria, try setting same_HWP = False')
            #for debugging
            # print(bkg_fns)
            
            self.ref_lib = bkg_fns
        
        #default shift and subtract method
        if method == 'shift_and_subtract':
            if verbose:
                print('Subtracting background using shift and subtract method.')
            self.bkg_image = spec_utils.shift_and_subtract(self.full_image, shift_dir=shift_dir, bkg_sub_shift_size=bkg_sub_shift_size, filter_bkg_size=filter_bkg_size)
            
        #PCA background subtraction
        if method == 'PCA':
            #If no ref lib is provided, use the default one. If it is provided, we don't want to overwrite the default at this point. 
            if bkg_fns is not None:
                self.ref_lib = bkg_fns        

            if num_PCA_modes is not None:
                #Do the PCA subtraction, save the model image to self.bkg_image. It also outputs a subtracted image, ignore that for now.
                if self.ref_lib is not None: 
                    _ , self.bkg_image  = calibration.PCA_subtraction(self.full_image, self.ref_lib, num_PCA_modes,**kwargs)
                else:
                    _ , self.bkg_image  = calibration.PCA_subtraction(self.full_image, ref_lib, num_PCA_modes,**kwargs)

        #median reference frame background subtraction
        elif method =='median_ref':

            #If no ref lib is provided, use the default one. If it is provided, we don't want to overwrite the default at this point. 
            if bkg_fns is not None:
                self.ref_lib = bkg_fns

            ref_lib = self.ref_lib

            if ref_lib is not None:

                if len(ref_lib) == 1:
                    self.bkg_image  = fits.getdata(ref_lib[0])
                else:
                    bkg_frames = []
                    for i in range(len(ref_lib)):
                        bkg_frames.append(fits.getdata(ref_lib[i]))
                    if verbose:
                        print('Subtracting background using median reference frame.')
                    # import pdb;pdb.set_trace()
                    self.bkg_image = np.nanmedian(np.array(bkg_frames), axis=0)        
            else:
                print('Must provide a list of reference files to perform PCA subtraction, either as a keyword argument "ref_lib" or in self.ref_lib.')

        #using background frame that you manually give it
        elif method == 'scaled_bkg':
            
            #If no background file is provided, use the default one. If it is provided, we don't want to overwrite the default at this point. 
            if bkg_fns is not None:
                self.ref_lib = bkg_fns

            bkg_fns = self.ref_lib

            if bkg_fns is not None: 

                #Check to see if we have a list of files. If so, we take the median
                if not isinstance(bkg_fns, str):
                    # import pdb; pdb.set_trace()
                    if len(bkg_fns) == 1:
                        background  = fits.getdata(bkg_fns[0])
                        background_hdu = fits.open(bkg_fns[0]) #Grab the last header
                    else:
                        bkg_frames = []
                        for i in range(len(bkg_fns)):
                            bkg_frames.append(fits.getdata(bkg_fns[i]))    
                        background_hdu = fits.open(bkg_fns[i]) #Grab the last header
                        background = np.nanmedian(np.array(bkg_frames), axis=0) 
                else:
                    #Just open the one file!
                    background_hdu = fits.open(bkg_fns)
                    background = background_hdu[0].data

                if verbose:
                    print('Subtracting background using scaled background frame from:')
                    print(bkg_fns)
                                
                bkg_exp_time = background_hdu[0].header["EXPTIME"]*background_hdu[0].header["COADDS"]
                
                #Check if background is already reduced
                try:
                    bkg_reduced = background_hdu[0].header["CALBRTED"]
                except KeyError:
                    bkg_reduced = False

                if bkg_reduced == False:
                    #Checking Dark Exposure times and scaling if need be
                    if self.dark_fn is not None:
                        #Open the master dark
                        master_dark_hdu = fits.open(self.dark_fn)
                        master_dark = master_dark_hdu[0].data
                        if verbose:
                            print(("Subtracting {} from the background image".format(self.dark_fn)))
                        dark_exp_time = master_dark_hdu[0].header['EXPTIME'] * master_dark_hdu[0].header['COADDS']
                        if dark_exp_time != bkg_exp_time:
                            if verbose:
                                print("The master dark file doesn't have the same exposure time as the background image. We'll scale the dark for now, but this isn't ideal")
                            bk_factor = bkg_exp_time/dark_exp_time
                        else:
                            bk_factor = 1.
                        if verbose:
                            print("Subtracting background frame {} from all science files".format(bkg_fns))

                        background = background - bk_factor*master_dark

                    if self.flat_fn is not None:
                        master_flat = fits.getdata(self.flat_fn)
                        background = background/master_flat
                # else: #if the background is already reduced
                    # pass #do nothing, it's already good!

                self.bkg_image = background

                #Read in the mask. 
                full_mask = fits.open(mask_path+"full_frame_mask_2019.fits")[0].data
                full_mask[full_mask==0] = np.nan

                ### If this flag is set, you estimate the scaling by the median of each quadrant, not the whole image. 
                if bkg_by_quadrants:
                    sci_mean, sci_med, sci_std = sigma_clipped_stats((self.full_image*full_mask)[:1063,:1027].flatten())
                    bkg_mean, bkg_med, bkg_std = sigma_clipped_stats((background*full_mask)[:1063,:1027].flatten())
                    scale_bkg1 = sci_med/bkg_med
                    # scale_bkg1 = np.nanmedian((self.full_image*full_mask)[:1063,:1027]/(background*full_mask)[:1063,:1027])
                    
                    sci_mean, sci_med, sci_std = sigma_clipped_stats((self.full_image*full_mask)[:1063,1027:].flatten())
                    bkg_mean, bkg_med, bkg_std = sigma_clipped_stats((background*full_mask)[:1063,1027:].flatten())
                    scale_bkg2 = sci_med/bkg_med
                    # scale_bkg2 = np.nanmedian((self.full_image*full_mask)[:1063,1027:])/np.nanmedian((background*full_mask)[:1063,1027:])

                    sci_mean, sci_med, sci_std = sigma_clipped_stats((self.full_image*full_mask)[1063:,:1027].flatten())
                    bkg_mean, bkg_med, bkg_std = sigma_clipped_stats((background*full_mask)[1063:,:1027].flatten())
                    scale_bkg3 = sci_med/bkg_med
                    # scale_bkg3 = np.nanmedian((self.full_image*full_mask)[1063:,:1027])/np.nanmedian((background*full_mask)[1063:,:1027])

                    sci_mean, sci_med, sci_std = sigma_clipped_stats((self.full_image*full_mask)[1063:,1027:].flatten())
                    bkg_mean, bkg_med, bkg_std = sigma_clipped_stats((background*full_mask)[1063:,1027:].flatten())
                    scale_bkg4 = sci_med/bkg_med
                    # scale_bkg4 = np.nanmedian((self.full_image*full_mask)[1063:,1027:])/np.nanmedian((background*full_mask)[1063:,1027:])                    
                    
                    background[:1063,:1027] *= scale_bkg1
                    background[:1063,1027:] *= scale_bkg2
                    background[1063:,:1027] *= scale_bkg3
                    background[1063:,1027:] *= scale_bkg4

                else:
                    scale_bkg = np.nanmedian(self.full_image)/np.nanmedian(background)
                    background *= scale_bkg

                # print(np.nanmedian(self.full_image*full_mask-background*full_mask))
                self.bkg_image = background

                if len(bkg_fns) == 1:
                    #Update the header
                    # print(bkg_fns)
                    self.header['HISTORY'] = "Generated scaled background frame {}".format(bkg_fns[0])
                    # self.header['BKG_FN'] = bkg_fns[0]
                else: 
                    self.header['HISTORY'] = "Generated a scaled background based on the median of a big background list"
                    
            else:
                print('Must give wirc object a bkg_fns in order to use this method.')        

        elif method == 'simple_median':
            print('Subtracting median pixel value from frame.')
            mean, med, std = sigma_clipped_stats(self.full_image.flatten())
            self.bkg_image = self.full_image*0.+med
            # self.full_image = self.full_image - med
            # self.bkg_subbed = True
        
        elif method == 'slit_background':
            if verbose:
                print("Subtracting the slit background only, this can sometimes take a few seconds")
            #TODO: Check to make sure we're only operating in J-band
            #TODO: Make sure we're in pol-mode
        
            self.bkg_image = image_utils.subtract_slit_background(self.full_image, bad_pixel_mask = self.DQ_image, band = self.filter_name[0] ,**kwargs)

        self.header['BKG_MTHD'] = method

        if method != "slit_background" and method != 'shift_and_subtract' and method != "cutout_median" and method != "simple_median":
            nbkg_files = len(self.ref_lib)
            self.header['N_BKGFN'] = str(nbkg_files)
            for i in range(nbkg_files):
                key = "BKGFN"+str(i).zfill(3)
                self.header[key] = self.ref_lib[i]

        if destripe:
            if verbose:
                print("Destriping the detector image")
            self.full_image = calibration.destripe_after_bkg_sub(self.full_image-self.bkg_image)+self.bkg_image

    def make_triplet_table(self, array_in, c1list, c2list, c3list):
        #convert array to fits columns and then fits tables. returns a fits table with 3 columns.

        #developed to be called by save_wirc_object (the previously listed function)

        #first verifies if array_in has information (not None)
        length=[] #initiates list
        if array_in is not None:
                #print ("array_in is not None")

                #verifies/determines if array_in.ndim is 2 or 3.
                #if 2
                if array_in.ndim ==2:
                    #print("array_in.ndim ==2");

                    #defines columns, including data
                    c1 = fits.Column(name=c1list[0],format=c1list[1],unit=c1list[2], array=array_in[0,:])
                    c2 = fits.Column(name=c2list[0],format=c2list[1],unit=c2list[2], array=array_in[1,:])
                    c3 = fits.Column(name=c3list[0],format=c3list[1],unit=c3list[2], array=array_in[2,:])

                #if 3
                elif array_in.ndim ==3:
                    #print("array_in.ndim ==3");
                    #finds the extra index from the name (0th item in list, 14th character in string, converted to int)
                    #print(c1list[0])
                    ex_i=int(c1list[0][14])

                    #defines columns, including data
                    c1 = fits.Column(name=c1list[0],format=c1list[1],unit=c1list[2], array=array_in[ex_i,0,:])
                    c2 = fits.Column(name=c2list[0],format=c2list[1],unit=c2list[2], array=array_in[ex_i,1,:])
                    c3 = fits.Column(name=c3list[0],format=c3list[1],unit=c3list[2], array=array_in[ex_i,2,:])

                #if array_in is neither ndim, raises warning to user, and leaves columns blank to allow rest of program to run
                else:
                    #print ("Warning: While trying to convert array_in into a 3 column table, array_in.ndim != 2 or 3")

                    #defines columns, not including data
                    c1 = fits.Column(name=c1list[0],format=c1list[1],unit=c1list[2], array=np.array([]))
                    c2 = fits.Column(name=c2list[0],format=c2list[1],unit=c2list[2], array=np.array([]))
                    c3 = fits.Column(name=c3list[0],format=c3list[1],unit=c3list[2], array=np.array([]))




        #if array_in is None, initiates blank columns
        else :
                #print ("array_in == None")

                #defines columns, not including data
                c1 = fits.Column(name=c1list[0],format=c1list[1],unit=c1list[2], array=np.array([]))
                c2 = fits.Column(name=c2list[0],format=c2list[1],unit=c2list[2], array=np.array([]))
                c3 = fits.Column(name=c3list[0],format=c3list[1],unit=c3list[2], array=np.array([]))

        length=[len(c1.array),len(c2.array),len(c2.array)] #defines length list as the length of the arrays given to each column

        #returns table equivalent of array_in and corresponding c<#>lists, also returns length list
        return fits.BinTableHDU.from_columns(fits.ColDefs([c1,c2,c3])),length

    def table_columns_to_array(self,table_in,prihdr,cil):
        list3columns = [] #initiates a list of arrays representing the columns

        if len(cil) ==3: #if there are 3 columns

            #appends the padding-removed arrays (from the columns) to the list3columns
            for j in range(len(cil)):
                list3columns.append(table_in.field(cil[j])[0:prihdr['TLENG'+str(cil[j]+1)]])

            #stacks the list together to make 2D output array
            array_out=np.stack((list3columns[0],list3columns[1],list3columns[2]))

        elif len(cil) ==12: #if there are 12 columns

            for j in range(len(cil)):
                list3columns.append(table_in.field(cil[j])[0:prihdr['TLENG'+str(cil[j]+1)]])

            #stacks portion of list together to form 4 2D arrays
            array_a=np.stack((list3columns[0],list3columns[1],list3columns[2]))
            array_b=np.stack((list3columns[3],list3columns[4],list3columns[5]))
            array_c=np.stack((list3columns[6],list3columns[7],list3columns[8]))
            array_d=np.stack((list3columns[9],list3columns[10],list3columns[11]))

            #stacks the 2D arrays to form a 3D output array
            array_out=np.stack((array_a,array_b,array_c,array_d),axis=0)


        else:
            print ("Warning: column list improper number of columns")
            array_out = np.array([])#None
        return array_out

    def save_wirc_object(self, wirc_object_filename, overwrite = True, save_full_image = True, verbose=True):
        #Save the object to a fits file

        # vers = version.get_version()
        # self.header.set('PL_VERS',vers,'Version of pipeline used for processing')
        #common indexing notes for save_wirc_object and load_wirc_object:
        #----#(2*i)+1 is a conversion from the index, i, of the source in source_list to the index of the source in hdulist
        #----#(2*i)+2 is a conversion from the index, i, of the source in source_list to the index of the source's corresponding table in hdulist



        #TODO: Update the header keywords below to include a keyword description like PS_VERS above

        #These may not always be set by other function
        self.header["NSOURCES"] = self.n_sources
        self.header["DARK_FN"] = self.dark_fn
        self.header["FLAT_FN"] = self.flat_fn
        self.header["BP_FN"] = self.bp_fn
        # self.header["BKG_FN"] = self.bkg_fn

        #Have the data been calibrated/background subtracted?
        self.header["CALBRTED"] = self.calibrated
        self.header["BKG_SUBD"] = self.bkg_subbed

        #add in time stamp
        self.header["BJD"]=self.bjd

        #Was it marked bad?
        if self.bad_flag:
            self.header['BAD_FLAG'] = "True"
            self.header.comments['BAD_FLAG'] = self.bad_reason
        else:
            self.header['BAD_FLAG'] = "False"

        if save_full_image:
            hdu = fits.PrimaryHDU(self.full_image)
        else:
            hdu = fits.PrimaryHDU([])

        hdu.header = self.header

        hdulist = fits.HDUList([hdu])

        #Append the dataquality frame
        if self.DQ_image is not None and save_full_image:
            hdulist.append(fits.ImageHDU(self.DQ_image))
        else:
            hdulist.append(fits.ImageHDU([]))

        #Now for each source, create a ImageHDU, this works even if the cutouts haven't been extracted
        #Now for each source, create a TableHDU

        for i in range(self.n_sources):
            #print ('Starting Iteration #',i);

            ### Create an ImageHDU for each of the sources

            #First a list of the different image we'll include
            traces_image_list = [self.source_list[i].trace_images]
            source_img_list = "SCI"

            if self.source_list[i].trace_images_DQ is not None:
                traces_image_list.append(self.source_list[i].trace_images_DQ)
                source_img_list += ",DQ"

            if self.source_list[i].trace_bkg is not None:
                traces_image_list.append(self.source_list[i].trace_bkg)
                source_img_list += ",BKG"

            if self.source_list[i].trace_images_extracted is not None:
                traces_image_list.append(self.source_list[i].trace_images_extracted)
                source_img_list += ",EXTR"

            traces_image_list = np.vstack(np.array(traces_image_list))
            # print(traces_image_list.shape)
            source_hdu = fits.PrimaryHDU(traces_image_list)
            source_hdu.header['TRC_IMGS'] = source_img_list

            # source_hdu = fits.ImageHDU(self.source_list[i].trace_images)
            # if self.source_list[i].trace_images_extracted is not None and self.source_list[i].trace_images_DQ is not None:
            #     source_hdu = fits.PrimaryHDU(np.concatenate([self.source_list[i].trace_images, self.source_list[i].trace_images_DQ,
            #                                             self.source_list[i].trace_images_extracted]))
            # elif self.source_list[i].trace_images_DQ is not None:
            #     source_hdu = fits.PrimaryHDU(np.concatenate([self.source_list[i].trace_images,
            #                                             self.source_list[i].trace_images_DQ]))
            # else:
            #     source_hdu = fits.PrimaryHDU(self.source_list[i].trace_images)

            #Put in the source info
            source_hdu.header["XPOS"] = self.source_list[i].pos[0]
            source_hdu.header["YPOS"] = self.source_list[i].pos[1]

            #only write position errors if they exist.
            if len(self.source_list[i].pos)>2:
                source_hdu.header["XPOS_ERR"] = self.source_list[i].pos[2]
                source_hdu.header['YPOS_ERR'] = self.source_list[i].pos[3]



            source_hdu.header["SLIT_LOC"] = self.source_list[i].slit_pos

            #Data reduction status headers for each source
            source_hdu.header["WL_CBRTD"] = (self.source_list[i].lambda_calibrated,"Wavelength Calibrated? status")
            source_hdu.header["POL_CMPD"] = (self.source_list[i].polarization_computed,"Polarization Computed? status")
            source_hdu.header["SPC_XTRD"] = (self.source_list[i].spectra_extracted,"Spectra Extracted? status")
            source_hdu.header["THMB_CUT"] = (self.source_list[i].thumbnails_cut_out,"Thumbnails cut out? status")

            #widths and angles of the traces
            if self.source_list[i].spectra_widths is not None:
                # print(self.source_list[i].spectra_widths)
                source_hdu.header["WIDTHS"] = (np.array2string(self.source_list[i].spectra_widths), "Widths of spectra in unrotated image")
                source_hdu.header["ANGLES"]=  (np.array2string(self.source_list[i].spectra_angles), "Angles of spectra in unrotated image")


            #Append it to the hdu list
            hdulist.append(source_hdu)


            #TODO: Add a fits table extension (or a series of them) to contain the spectra
            #Create a TableHDU for each of the sources

            #The source_list attributes, trace_spectra(four separate trace spectra), Q, U, P, theta, are converted into tables of three columns each. Also returns length lists of each array
            t_ts_0,l0=self.make_triplet_table(self.source_list[i].trace_spectra, ['trace_spectra_0 wavelength','D','nm'], ['trace_spectra_0 flux','D','units?'], ['trace_spectra_0 flux error','D','units?'])#trace spectra 0
            t_ts_1,l1=self.make_triplet_table(self.source_list[i].trace_spectra, ['trace_spectra_1 wavelength','D','nm'], ['trace_spectra_1 flux','D','units?'], ['trace_spectra_1 flux error','D','units?'])#trace spectra 1
            t_ts_2,l2=self.make_triplet_table(self.source_list[i].trace_spectra, ['trace_spectra_2 wavelength','D','nm'], ['trace_spectra_2 flux','D','units?'], ['trace_spectra_2 flux error','D','units?'])#trace spectra 2
            t_ts_3,l3=self.make_triplet_table(self.source_list[i].trace_spectra, ['trace_spectra_3 wavelength','D','nm'], ['trace_spectra_3 flux','D','units?'], ['trace_spectra_3 flux error','D','units?'])#trace spectra 3

            #if calibrated, add the calibrated trace

            t_ts_0_cal,l0_cal=self.make_triplet_table(self.source_list[i].calibrated_trace_spectra, ['trace_spectra_0_calibrated wavelength','D','nm'], ['trace_spectra_0_calibrated flux','D','units?'], ['trace_spectra_0_calibrated flux error','D','units?'])#trace spectra 0
            t_ts_1_cal,l1_cal=self.make_triplet_table(self.source_list[i].calibrated_trace_spectra, ['trace_spectra_1_calibrated wavelength','D','nm'], ['trace_spectra_1_calibrated flux','D','units?'], ['trace_spectra_1_calibrated flux error','D','units?'])#trace spectra 1
            t_ts_2_cal,l2_cal=self.make_triplet_table(self.source_list[i].calibrated_trace_spectra, ['trace_spectra_2_calibrated wavelength','D','nm'], ['trace_spectra_2_calibrated flux','D','units?'], ['trace_spectra_2_calibrated flux error','D','units?'])#trace spectra 2
            t_ts_3_cal,l3_cal=self.make_triplet_table(self.source_list[i].calibrated_trace_spectra, ['trace_spectra_3_calibrated wavelength','D','nm'], ['trace_spectra_3_calibrated flux','D','units?'], ['trace_spectra_3_calibrated flux error','D','units?'])#trace spectra 3


            t_Q,lQ=self.make_triplet_table(self.source_list[i].Q, ['Q wavelength','D','nm'], ['Q stokes','D','units?'], ['Q stokes error','D','units?'])               #Q

            t_U,lU=self.make_triplet_table(self.source_list[i].U, ['U wavelength','D','nm'], ['U stokes','D','units?'], ['U stokes error','D','units?'])               #U

            t_P,lP=self.make_triplet_table(self.source_list[i].P, ['P wavelength','D','nm'], ['P','D','units?'], ['P error','D','units?'])               #P

            t_theta,ltheta=self.make_triplet_table(self.source_list[i].theta, ['theta wavelength','D','nm'], ['theta','D','units?'], ['theta error','D','units?'])       #theta
            #tables of 3 columns each have been made



            #big table gets made
            #the .columns of each three-column-table are added up to form column_list
            column_list= t_ts_0.columns + t_ts_1.columns + t_ts_2.columns + t_ts_3.columns +  t_Q.columns + t_U.columns  + t_P.columns + t_theta.columns + t_ts_0_cal.columns+t_ts_1_cal.columns+t_ts_2_cal.columns+t_ts_3_cal.columns

            #the column_list becomes a quite large fits table called source_tbl_hdu
            source_tbl_hdu=fits.BinTableHDU.from_columns(column_list)



            #Append it to the hdu list
            hdulist.append(source_tbl_hdu)

            length_list=l0+l1+l2+l3+lQ+lU+lP+ltheta+l0_cal+l1_cal+l2_cal+l3_cal  #making a list of the lengths of columns
            #print ('Ending Iteration #',i);

            #Creates a header keyword, value, and comment.
            #The value designates the length the array that would correspond to the column.
            for k in range(len(length_list)):
                #defines keyword string
                header_keyword="TLENG"+str(k+1)
                #defines comment string
                header_comment="Length of "+hdulist[(2*i)+3].data.names[k]


                hdulist[(2*i)+3].header[header_keyword]=(length_list[k],header_comment) #defines the keyword with value and comment

        #For loop ended
        #print ('No more iterations');




        #Saving a wirc_object (hdulist)
        if verbose:
            print("Saving a wirc_object to {}".format(wirc_object_filename));
        self.header["FN"] = wirc_object_filename
        hdulist.writeto(wirc_object_filename, overwrite=overwrite)

        
    def load_wirc_object(self, wirc_object_filename, load_full_image = True, verbose=True, clear_sources=False):
        '''
        Read in the wircpol_object file from a fits file

        Loading in a full_image is optional, in case you only need to look at extracted spectra and cutouts.
        Set load_full_image = False to not load full image.

        '''
        #common indexing notes for save_wirc_object and load_wirc_object:
        #----#(2*i)+1 is a conversion from the index, i, of the source in source_list to the index of the source in hdulist
        #----#(2*i)+2 is a conversion from the index, i, of the source in source_list to the index of the source's corresponding table in hdulist

        #Open the fits file
        with fits.open(wirc_object_filename) as hdulist:

            #Read in the full image and the primary header
            if load_full_image:
                temp = hdulist[0].data
                self.full_image = copy.deepcopy(temp)
                try:
                    temp2 = hdulist[1].data
                    self.DQ_image = copy.deepcopy(temp2)
                except:
                    if verbose:
                        print("Couldn't open DQ extension, maybe you saved the wircpol object with an older version?")

            else:
                self.full_image = None
            
        
            temp = hdulist[0].header
            self.header = copy.deepcopy(temp)

            self.header["FN"] = wirc_object_filename

            #What are the calibration filenames?
            self.dark_fn = self.header["DARK_FN"]
            self.flat_fn = self.header["FLAT_FN"]
            self.bp_fn = self.header["BP_FN"]
            # self.bkg_fn = self.header["BKG_FN"]

            self.filter_name = self.header['AFT'][0]

            #What's the calibration status?
            self.calibrated = self.header["CALBRTED"]
            self.bkg_subbed = self.header["BKG_SUBD"]

            #How many sources are there
            self.n_sources = self.header["NSOURCES"]

            #BJD_TDB
            try:
                self.bjd = self.header["BJD"]
            except KeyError as err:
                if verbose:
                    print(err)
            
            #A bad flag that indicates this whole file should be disregarded.
            try:
                self.bad_flag = self.header['BAD_FLAG']
                self.bad_reason = self.header.comments['BAD_FLAG']

                if self.bad_flag == "True":
                    self.bad_flag = True
                elif self.bad_flag == "False":
                    self.bad_flag = False

            except Exception as e:

                # if verbose:
                    # print("BAD_FLAG Error {}".format(e))

                self.bad_flag = False
                self.bad_reason = ""

            #TODO: This is temporary and needs fixing. 
            self.bkg_image = None

            if clear_sources:
                self.n_sources=0

            try: 
                nbkg_files = int(self.header['N_BKGFN'])
                self.ref_lib = [] #Background reference library
                for i in range(nbkg_files):
                    self.ref_lib.append(self.header["BKGFN"+str(i).zfill(3)])
            except KeyError:
                if verbose:
                    print("Didn't find any background files")
                self.ref_lib = None
            

            #Create one source object for each source and append it to source_list
            self.source_list = []

            for i in range(self.n_sources):
                #print ("starting iteration #",i)
                #Extract the source info from the header
                xpos        = copy.deepcopy(hdulist[(2*i)+2].header["XPOS"])
                ypos        = copy.deepcopy(hdulist[(2*i)+2].header["YPOS"])
                slit_loc    = copy.deepcopy(hdulist[(2*i)+2].header["SLIT_LOC"])

                #if they are there positions errors, then take them. If not, don't!
                try:
                    xpos_err = copy.deepcopy(hdulist[(2*i)+2].header["XPOS_ERR"])
                    ypos_err = copy.deepcopy(hdulist[(2*i)+2].header["YPOS_ERR"])
                    new_source = wircpol_source([xpos,ypos,xpos_err,ypos_err],slit_loc, i)

                except KeyError:
                    new_source = wircpol_source([xpos,ypos],slit_loc, i)


                #Get the list of saved trace images - an old method and a new one. 
                try:
                    # if hdulist[(2*i)+2].header['TRC_IMGS'] is not None:
                        # print(hdulist[(2*i)+2].header['TRC_IMGS'])

                    source_img_list=hdulist[(2*i)+2].header['TRC_IMGS'].split(",")
                    
                    if "SCI" in source_img_list:
                        sci_ind = source_img_list.index("SCI")
                        new_source.trace_images = copy.deepcopy(hdulist[(2*i)+2].data[4*sci_ind:4*(sci_ind+1)])
                    
                    if "DQ" in source_img_list:
                        dq_ind =  source_img_list.index("DQ")
                        new_source.trace_images_DQ = copy.deepcopy(hdulist[(2*i)+2].data[4*dq_ind:4*(dq_ind+1)])
                    
                    if "EXTR" in source_img_list:
                        extr_ind =  source_img_list.index("EXTR")
                        new_source.trace_images_extracted = copy.deepcopy(hdulist[(2*i)+2].data[4*extr_ind:4*(extr_ind+1)])

                    if "BKG" in source_img_list:
                        bkg_ind =  source_img_list.index("BKG")
                        new_source.trace_bkg = copy.deepcopy(hdulist[(2*i)+2].data[4*bkg_ind:4*(bkg_ind+1)])

                    
                        # new_source.trace_images = copy
                except KeyError:
                    if verbose:
                        print("Using the old way of reading in the trace images")

                    try: 
                        new_source.trace_images             = copy.deepcopy(hdulist[(2*i)+2].data[0:4]) #finds the i'th source image data in the hdulist, first 4 are raw images
                        new_source.trace_images_DQ          = copy.deepcopy(hdulist[(2*i)+2].data[4:8])
                        new_source.trace_images_extracted   = copy.deepcopy(hdulist[(2*i)+2].data[8:] )#last 4 images are from which extraction is done.
                    except: 
                        print("Some error extracting cutout images. Maybe they weren't saved.")

                #finds the table data of the TableHDU corresponding to the i'th source
                big_table = copy.deepcopy(hdulist[(2*i)+3].data )

                #finds the header of the TableHDU corresponding to the i'th source
                prihdr = copy.deepcopy(hdulist[(2*i)+3].header )


                #for the column number, refers to the variable "column_list" in save_wirc_object. each variable has 4 columns for 4 traces
                #finds 3D array for trace_spectra
                new_source.trace_spectra = self.table_columns_to_array(big_table,prihdr,[0,1,2,3,4,5,6,7,8,9,10,11])
                #if extracted trace_spectra exists
                #print('big table len is ',len(big_table))
                #print('prihdr len is ',len(prihdr))
               # print(prihdr['TFIELDS'])
                if prihdr['TFIELDS'] ==36:
                    new_source.calibrated_trace_spectra = self.table_columns_to_array(big_table,prihdr,[24,25,26,27,28,29,30,31,32,33,34,35])

                #finds 2D array for Q
                new_source.Q = self.table_columns_to_array(big_table,prihdr,[12,13,14])

                #finds 2D array for U
                new_source.U = self.table_columns_to_array(big_table,prihdr,[15,16,17])

                #finds 2D array for P
                new_source.P = self.table_columns_to_array(big_table,prihdr,[18,19,20])

                #finds 2D array for theta
                new_source.theta = self.table_columns_to_array(big_table,prihdr,[21,22,23])

                #adjusting source header statuses
                new_source.lambda_calibrated        = copy.deepcopy(hdulist[(2*i)+2].header["WL_CBRTD"])#source attribute, later applied to header["WL_CBRTD"]
                new_source.polarization_computed    = copy.deepcopy(hdulist[(2*i)+2].header["POL_CMPD"]) #source attribute, later applied to header["POL_CMPD"]
                new_source.spectra_extracted        = copy.deepcopy(hdulist[(2*i)+2].header["SPC_XTRD"]) #source attribute, later applied to header["SPC_XTRD"]
                new_source.thumbnails_cut_out       = copy.deepcopy(hdulist[(2*i)+2].header["THMB_CUT"]) #source attribute, later applied to header["THMB_CUT"]

                try:
                    new_source.spectra_widths = np.fromstring(hdulist[(2*i)+2].header["WIDTHS"][1:-1], sep = ' ')
                    new_source.spectra_angles = np.fromstring(hdulist[(2*i)+2].header["ANGLES"][1:-1], sep = ' ')
                except KeyError:
                    None



                #Append it to the source_list
                self.source_list.append(new_source)
            hdulist.close()
            self.header["FN"] = wirc_object_filename
            #print ("ending iteration #",i)

    def find_sources_v2(self, bkg_im=None, cross_correlation_template=None, sigma_threshold=0, show_plots=True,perc_threshold=98,
    update_w_chi2_shift=False, slit_mask=None, verbose=False):
        """
        Finds the number of sources in the image.

        cross_correlation_template: 2-D np.array
            By default, uses cross_correlation_template under wircpol_masks module of wirc_drp, but if you provide a template, it will use that instead. (Default=None)
        sigma_threshold: fl
            only keeps sources that have a flux above specified sigma threshold. (Default=0)
        show_plots: bool
            if True, shows plots. (Default=True)
        """
        if self.cross_correlation_template is None:
            if cross_correlation_template is None:
                self.cross_correlation_template = wircpol_masks.cross_correlation_template
            else:
                self.cross_correlation_template = cross_correlation_template
        #self.source_list, self.trace_fluxes = image_utils.find_sources_in_direct_image_v2(self.full_image, self.cross_correlation_template, sigma_threshold=sigma_threshold, show_plots=show_plots)
		#make sure the source_list format is correct
        if bkg_im is not None:
            loc_list, self.trace_fluxes = image_utils.find_sources_in_wircpol_image(self.full_image-bkg_im, self.cross_correlation_template,
            sigma_threshold=sigma_threshold, show_plots=show_plots,perc_threshold=perc_threshold,verbose=verbose, slit_mask=slit_mask)
        elif bkg_im is None:
            loc_list, self.trace_fluxes = image_utils.find_sources_in_wircpol_image(self.full_image, self.cross_correlation_template,
            sigma_threshold=sigma_threshold, show_plots=show_plots,perc_threshold=perc_threshold,verbose=verbose, slit_mask=slit_mask)

        if self.source_list:
            print('emptying source list to find sources again.')
            self.source_list = []

        if len(loc_list) >= 1:
            for loc in loc_list:
                slit_dist = np.sqrt((loc[0] - constants.slit_position_x)**2 + (loc[1] - constants.slit_position_y)**2) 
                if slit_dist < 20:
                    self.add_source(loc[0], loc[1],update_w_chi2_shift=update_w_chi2_shift,slit_pos = "slitless")
                else: 
                    self.add_source(loc[0], loc[1],update_w_chi2_shift=update_w_chi2_shift,slit_pos = 1)

        self.n_sources = len(self.source_list)
        self.header['NSOURCES'] = self.n_sources

    def mask_sources(self, trace_template=None, sigma_threshold=0, boxsize=10, save_path=None, show_plot=True, overwrite=False):
        """
        masks sources in image

        trace_template: 2-D np.array
            By default, uses the trace_template under wircpol_masks module of wirc_drp, but if you provide a template, it will use that instead. (Default=None)
        boxsize: int
            size of box used to do median fill of traces. (Default=10)
        save_path: str
            if not None, saves masked image as .fits file to specified path. (Default=None)
        show_plot: bool
            if True, shows masked image. (Default=True)
        overwrite: bool
            if True, overwrites full_image with masked image. Else, saves masked image to self.masked_image (Default=False)
        """
        if self.trace_template is None:
            if trace_template is None:
                self.trace_template = wircpol_masks.trace_template
            else:
                self.trace_template = trace_template
        if (not any(self.source_list)) and (not self.already_masked):
            print('Need to find sources first. Running source finding algorithm.')
            self.find_sources_v2(sigma_threshold=sigma_threshold, show_plots=show_plot)

            if overwrite:
                self.full_image = image_utils.mask_sources_util(self.full_image.copy().astype(float), self.trace_template, self.source_positions, self.trace_fluxes,
                                                                boxsize=boxsize, save_path=save_path, show_plot=show_plot)
            else:
                self.masked_image = image_utils.mask_sources_util(self.full_image.copy().astype(float), self.trace_template, self.source_positions, self.trace_fluxes,
                                                                boxsize=boxsize, save_path=save_path, show_plot=show_plot)

            self.already_masked=True

        elif not any(self.source_list):
            print('No sources to mask.')
        elif self.already_masked:
            print('Image already masked. See self.masked_image or self.full_image if overwrite=True')
        else:
            if overwrite:
                self.full_image = image_utils.mask_sources_util(self.full_image.copy().astype(float), self.trace_template, self.source_positions, self.trace_fluxes,
                                                                boxsize=boxsize, save_path=save_path, show_plot=show_plot)
            else:
                self.masked_image = image_utils.mask_sources_util(self.full_image.copy().astype(float), self.trace_template, self.source_positions, self.trace_fluxes,
                                                                boxsize=boxsize, save_path=save_path, show_plot=show_plot)

            self.already_masked=True
            
    def find_sources(self, image_fn, sky = None, threshold_sigma = 5, guess_seeing = 1, plot = False, verbose = False, brightness_sort=False, update_w_chi2_shift=True, im_package = 'cv2', max_sources=5,
                    use_full_frame_mask=True, force_figures = False, mode = 'pol', guess_location = None):

        """
        Find the number of sources in the image and create a wircpol_source objects for each one. In 'pol' mode the traces will be verified and only the good sources will be saved.

        Args:
            image_fn - Either the *direct image* with no mask or PG (for 'direct' mode), specpol image with only source in slit or at known location ('simple' mode),
                        specpol image with multiple sources ('pol' mode), or grism image ('spec' mode, still not incorporated here)
            sky - Offset image for sky background subtraction (if not already applied)
            threshold_sigma - Sigma cutoff for trace detection
            guess_seeing - Approximate seeing during observation
            guess_location - The location of the source, only work if mode is 'simple'. Format is [x,y]
        """

        if mode == 'direct':
            if verbose:
                print('Finding sources in direct image with no mask or PG ...')

            #Open the direct image
            direct_image = fits.open(image_fn)[0].data

            #Get the focal plane mask.
            mask = cross_mask_ns.astype('bool') #What does our mask look like?

            #Find the sources
            locations = image_utils.find_sources_in_direct_image(direct_image, mask, threshold_sigma = threshold_sigma, guess_seeing = guess_seeing, plot = plot)

            #How many sources are there?
            self.n_sources = np.shape(locations[0,:])[0]+1
            self.header['NSOURCES'] = self.n_sources

            #Append all the new objects
            for source in range(self.n_sources):
                self.source_list.append(wircpol_source(locations[source, 0], locations[source,1],source))

            # else:
            #     print("No direct image filename given. For now we can only find sources automatically in a direct image, so we'll assume that there's a source in the middle slit. If you wish you can add other sources as follows: \n\n > wirc_data.source_list.append(wircpol_source([y,x],slit_pos,wirc_data.n_sources+1) \
            #     #where slit_pos is '0','1','2' or slitless. \n > wirc_data.n_sources += 1")
        elif mode == 'simple':
            # if guess_location is None:
            #     if verbose:
            #         print("Assuming just a source in slit. If you wish you can add more sources as follows: \n\n > wirc_data.source_list.append(wircpol_source([y,x],slit_pos,wirc_data.n_sources+1) where slit_pos is '0','1','2' or slitless. \n > wirc_data.n_sources += 1 ")
            #     self.source_list.append(wircpol_source([1063,1027],'1',self.n_sources+1))
            # elif len(guess_location) == 2:
            #     x = guess_location[0]
            #     y = guess_location[1]
            #     if update_w_chi2_shift:
            #         x,y = image_utils.update_location_w_chi2_shift(self.full_image, x, y, self.filter_name)
            #     if verbose:
            #         print("Use the given location x,y = %.2f, %.2f"%(guess_location[0], guess_location[1]))
            #     self.source_list.append(wircpol_source([y,x],'slitless',self.n_sources+1))
            #revert to just slit
            # else:
            #     #if verbose:
            #     print("Leave guess_location as None if source is in slit, otherwise give guess_location in [x,y] format.")

            if verbose:
                print("Assuming just a source in slit. If you wish you can add more sources as follows: \n\n > wirc_data.source_list.append(wircpol_source([y,x],slit_pos,wirc_data.n_sources+1) where slit_pos is '0','1','2' or slitless. \n > wirc_data.n_sources += 1 ")
            self.source_list.append(wircpol_source([1063,1027],'1',self.n_sources+1))
            self.n_sources = 1

            self.header['NSOURCES'] = self.n_sources
        elif mode == 'spec':
            print("AUTOMATIC Identification of spec mode sources is not yet implemented. Hopefully soon.")

        elif mode == 'pol':
            if verbose:
                print('Finding sources in specpol image ...')
            locations = image_utils.locate_traces(image_fn, sky = sky, sigmalim = threshold_sigma, plot = plot, verbose = verbose, brightness_sort=brightness_sort, update_w_chi2_shift=update_w_chi2_shift, im_package = im_package, max_sources=max_sources, use_full_frame_mask=use_full_frame_mask, force_figures = force_figures, seeing = guess_seeing)

            # Number of sources
            # self.n_sources = len(locations['UL'][0])
            # self.header['NSOURCES'] = self.n_sources

            source_check = image_utils.check_traces(image_fn, locations, verbose = verbose)

            source_ok = source_check[0]
            source_brightness = source_check[1]

            source_flag = [not i for i in source_ok]

            locations['flag'] = source_flag

            source_ok_ind = [i for i, x in enumerate(source_ok) if x]

            self.n_sources = len(source_ok_ind)
            self.header['NSOURCES'] = self.n_sources

            # List of sources. Create source with wircpol_source([x,y],slit_pos,wirc_data.n_sources+1)
            print('Adding '+str(len(source_ok_ind))+' good sources to source_list.')
            for source in source_ok_ind:
                slit_dist = np.sqrt((locations['spot0'][0][source] - constants.slit_position_x)**2 + (locations['spot0'][1][source] - constants.slit_position_y)**2) # source distance to slit
                if slit_dist < 20: # if distance to slit is less than 20 pix
                    self.source_list.append(wircpol_source([locations['spot0'][1][source],locations['spot0'][0][source]], 1, source)) # assume it's in slit, and add source to list
                else: # if not
                    self.source_list.append(wircpol_source([locations['spot0'][1][source],locations['spot0'][0][source]], 'slitless', source)) # assume it's outside the slit ('slitless'), and add source to list

            # Sort the source_list with brightest source on top
            if brightness_sort:
                self.source_list = [x for _,x in sorted(zip(source_brightness,self.source_list),reverse=True)] # brightness sorted source_list

    def add_source(self, x,y, slit_pos = "slitless", update_w_chi2_shift = True, n_chi2_iters = 1, chi2_cutout_size=None,max_offset=10, verbose = False, trace_template = None,sub_bkg=False):
        """trace_template is the template you want to align the new location to
        """
        if update_w_chi2_shift:
            im = copy.deepcopy(self.full_image)

            if self.bkg_image is not None and sub_bkg:
                im -= self.bkg_image
            for i in range(n_chi2_iters):
                x, y =  image_utils.update_location_w_chi2_shift(self.full_image, x, y, self.filter_name, slit_pos = slit_pos,
                    verbose = verbose, cutout_size=chi2_cutout_size, max_offset=max_offset,trace_template = trace_template)
        self.source_list.append(wircpol_source([y,x],slit_pos,self.n_sources+1)) #where slit_pos is '0','1','2' or slitless.
        self.n_sources += 1

    def get_source_cutouts(self):
        """
        Get thumbnail cutouts for the spectra of for each source in the image.
        """

        for source in range(self.n_sources):
            self.source_list[source].get_cutouts(self.full_image, self.DQ_image, filter_name = self.filter_name, sub_bar = True)

    def mark_bad(self, reason = "A good reason"):
        self.bad_flag = True
        self.bad_reason = reason

class wircpol_source(object):
    """
    A point-source in a wircpol_data image

    Args:
        pos - [x,y] - the location in the image of the source
        slit_pos - Is it in the slit with possible values of [0,1,2,'slitless']


    Attributes:
        trace_images - An array of size [4,N,N], where n is the width of the box, and there is one image for each trace
        trace_spectra - An array of size [4,3, m], where each m-sized spectrum as a wavelength, a flux and a flux error
        pol_spectra - An array of size [3,3, m], where each m-sized spectrum as a wavelength, a flux and a flux error
        calibrated_pol_spectra - An array of size [5,3, m], where each m-sized spectrum as a wavelength, a flux and a flux error
        Q - an array of size 3,m, where each m sized stokes-Q has a wavelength, stokes Q  and Q error
        U - an array of size 3,m, where each m sized stokes-U has a wavelength, stokes U  and U error
        P - an array of size 3,m, where each m sized stokes-Q has a wavelength, P  and P error
        theta - an array of size 3,m, where each m sized stokes-Q has a wavelength, theta  and theta error
        lambda_calibrated - value of associated header["WL_CBRTD"]. designates whether wavelength has been calibrated
        polarization_compute - value of associated header["POL_CMPD"]. designates whether polarization has been computed
        spectra_extracted - value of associated header["SPC_XTRD"]. designates whether spectra has been extracted
        thumbnails_cut_out - value of associated header["THMB_CUT"]. designates whether thumbnails have been cut out


    """
    def __init__(self, pos, slit_pos, index):

        #The source position
        self.pos = pos
        #The source locationr relative to the slit
        self.slit_pos = slit_pos

        #The traces of each spectra
        self.trace_images = None
        self.trace_images_extracted = None
        self.trace_images_DQ = None
        self.trace_bkg = None

        #width and angle info
        self.spectra_widths = None
        self.spectra_angles = None

        #The source index (from the parent object)
        self.index = index

        #Extracted spectra
        self.trace_spectra = None
        self.calibrated_trace_spectra = None
        self.pol_spectra = None
        self.bbQ = None #Broadband Q
        self.bbU = None #Braodband U
        self.Q = None
        self.U = None
        self.P = None
        self.theta = None

        #source reduction status?
        self.lambda_calibrated = False #source attribute, later applied to header["WL_CBRTD"]
        self.polarization_computed = False #source attribute, later applied to header["POL_CMPD"]
        self.spectra_extracted = False #source attribute, later applied to header["SPC_XTRD"]
        self.spectra_aligned = False
        self.thumbnails_cut_out = False #source attribute, later applied to header["THMB_CUT"]


    #def get_cutouts(self, image, image_DQ, filter_name, replace_bad_pixels = True, method = 'median', box_size = 5, cutout_size = None, sub_bar=True, verbose=False):
    def get_cutouts(self, image, image_DQ, filter_name, bkg_image = None, replace_bad_pixels = True, method = 'median', \
    box_size = 5, cutout_size = None, sub_bar=True, verbose=False, ref_lib=None, get_PCA_cutouts=False, num_PCA_modes=None):
        """
        Cutout thumbnails and put them into self.trace_images
        if replace_bad_pixels = True, read teh DQ image and replace pixels with value != 0 by interpolation
        method can be 'median' or 'interpolate'


        """
        locs = [int(self.pos[0]),int(self.pos[1])]

        self.trace_images = np.array(image_utils.cutout_trace_thumbnails(image, np.expand_dims([locs, self.slit_pos],axis=0), flip=False,filter_name = filter_name,
            cutout_size= cutout_size, sub_bar = sub_bar, verbose=verbose)[0])
        try:
            self.trace_images_DQ = np.array(image_utils.cutout_trace_thumbnails(image_DQ, np.expand_dims([locs, self.slit_pos],axis=0), flip=False,filter_name = filter_name,
            cutout_size= cutout_size, sub_bar = False, verbose = verbose)[0])
        except:
            if verbose:
                print("Could not cutout data quality (DQ) thumbnails. Assuming everything is good.")
            self.trace_images_DQ = np.ndarray.astype(copy.deepcopy(self.trace_images*0),int)

        # if replace_bad_pixels:
        #     #iterate through the 4 thumbnails
        #     for i in range(len(self.trace_images)):
        #         bad_pix_map = self.trace_images_DQ[i] != 0
        #         self.trace_images[i] = calibration.replace_bad_pix_with_interpolation(self.trace_images[i], self.trace_images_DQ[i])

        if replace_bad_pixels:
            #check method
            if method == 'interpolate':
                #iterate through the 4 thumbnails
                for i in range(len(self.trace_images)):
                    bad_pix_map = self.trace_images_DQ[i].astype(bool)
                    self.trace_images[i] = calibration.replace_bad_pix_with_interpolation(self.trace_images[i], self.trace_images_DQ[i])
                    # except:
                    #     print("Cannot replace bad_pixels if the DQ image doesn't exist.")
            elif method == 'median':
                #iterate through the 4 thumbnails
                for i in range(len(self.trace_images)):
                    bad_pix_map = self.trace_images_DQ[i].astype(bool)
                    self.trace_images[i] = calibration.cleanBadPix(self.trace_images[i], bad_pix_map, replacement_box = box_size)
                # except:
                #     print("Cannot replace bad_pixels if the DQ image doesn't exist.")

        self.thumbnails_cut_out = True #source attribute, later applied to header["THMB_CUT"]

        #Deal with background frame
        if get_PCA_cutouts and bkg_image is None:
            UL_pca_cutouts = []
            LR_pca_cutouts = []
            UR_pca_cutouts = []
            LL_pca_cutouts = []

            for i in range(len(ref_lib)):
                cutouts = np.array(image_utils.cutout_trace_thumbnails(fits.getdata(ref_lib[i]), np.expand_dims([locs, self.slit_pos],axis=0), flip=False,filter_name = filter_name,   
                                        cutout_size= cutout_size, sub_bar = sub_bar, verbose=verbose)[0]) 

                UL_pca_cutouts.append(cutouts[0])
                LR_pca_cutouts.append(cutouts[1])
                UR_pca_cutouts.append(cutouts[2])
                LL_pca_cutouts.append(cutouts[3])

            _, UL_bkg = calibration.PCA_subtraction(self.trace_images[0], UL_pca_cutouts, num_PCA_modes)
            _, LR_bkg = calibration.PCA_subtraction(self.trace_images[1], LR_pca_cutouts, num_PCA_modes) 
            _, UR_bkg = calibration.PCA_subtraction(self.trace_images[2], UR_pca_cutouts, num_PCA_modes) 
            _, LL_bkg = calibration.PCA_subtraction(self.trace_images[3], LL_pca_cutouts, num_PCA_modes)

            self.trace_bkg = np.array([UL_bkg, LR_bkg, UR_bkg, LL_bkg])

            if replace_bad_pixels: 
                #check method   
                if method == 'interpolate': 
                    #iterate through the 4 thumbnails   
                    for i in range(len(self.trace_bkg)):        
                        bad_pix_map = self.trace_images_DQ[i].astype(bool)  
                        self.trace_bkg[i] = calibration.replace_bad_pix_with_interpolation(self.trace_bkg[i], self.trace_images_DQ[i])  
                        # except:   
                        #     print("Cannot replace bad_pixels if the DQ image doesn't exist.") 
                elif method == 'median':    
                    #iterate through the 4 thumbnails   
                    for i in range(len(self.trace_bkg)):        
                        bad_pix_map = self.trace_images_DQ[i].astype(bool)  
                        self.trace_bkg[i] = calibration.cleanBadPix(self.trace_bkg[i], bad_pix_map, replacement_box = box_size)

        elif bkg_image is not None:   
            self.trace_bkg = np.array(image_utils.cutout_trace_thumbnails(bkg_image, np.expand_dims([locs, self.slit_pos],axis=0), flip=False,filter_name = filter_name,   
                                cutout_size= cutout_size, sub_bar = sub_bar, verbose=verbose)[0])   
            if replace_bad_pixels: 
                #check method   
                if method == 'interpolate': 
                    #iterate through the 4 thumbnails   
                    for i in range(len(self.trace_bkg)):        
                        bad_pix_map = self.trace_images_DQ[i].astype(bool)  
                        self.trace_bkg[i] = calibration.replace_bad_pix_with_interpolation(self.trace_bkg[i], self.trace_images_DQ[i])  
                        # except:   
                        #     print("Cannot replace bad_pixels if the DQ image doesn't exist.") 
                elif method == 'median':    
                    #iterate through the 4 thumbnails   
                    for i in range(len(self.trace_bkg)):        
                        bad_pix_map = self.trace_images_DQ[i].astype(bool)  
                        self.trace_bkg[i] = calibration.cleanBadPix(self.trace_bkg[i], bad_pix_map, replacement_box = box_size)  
        else:
            self.trace_bkg = None
 
    def plot_cutouts(self, fig_num = None, figsize=(6.4,4.8),plot_dq = False, plot_bkg_sub = False, 
                        plot_bkg=False, plot_extracted=False,origin='lower', 
                        output_name='', show=True, **kwargs):
        '''
        Plot the source cutouts

        -------
        plot_dq  - Show the dq images instead of the cutouts
        '''

        if not show:
            default_back = matplotlib.get_backend()
            plt.switch_backend('Agg')

        #Would you like to choose the specific figure that we're using?
        if fig_num is not None:
            fig = plt.figure(fig_num,figsize=figsize)
            plt.clf()
        else: #If not, then we'll make a new figure.
            fig = plt.figure(figsize=figsize)

        #What to plot?          
        if plot_dq:           
            to_plot = self.trace_images_DQ           
        elif plot_bkg_sub:  
            to_plot = self.trace_images - self.trace_bkg    
        elif plot_bkg:
            to_plot = self.trace_bkg
        elif plot_extracted: 
            to_plot = self.trace_images_extracted
        else:         
            to_plot = self.trace_images

        if to_plot is None:
            print("The thing you want plotted doesn't exist, returning")
            return
        
        texts = ['Top - Left', 'Bottom - Right', 'Top - Right', 'Bottom - Left']        
        for i in range(4):        
            ax = fig.add_subplot(1,4,i+1)             
            plt.imshow(to_plot[i,:,:], origin = origin , **kwargs)         
            plt.text(5,140, texts[i], color = 'w')             
            ax.set_yticklabels([])

        # ax = fig.add_subplot(141)

        # if plot_dq:
        #     plt.imshow(self.trace_images_DQ[0,:,:], origin=origin, **kwargs)
        # else:
        #     plt.imshow(self.trace_images[0,:,:], origin=origin, **kwargs)
        # plt.text(5,140,"Top - Left", color='w')

        # ax = fig.add_subplot(142)
        # if plot_dq:
        #     plt.imshow(self.trace_images_DQ[1,:,:], origin=origin, **kwargs)
        # else:
        #     plt.imshow(self.trace_images[1,:,:], origin=origin, **kwargs)
        # plt.text(5,140,"Bottom - Right", color='w')
        # ax.set_yticklabels([])

        # ax = fig.add_subplot(143)
        # if plot_dq:
        #     plt.imshow(self.trace_images_DQ[2,:,:], origin=origin, **kwargs)
        # else:
        #     plt.imshow(self.trace_images[2,:,:], origin=origin, **kwargs)
        # plt.text(5,140,"Top - Right", color='w')
        # ax.set_yticklabels([])

        # ax = fig.add_subplot(144)
        # if plot_dq:
        #     plt.imshow(self.trace_images_DQ[3,:,:], origin=origin, **kwargs)
        # else:
        #     plt.imshow(self.trace_images[3,:,:], origin=origin, **kwargs)
        # plt.text(5,140,"Bottom - Left", color='w')
        # ax.set_yticklab gels([])

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.90, 0.38, 0.03, 0.24])
        plt.colorbar(cax = cbar_ax)

        if output_name != '':
            plt.savefig(output_name, format='png')
        if show:
            plt.show()
        else:
            plt.switch_backend(default_back)

    def generate_cutout_backgrounds(self,update=False,method='median',mask_diag = True):
        '''
        Generate or update cutout backgrounds.
    
        Currently the only support method is 'median'
        '''
        if self.trace_bkg is None:
            self.trace_bkg=[None,None,None,None]
    
        #Cycle through the four traces
        for i in range(4):
            #Get the median
            if update:
                if self.trace_bkg[i] is not None:
                    to_update = self.trace_images[i,:,:]-self.trace_bkg[i]
                    if mask_diag:
                        mask = wircpol_masks.makeDiagMask(to_update.shape[0],30)
                        mask = np.logical_not(mask)
                        mask = mask[::-1]
                        if i < 2:
                            mask = mask[:,::-1]
                        to_update = to_update*mask
                    
                        # plt.figure()
                        # plt.imshow(to_update,origin='lower',vmin=-100,vmax=100,cmap='RdBu')
                    to_update[to_update == 0.00] = np.nan
                    mn,md,std = sigma_clipped_stats(to_update[np.isfinite(to_update)], sigma=3.0)
                    # print(md)
                    # fig,axes = plt.subplots(1,2)
                    # axes[0].imshow(to_update,origin='lower',vmin=-100,vmax=100,cmap='RdBu')
                    # axes[1].imshow(to_update-md,origin='lower',vmin=-100,vmax=100,cmap='RdBu')
                    
                    self.trace_bkg[i] += md
                else:
                    print("Can't update if there isn't already a trace_bkg")
                    to_update = self.trace_images[i,:,:]
                    if mask_diag:
                        mask = wircpol_masks.makeDiagMask(to_update.shape[0],30)
                        mask = np.logical_not(mask)
                        mask = mask[::-1]
                        if i < 2:
                            mask = mask[:,::-1]
                        to_update = to_update*mask
                        to_update[to_update == 0.00] = np.nan
                    mn,md,std = sigma_clipped_stats(to_update, sigma=3.0)
                    self.trace_bkg[i] = md
            else:
                to_update = self.trace_images[i,:,:]
                if mask_diag:
                    mask = wircpol_masks.makeDiagMask(to_update.shape[0],30)
                    mask = np.logical_not(mask)
                    mask = mask[::-1]
                    if i < 2:
                        mask = mask[:,::-1]
                    to_update = to_update*mask
                    to_update[to_update == 0.00] = np.nan
                mn,md,std = sigma_clipped_stats(to_update, sigma=3.0)
                self.trace_bkg[i] = self.trace_images[i,:,:]*0. + md
        
        self.trace_bkg = np.array(self.trace_bkg)
        #Done!

    def plot_extracted_cutouts(self, output_name='', show=True, **kwargs):

        if not show:
            default_back = matplotlib.get_backend()
            plt.switch_backend('Agg')

        fig = plt.figure(figsize = (12,8))

        ax = fig.add_subplot(141)
        plt.imshow(self.trace_images_extracted[0,:,:],vmin=np.percentile(self.trace_images_extracted[0,:,:],5),vmax=np.percentile(self.trace_images_extracted[0,:,:],99), **kwargs)
        plt.text(5,275,"Top - Left", color='w')


        ax = fig.add_subplot(142)
        plt.imshow(self.trace_images_extracted[1,:,:],vmin=np.percentile(self.trace_images_extracted[0,:,:],5),vmax=np.percentile(self.trace_images_extracted[0,:,:],99), **kwargs)
        plt.text(5,275,"Bottom - Right", color='w')
        ax.set_yticklabels([])

        ax = fig.add_subplot(143)
        plt.imshow(self.trace_images_extracted[2,:,:],vmin=np.percentile(self.trace_images_extracted[0,:,:],5),vmax=np.percentile(self.trace_images_extracted[0,:,:],99), **kwargs)
        plt.text(5,275,"Top - Right", color='w')
        ax.set_yticklabels([])

        ax = fig.add_subplot(144)
        plt.imshow(self.trace_images_extracted[3,:,:],vmin=np.percentile(self.trace_images_extracted[0,:,:],5),vmax=np.percentile(self.trace_images_extracted[0,:,:],99), **kwargs)
        plt.text(5,275,"Bottom - Left", color='w')
        ax.set_yticklabels([])

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.38, 0.03, 0.24])
        plt.colorbar(cax = cbar_ax)

        if output_name != '':
            plt.savefig(output_name, format='png')
        if show:
            plt.show()
        else:
            plt.switch_backend(default_back)

    def clean_cutouts_for_cosmic_rays(self,nsig=10, method='lacosmic'):
        '''
        Wrapper for the image_utils function clean_thumbnails_for_cosmicrays. Replaces the trace_image_DQs.
        '''

        self.trace_images, self.trace_images_DQ = image_utils.clean_thumbnails_for_cosmicrays(self.trace_images,
            thumbnails_dq=self.trace_images_DQ, nsig=nsig, method=method)

    
    def find_traces(self,trace_angles=None,plot_findTrace=False,diag_mask=True,
                    mode='pol',diag_mask_width = 70):
        """
        Get the trance angles and widths

        Args:
            trace_angles ([type], optional): [description]. Defaults to None.
            plot_findTrace (bool, optional): [description]. Defaults to False.
            diag_mask (bool, optional): [description]. Defaults to True.
            mode (str, optional): [description]. Defaults to 'pol'.
            diag_mask_width (int, optional): [description]. Defaults to 70.
        """                    


        thumbnails_copy = copy.deepcopy(self.trace_images)

        if self.trace_images_DQ is not None:
            DQ_copy = copy.deepcopy(self.trace_images_DQ)
        else:
            DQ_copy = np.zeros(thumbnails_copy.shape)

        if self.trace_bkg is not None:
            bkg_copy = copy.deepcopy(self.trace_bkg)
        else:
            bkg_copy = np.zeros(thumbnails_copy.shape)

        
        
        #Flip some of the traces around.
        if mode=='pol': 
            thumbnails_copy[1,:,:] = thumbnails_copy[1,-1::-1, -1::-1] #flip y, x. Bottom-right
            thumbnails_copy[2,:,:] = thumbnails_copy[2,:,-1::-1] #flip x #Top-right
            thumbnails_copy[3,:,:] = thumbnails_copy[3,-1::-1, :] #flip y #Bottom-left

            DQ_copy[1,:,:] = DQ_copy[1,-1::-1, -1::-1] #flip y, x. Bottom-right
            DQ_copy[2,:,:] = DQ_copy[2,:,-1::-1] #flip x #Top-right
            DQ_copy[3,:,:] = DQ_copy[3,-1::-1, :] #flip y #Bottom-left

            bkg_copy[1,:,:] = bkg_copy[1,-1::-1,-1::-1]
            bkg_copy[2,:,:] = bkg_copy[2,:,-1::-1]
            bkg_copy[3,:,:] = bkg_copy[3,-1::-1,:]

        trace_titles=["Top-Left", "Bottom-Right", "Top-Right", "Bottom-left"]
        
        #Cycle through the traces
        # 
        if trace_angles is None:
            trace_angle = [None,None,None,None]
        else:
            trace_angle = trace_angles

        #The measured trace widths
        widths = np.zeros([4])
        
        for j in range(4):            
            
            trace_title = trace_titles[j]

            #copy thumbnails
            thumbnail = thumbnails_copy[j,:,:]
            bkg = bkg_copy[j,:,:]
            DQ = DQ_copy[j,:,:]

            if trace_angle[j] is None:
                raw, trace, trace_width, measured_trace_angle = spec_utils.findTrace(thumbnail - bkg, poly_order = 1, weighted=True, 
                    plot = plot_findTrace, diag_mask=diag_mask, mode=mode,diag_mask_width=diag_mask_width) #linear fit to the trace

                trace_angle[j] = measured_trace_angle

            else: #angle given, still run the fit but only record width

                raw, trace, trace_width, measured_trace_angle = spec_utils.findTrace(thumbnail - bkg, poly_order = 1, weighted = True, plot = plot_findTrace, diag_mask = diag_mask, mode = mode,
                                                            fractional_fit_type = None,diag_mask_width=diag_mask_width) #for quickly getting trace width, which is needed to determine extraction range
                # widths += [trace_width]
           
            if diag_mask and mode=='pol':
                mask = spec_utils.makeDiagMask(np.shape(thumbnail)[0],diag_mask_width)
                thumbnail[~mask] = 0.0
                bkg[~mask] = 0.0
                DQ_copy[j,:,:][~mask] = 0.0
                # plt.imshow(thumbnail)
                # print(trace)
                # print(trace_width)
                # print(measured_trace_angle)

                if trace_width is None:
                    trace_width = 7
                mask = spec_utils.make_mask_from_findTrace(trace, 3*trace_width, measured_trace_angle)

            #rotate the spectrum here. rotation axis is the middle of the image
            width_thumbnail = thumbnail.shape[0]
            sub_rotated = spec_utils.frame_rotate(thumbnail-bkg, trace_angle[j]+180,cxy=[width_thumbnail/2,width_thumbnail/2])

            real_width = image_utils.traceWidth_after_rotation(sub_rotated)

            widths[j]= real_width
        self.spectra_widths = widths
        self.spectra_angles = trace_angle


    
    def extract_spectra(self, method = 'optimal_extraction', niter = 2, sig_clip = 5, 
                        bad_pix_masking = 1, width_scale=1., diag_mask=False, diag_mask_width = 70, 
                        trace_angle = None, mode = 'pol', 
                        spatial_sigma = 5, fixed_width = None,
                        use_DQ=True, debug_DQ=False,
                        spatial_smooth=1, spectral_smooth=10, fractional_fit_type = False,
                        verbose=True, plot_result = False,  plot_optimal_extraction = False, plot_findTrace = False, 
                        align = True):
        """
        *method:        method for spectral extraction. Choices are
        (i) skimage: this is just the profile_line method from skimage. Order for interpolation
        is in skimage_order parameter (fast).
        (ii) weightedSum: this is 2D weighted sum assuming Gaussian profile. Multiply the PSF with data
        and sum for each location along the dispersion direction (fast). The width of the Gaussian
        is based on the measured value by 'findTrace'. One can adjust this using the parameter 'width_scale'.
        (iii) fit_across_trace: this method rotates the trace, loops along the dispersion direction, and fit a profile in the
        spatial direction. The fit function is either 'Moffat' or 'Gaussian'. One can also
        select how to extract flux: by summing the fitted model, or the data weighted by the model.
        ('model_sum' vs 'weighted_sum'). These are in 'fitfunction' and 'sum_method' parameters.
        box_size determine how many columns of pixel we will use. poly_order is the order of polynomial used to
        fit the background. trace_angle is the angle to rotate the cutout so it's aligned with the pixel grid.
        If None, it uses value from fitTraces.

        """
        if verbose:
            print("Performing Spectral Extraction for source {}".format(self.index))

        #call spec_extraction to actually extract spectra
        # spectra, spectra_std, spectra_widths, spectra_angles, thumbnail_to_extract = spec_utils.spec_extraction(self.trace_images, plot=plot, bkg_poly_order = bkg_poly_order, 
        #     bkg_thumbnails = self.trace_bkg, plot_optimal_extraction = plot_optimal_extraction , plot_findTrace = plot_findTrace, method=method,
        #     width_scale=width_scale, diag_mask=diag_mask, niter = niter, sig_clip = sig_clip, bad_pix_masking = bad_pix_masking, fitfunction = fitfunction,
        #     sum_method = sum_method, box_size = box_size, poly_order = poly_order, trace_angle = trace_angle, verbose=verbose, DQ_thumbnails=self.trace_images_DQ,
        #     use_DQ = use_DQ, debug_DQ=debug_DQ,spatial_smooth=spatial_smooth,spectral_smooth=spectral_smooth,spatial_sigma=spatial_sigma, fixed_width = fixed_width)

        spectra, spectra_std, spectra_widths, spectra_angles, thumbnail_to_extract =  \
            spec_utils.spec_extraction(self.trace_images, bkg_thumbnails = self.trace_bkg, method = method, niter = niter, sig_clip = sig_clip, 
                bad_pix_masking = bad_pix_masking, width_scale=width_scale, diag_mask = diag_mask, diag_mask_width = diag_mask_width, trace_angle = trace_angle, mode = mode, 
                spatial_sigma = spatial_sigma, fixed_width = fixed_width, 
                DQ_thumbnails = self.trace_images_DQ, use_DQ=use_DQ, debug_DQ=debug_DQ, 
                spatial_smooth=spatial_smooth, spectral_smooth=spectral_smooth, fractional_fit_type=fractional_fit_type, 
                verbose = verbose, plot_optimal_extraction = plot_optimal_extraction, plot_findTrace = plot_findTrace, plot_result = plot_result)
        #if align, then call align_set_of_traces to align 4 traces to the Q plus, using cross-correlation

        if align:
            spectra = spec_utils.align_set_of_traces(spectra, spectra[0])

        spectra_length = spectra.shape[1]

        #set values
        self.trace_spectra = np.zeros((4,3,spectra_length))
        self.trace_spectra[:,0,:] = np.arange(spectra_length) #The wavelength axis, to be calibrated later.
        self.trace_spectra[:,1,:] = spectra
        self.trace_spectra[:,2,:] = spectra_std

        self.extract_method = method

        self.spectra_extracted = True #source attribute, later applied to header["SPC_XTRD"]
        self.spectra_aligned = align

        self.trace_images_extracted = thumbnail_to_extract #these are thumbnails on which the extraction was done

        self.spectra_widths = spectra_widths
        self.spectra_angles = spectra_angles

    def align_spectra(self, lowcut = 0, highcut=-1, big_filt_sz = 30, little_filt_sz = 3, x_start = [0.,1.,0.,0.,0.]):
        '''
        This function attemps to align the four spectra by minimizing the residuals of subtracting the two orthogonal
        traces after applying a 2nd-order polynomial wavelength shift and scaling the fluxes to match each other.
        Only the 2nd-order polynomical shift is then used
        '''
        if self.lambda_calibrated:
            lowcut = np.where(self.trace_spectra[0,0,:] > 1.20)[0][0]
            highcut = np.where(self.trace_spectra[0,0,:] > 1.30)[0][0]
            print(highcut)

        self.trace_spectra = spec_utils.align_spectra(self.trace_spectra, lowcut=lowcut,
            highcut=highcut, big_filt_sz=big_filt_sz, little_filt_sz=little_filt_sz, x_start=x_start)

    def rough_lambda_calibration(self, filter_name="J", method=1, lowcut=0, highcut=-1):
        #Rough wavelength calibration. Will have to get better later!

        """

        lowcut - The lowest pixel to use in the traces
        highcut - The highest pixel to use in the traces

        #TODO: It would be good to have lowcut and highcut only apply to the calculation, and not affect the data at this point (I think)

        """

        #Calibrate using a spectrum we trust (at least kind of) - This is because this method isn't as reliable for BDs
        if method ==3: 
            ## Read in a spectrum for spectral alignment
            cal_spec = np.load("/scr/data/Other_Files/Tinyanont_2018_Fig6/HD109055_set2.npy")[:,:,84:-60]
            # First we get the wavelength calibration
            wvs = spec_utils.rough_wavelength_calibration_v2(cal_spec[0,1], 'J')
            wvs = np.pad(wvs,2,mode='edge')
            
            #Now we cross-correlate with the know spectrum
            self.trace_spectra = spec_utils.align_spectral_cube(self.trace_spectra[None],ref_trace=cal_spec[0,1])[0]
            self.trace_spectra[0,0,:] = wvs
            self.trace_spectra[1,0,:] = wvs
            self.trace_spectra[2,0,:] = wvs
            self.trace_spectra[3,0,:] = wvs
             

        aligned = self.spectra_aligned
        
        if aligned: #do wavelength calibration to Qp, then apply it to everything else
            if method == 1:
                self.trace_spectra[0,0,:] = spec_utils.rough_wavelength_calibration_v1(self.trace_spectra[0,1,:], filter_name)
                self.trace_spectra[1,0,:] = self.trace_spectra[0,0,:]
                self.trace_spectra[2,0,:] = self.trace_spectra[0,0,:]
                self.trace_spectra[3,0,:] = self.trace_spectra[0,0,:]
            if method == 2:
                self.trace_spectra[0,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[0,1,:], filter_name, lowcut=lowcut, highcut=highcut)
                self.trace_spectra[1,0,:] = self.trace_spectra[0,0,:]
                self.trace_spectra[2,0,:] = self.trace_spectra[0,0,:]
                self.trace_spectra[3,0,:] = self.trace_spectra[0,0,:]

            
        else:
            if method == 1:
                self.trace_spectra[0,0,:] = spec_utils.rough_wavelength_calibration_v1(self.trace_spectra[0,1,:], filter_name)
                self.trace_spectra[1,0,:] = spec_utils.rough_wavelength_calibration_v1(self.trace_spectra[1,1,:], filter_name)
                self.trace_spectra[2,0,:] = spec_utils.rough_wavelength_calibration_v1(self.trace_spectra[2,1,:], filter_name)
                self.trace_spectra[3,0,:] = spec_utils.rough_wavelength_calibration_v1(self.trace_spectra[3,1,:], filter_name)


            elif method == 2:
                self.trace_spectra[0,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[0,1,:], filter_name, lowcut=lowcut, highcut=highcut)
                self.trace_spectra[1,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[1,1,:], filter_name, lowcut=lowcut, highcut=highcut)
                self.trace_spectra[2,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[2,1,:], filter_name, lowcut=lowcut, highcut=highcut)
                self.trace_spectra[3,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[3,1,:], filter_name, lowcut=lowcut, highcut=highcut)

        if method == 3:
            self.trace_spectra[0,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[0,1,:], filter_name, lowcut=lowcut, highcut=highcut)

        self.lambda_calibrated = True #source attribute, later applied to header["WL_CBRTD"]

    def rough_lambda_and_filter_calibration(self, filter_name = "J", verbose = False, plot_alignment = False, tilt_angle=45, source_compensation=False):

        if filter_name == "H":
            print("H-band not yet supported")

        self.calibrated_trace_spectra = spec_utils.rough_lambda_and_filter_calibration(self.trace_spectra, self.spectra_widths, self.pos[1],
            self.pos[0],verbose=verbose, plot_alignment = plot_alignment, tilt_angle = tilt_angle, source_compensation = source_compensation)
        self.lambda_calibrated = True

    def compute_polarization(self, cutmin=0, cutmax=-1):


        ## The output of the computer_polarization function is based on the initial assumptions on Q and U, which we know are wrong.
        ## We will keep the compute_polarization the same, but change it when putting it into the source object.
        wlQp, q, dq, wlUp,u, du = spec_utils.compute_polarization(self.trace_spectra, cutmin=cutmin, cutmax = cutmax)

        pol_spectra_length = q.shape[0]

        self.Q = np.zeros([3,pol_spectra_length])
        self.U = np.zeros([3,pol_spectra_length])

        ##### OLD VERSION ######
        # self.Q[0,:] = wlQp
        # self.Q[1,:] = q
        # self.Q[2,:] = dq

        # self.U[0,:] = wlUp
        # self.U[1,:] = u
        # self.U[2,:] = du
        #########################

        ##### NEW VERSION #####
        # Based on Kaew's Twilight Measurements March 2018
        self.Q[0,:] = wlUp
        self.Q[1,:] = -u
        self.Q[2,:] = du

        self.U[0,:] = wlQp
        self.U[1,:] = -q
        self.U[2,:] = dq
        #######################

        self.polarization_computed = True #source attribute, later applied to header["POL_CMPD"]

    def plot_trace_spectra(self, with_errors = False, filter_name="J", smooth_size = 1, smooth_ker = 'Gaussian', fig_num = None, figsize=(6.4,4.8), output_name='', show=True,\
                xlow = None, xhigh = None, ylow=None, yhigh = None,**kwargs):

        if not show:
            default_back = matplotlib.get_backend()
            plt.switch_backend('Agg')

        #Would you like to choose the specific figure that we're using?
        if fig_num is not None:
            fig = plt.figure(fig_num,figsize=figsize)
            plt.clf()
        else:
            fig = plt.figure(figsize=figsize)

        labels = ["Top-Left", "Bottom-Right", "Top-Right", "Bottom-left"]
        for i in range(4):
            wl = self.trace_spectra[i,0,:]
            flux = self.trace_spectra[i,1,:]
            err = self.trace_spectra[i,2,:]
            if smooth_size > 1:
                flux = spec_utils.smooth_spectra(flux, smooth_ker, smooth_size)
            if with_errors:
                plt.errorbar(wl, flux,yerr = err, label=labels[i], **kwargs)

            else:
                plt.plot(wl, flux, label=labels[i], **kwargs)
            plt.draw()

        plt.ylabel("Flux [ADU]")

        if ylow is not None and yhigh is not None:  
            plt.ylim([ylow,yhigh])

        if self.lambda_calibrated: #plot is not perfectly the same
            plt.xlabel("Wavelength [um]")
            if xlow is not None and xhigh is not None:              
                plt.xlim([xlow,xhigh])  
            else:   
                plt.xlim([1.17,1.32]) #wavelength display range -- Where the J-band filter has  > 80% throughput
        else:
            plt.xlabel("Wavelength [Arbitrary Unit]")
            if xlow is not None and xhigh is not None:             
                plt.xlim([xlow,xhigh])  
            else:   
                plt.xlim([0,225]) #arbitrary unit wavelength display range  
    

        plt.legend()

        if output_name != '':
            plt.savefig(output_name, format='png')

        if not show:
            plt.switch_backend(default_back)

    def plot_Q_and_U(self, with_errors = False, figsize=(7,7), xlow=1.15, xhigh=1.35, ylow=-0.2, yhigh=0.2, output_name='', show=True, **kwargs):

        if not show:
            default_back = matplotlib.get_backend()
            plt.switch_backend('Agg')

        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.set_title("Stokes Q")
        ax2.set_title("Stokes U")

        if with_errors:
            ax1.errorbar(self.Q[0,:], self.Q[1,:],yerr=self.Q[2,:], **kwargs)
            ax2.errorbar(self.U[0,:], self.U[1,:],yerr=self.U[2,:], **kwargs)
        else:
            ax1.plot(self.Q[0,:], self.Q[1,:], **kwargs)
            ax2.plot(self.U[0,:], self.U[1,:], **kwargs)

        ax1.set_ylim(ylow,yhigh)
        ax2.set_ylim(ylow,yhigh)

        if self.lambda_calibrated:
            ax1.set_xlabel("Wavelength [um]")
            ax2.set_xlabel("Wavelength [um]")
            ax1.set_xlim(xlow,xhigh)
            ax2.set_xlim(xlow,xhigh)
        else:
            ax1.set_xlabel("Wavelength [Arbitrary Units]")
            ax2.set_xlabel("Wavelength [Arbitrary Units]")

        if output_name != '':
            plt.savefig(output_name, format='png')

        if not show:
            plt.switch_backend(default_back)

    def get_broadband_polarization(self, mode ='from_spectra', xlow=0, xhigh=-1, weighted=False, bkg=None, exp_time=None, stretch=1, plot=False, 
                                    savefig=None, verbose=True, trace_len=90, trace_wid=8, fit_trace=False):
        '''
        A function to measure the broadband polarization from a source. 
        Modes: 
        from_spectra        - Sum the polarization in each trace (between xlow and xhigh) and measure the broadband polarization. 
        aperture_photometry - Use photutils to measure the aperture photometry. 
        '''
        if mode == "from_spectra":
            bb_traces = np.zeros([4,3])
            for i in range(4):
                #Do we actualy want the option of weighted mean?
                if weighted:
                    bb_traces[i,0] = np.average(self.trace_spectra[i,0,xlow:xhigh]) #Don't weight the wavelength
                    bb_traces[i,1] = np.average(self.trace_spectra[i,1,xlow:xhigh], weights=1/self.trace_spectra[i,2,xlow:xhigh])
                    bb_traces[i,2] = np.sqrt(np.sum(self.trace_spectra[i,2,xlow:xhigh]**2)/np.size(self.trace_spectra[i,2,xlow:xhigh])**2) #This isn't the correct error formula for weighted means
                else:
                    bb_traces[i,0] = np.average(self.trace_spectra[i,0,xlow:xhigh]) 
                    bb_traces[i,1] = np.average(self.trace_spectra[i,1,xlow:xhigh])
                    bb_traces[i,2] = np.sqrt(np.sum(self.trace_spectra[i,2,xlow:xhigh]**2)/np.size(self.trace_spectra[i,2,xlow:xhigh])**2)

            
            ## The calculations of bbQ and bbU are based on the initial assumptions on Q and U, which we know are wrong. 
            ## We'll keep the the same for now, but addjust the input to the source object properties. 

            bbQ = (bb_traces[0,1]-bb_traces[1,1])/(bb_traces[0,1]+bb_traces[1,1])
            #If f = A/B
            #sigma_f = f*sqrt ( (sigma_A/A)**2 + (sigma_B/B)**2 )
            bbQ_err = bbQ*np.sqrt( (bb_traces[0,2]**2 + bb_traces[1,2]**2)/(bb_traces[0,1]-bb_traces[1,1])**2 + (bb_traces[0,2]**2 + bb_traces[1,2]**2)/(bb_traces[0,1]+bb_traces[1,1])**2)

            bbU = (bb_traces[2,1]-bb_traces[3,1])/(bb_traces[2,1]+bb_traces[3,1])

            bbU_err = bbU*np.sqrt( (bb_traces[2,2]**2 + bb_traces[3,2]**2)/(bb_traces[2,1]-bb_traces[3,1])**2 + (bb_traces[2,2]**2 + bb_traces[3,2]**2)/(bb_traces[2,1]+bb_traces[3,1])**2)


            ###### OLD VERSION ###### - Do not use March 27. 2018 MMB
            # self.bbQ = [bb_traces[0,0], bbQ, bbQ_err] #Return, [wavelength, flux, error]
            # self.bbU = [bb_traces[2,0], bbU, bbU_err]
            #########################

            ###### NEW VERSION ######
            #Based on Kaew's twilight measurements March 27 2018
            self.bbQ = [bb_traces[2,0], -bbU, bbU_err] #Return, [wavelength, flux, error]
            self.bbU = [bb_traces[0,0], -bbQ, bbQ_err]
            #########################
        elif mode =="aperture_photometry":
            # print("Not yet implemented")
            
            from astropy.stats import sigma_clipped_stats
            from photutils import make_source_mask

            
            bb_flux = []
            #First estimate the background
            for i in range(4):
                _, median, _ = sigma_clipped_stats(self.trace_images[i,:,:], sigma=3.0)
                this_trace_image = self.trace_images[i,:,:] - median
                mask = make_source_mask(this_trace_image, snr=3, npixels=200, dilate_size=5)
                bb_flux.append(np.sum(this_trace_image[mask]))

            self.bbQ = [None, -(bb_flux[2]-bb_flux[3])/(bb_flux[2]+bb_flux[3]), None] #Return, [wavelength, flux, error]
            self.bbU = [None, -(bb_flux[0]-bb_flux[1])/(bb_flux[0]+bb_flux[1]), None]

        elif mode=="aperture_photometry_new":
            
            apertures, total_flux, SNR = source_utils.fit_aperture(self, exp_time=exp_time, stretch=stretch, plot=plot, savefig=savefig, verbose=verbose,
                                                                    trace_len=trace_len, trace_wid=trace_wid, fit_trace=fit_trace)

            bb_flux = []

            for i in range(4):
                bb_flux.append(np.nansum(apertures[i]))

            self.bbQ = [None, -(bb_flux[2]-bb_flux[3])/(bb_flux[2]+bb_flux[3]), None] #Return, [wavelength, flux, error]
            self.bbU = [None, -(bb_flux[0]-bb_flux[1])/(bb_flux[0]+bb_flux[1]), None]

            self.total_flux = total_flux
            self.SNR = SNR
            
        else:
            print("Only 'from_spectra' and 'aperture_photometry' modes are supported. Returning.")

    def smooth_cutouts(self,method='gaussian',width=3):
        '''
        A function to smooth the thumbnails
        '''
        image_utils.smooth_cutouts(self.trace_images,method=method,width=width)

    def _get_bkg_cutouts(self,bkg_image,filter_name,replace_bad_pixels = True, method = 'median', 
        box_size = 5,cutout_size = None,sub_bar=True, verbose=False):
        """
        DEPRECATED
        Cutout thumbnails from a background image in self.bkg_image and put them into self.trace_bkg
        if replace_bad_pixels = True, read teh DQ image and replace pixels with value != 0 by interpolation
        method can be 'median' or 'interpolate'
        """

        locs = [int(self.pos[0]),int(self.pos[1])]

        self.trace_bkg = np.array(image_utils.cutout_trace_thumbnails(bkg_image, np.expand_dims([locs, self.slit_pos],axis=0), flip=False,
            filter_name = filter_name, cutout_size= cutout_size, sub_bar = sub_bar, verbose=verbose)[0])   


        if replace_bad_pixels:
            #check method
            if method == 'interpolate':
                #iterate through the 4 thumbnails
                for i in range(len(self.trace_bkg)):
                    bad_pix_map = self.trace_images_DQ[i].astype(bool)
                    self.trace_bkg[i] = calibration.replace_bad_pix_with_interpolation(self.trace_bkg[i], self.trace_images_DQ[i])
                    # except:
                    #     print("Cannot replace bad_pixels if the DQ image doesn't exist.")
            elif method == 'median':
                #iterate through the 4 thumbnails
                for i in range(len(self.trace_bkg)):
                    bad_pix_map = self.trace_images_DQ[i].astype(bool)
                    self.trace_bkg[i] = calibration.cleanBadPix(self.trace_bkg[i], bad_pix_map, replacement_box = box_size)
                # except:
                #     print("Cannot replace bad_pixels if the DQ image doesn't exist.")

class wircspec_source(object):
    """
    A point-source in a a wircspec_data image

    Args:
        pos - [x,y] - the location in the image of the source
    Attributes:
        trace_images - An array of size [N,N], where n is the width of the box, and there is one image for each trace
        raw_spectrum - An array of size [3, m], where each m-sized spectrum as a wavelength, a flux and a flux error
        calibrated_spectrum - An array of size [3, m], where each m-sized spectrum as a wavelength, a flux and a flux error

        lambda_calibrated - value of associated header["WL_CBRTD"]. designates whether wavelength has been calibrated
        spectra_extracted - value of associated header["SPC_XTRD"]. designates whether spectra has been extracted
        thumbnails_cut_out - value of associated header["THMB_CUT"]. designates whether thumbnails have been cut out
    """
    def __init__(self, pos, slit_pos, index):

        #The source position
        self.pos = pos

        #The source locationr relative to the slit
        self.slit_pos = slit_pos

        #The traces of each spectra
        self.trace_images = None

        #The source index (from the parent object)
        self.index = index

        #Extracted spectra
        self.trace_spectra = None

        #source reduction status?
        self.lambda_calibrated = False #source attribute, later applied to header["WL_CBRTD"]
        self.spectra_extracted = False #source attribute, later applied to header["SPC_XTRD"]
        self.spectra_aligned = False
        self.thumbnails_cut_out = False #source attribute, later applied to header["THMB_CUT"]

    # def get_cutouts(self, image, filter_name, sub_bar=True, cutout_size=80, flip=False):
    #     """
    #     Cutout thumbnails and put them into self.trace_images.  Thumbnails are placed directly on input positions.
    #     Optional: set cutout size.
    #     """

    #     self.trace_images = np.array(image_utils.cutout_trace_thumbnails(image, np.expand_dims([self.pos, self.slit_pos],axis=0),flip=flip, \
    #                                             filter_name = filter_name, sub_bar = sub_bar,mode = 'spec', cutout_size = cutout_size)[0])

    #     self.thumbnails_cut_out = True #source attribute, later applied to header["THMB_CUT"]


    def get_cutouts(self, image, filter_name, image_DQ = None, method = 'median', box_size = 5, sub_bar=True, cutout_size=None, flip=False, verbose=False):
        """
        Cutout thumbnails and put them into self.trace_images
        if replace_bad_pixels = True, read the DQ image and replace pixels with value != 0 by method = 'interpolate' or 'median'

        """

        locs = [int(self.pos[0]),int(self.pos[1])]
        self.trace_images = np.array(image_utils.cutout_trace_thumbnails(image, np.expand_dims([locs, self.slit_pos],axis=0), flip=flip,filter_name = filter_name, sub_bar = sub_bar, mode = 'spec', cutout_size = cutout_size, verbose=verbose)[0])
        if image_DQ is not None:

            try:
                self.trace_images_DQ = np.array(image_utils.cutout_trace_thumbnails(image_DQ, np.expand_dims([locs, self.slit_pos],axis=0), flip=flip,\
                                        filter_name = filter_name, sub_bar = sub_bar, mode = 'spec', cutout_size = cutout_size, verbose = verbose)[0])
            except:
                if verbose:
                    print("Could not cutout data quality (DQ) thumbnails. Assuming everything is good.")
                self.trace_images_DQ = np.ndarray.astype(copy.deepcopy(self.trace_images*0),int)
            #check method
            if method == 'interpolate':
                #iterate through the 4 thumbnails
                for i in range(len(self.trace_images)):
                    bad_pix_map = self.trace_images_DQ[i].astype(bool)
                    self.trace_images[i] = calibration.replace_bad_pix_with_interpolation(self.trace_images[i], self.trace_images_DQ[i])

            elif method == 'median':
                #iterate through the 4 thumbnails
                for i in range(len(self.trace_images)):
                    bad_pix_map = self.trace_images_DQ[i].astype(bool)
                    self.trace_images[i] = calibration.cleanBadPix(self.trace_images[i], bad_pix_map, replacement_box = box_size)
        self.thumbnails_cut_out = True #source attribute, later applied to header["THMB_CUT"]

    def plot_cutouts(self, **kwargs):

        fig = plt.figure(figsize = (12,8))

        ax = fig.add_subplot(141)
        plt.imshow(self.trace_images[0,:,:], **kwargs)

        #fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.90, 0.38, 0.03, 0.24])
        plt.colorbar(cax = cbar_ax)
        plt.show()

    def extract_spectra(self, sub_background = False, plot=False, plot_optimal_extraction = False, plot_findTrace = False,
                         method = 'optimal_extraction', bad_pix_masking = 1, width_scale=1., diag_mask=False, filter_bkg_size = None,\
                        trace_angle = None, align = True, verbose = True,
                        fractional_fit_type = None,  spatial_sigma = 3):
        """
        *method:        method for spectral extraction. Choices are
        (i) skimage: this is just the profile_line method from skimage. Order for interpolation
        is in skimage_order parameter (fast).
        (ii) weightedSum: this is 2D weighted sum assuming Gaussian profile. Multiply the PSF with data
        and sum for each location along the dispersion direction (fast). The width of the Gaussian
        is based on the measured value by 'findTrace'. One can adjust this using the parameter 'width_scale'.
        (iii) fit_across_trace: this method rotates the trace, loops along the dispersion direction, and fit a profile in the
        spatial direction. The fit function is either 'Moffat' or 'Gaussian'. One can also
        select how to extract flux: by summing the fitted model, or the data weighted by the model.
        ('model_sum' vs 'weighted_sum'). These are in 'fitfunction' and 'sum_method' parameters.
        box_size determine how many columns of pixel we will use. poly_order is the order of polynomial used to
        fit the background.
        trace_angle is the angle used to rotate the spectral trace for the fit_across_trace method; None uses the
        measured angle for each individual spectrum
        """
        if verbose:
            print("Performing Spectral Extraction for source {}".format(self.index))


        #call spec_extraction to actually extract spectra
        spectra, spectra_std, spectra_widths, spectra_angles, thumbnail_to_extract= spec_utils.spec_extraction(self.trace_images, self.slit_pos,
            sub_background = sub_background,plot=plot, plot_optimal_extraction = plot_optimal_extraction , plot_findTrace = plot_findTrace,
            method=method, width_scale=width_scale, diag_mask=diag_mask, filter_bkg_size = filter_bkg_size,
            trace_angle = trace_angle, fitfunction = fitfunction, sum_method = sum_method, bad_pix_masking = bad_pix_masking,
            box_size = box_size, poly_order = poly_order,mode='spec', verbose = verbose,fractional_fit_type = fractional_fit_type,
            bkg_sub_shift_size = bkg_sub_shift_size, bkg_poly_order = bkg_poly_order, spatial_sigma = spatial_sigma)

        #if align, then call align_set_of_traces to align 4 traces to the Q plus, using cross-correlation
        #for i in spectra:
        #    plt.plot(i)
        #plt.show()
        if align:
            spectra = spec_utils.align_set_of_traces(spectra, spectra[0])
        #for i in spectra:
        #    plt.plot(i)
        #plt.show()
        spectra_length = spectra.shape[1]

        self.trace_spectra = np.zeros((1,3,spectra_length))
        self.trace_spectra[:,0,:] = np.arange(spectra_length) #The wavelength axis, to be calibrated later.
        self.trace_spectra[:,1,:] = spectra
        self.trace_spectra[:,2,:] = spectra_std

        self.spectra_extracted = True #source attribute, later applied to header["SPC_XTRD"]
        self.spectra_aligned = align

        self.trace_images_extracted = thumbnail_to_extract #these are thumbnails on which the extraction was done

        self.spectra_widths = spectra_widths
        self.spectra_angles = spectra_angles

    def rough_lambda_calibration(self, filter_name="J", method=1, lowcut=0, highcut=-1):
        #Rough wavelength calibration. Will have to get better later!

        """

        lowcut - The lowest pixel to use in the traces
        highcut - The highest pixel to use in the traces

        #TODO: It would be good to have lowcut and highcut only apply to the calculation, and not affect the data at this point (I think)

        """
        aligned = self.spectra_aligned

        if aligned:
            if method == 1:
                self.trace_spectra[0,0,:] = spec_utils.rough_wavelength_calibration_v1(self.trace_spectra[0,1,:], filter_name)
            if method == 2:
                self.trace_spectra[0,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[0,1,:], filter_name, lowcut=lowcut, highcut=highcut)

        else:
            if method == 1:
                self.trace_spectra[0,0,:] = spec_utils.rough_wavelength_calibration_v1(self.trace_spectra[0,1,:], filter_name)
            elif method == 2:
                self.trace_spectra[0,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[0,1,:], filter_name, lowcut=lowcut, highcut=highcut)

        self.lambda_calibrated = True #source attribute, later applied to header["WL_CBRTD"]


    def plot_trace_spectra(self, with_errors = False, filter_name="J", smooth_size = 1, smooth_ker = 'Gaussian', **kwargs):

        fig = plt.figure(figsize=(7,7))
        #labels = ["Top-Left", "Bottom-Right", "Top-Right", "Bottom-left"]
        for i in range(1):
            wl = self.trace_spectra[i,0,:]
            flux = self.trace_spectra[i,1,:]
            err = self.trace_spectra[i,2,:]
            if smooth_size > 1:
                flux = spec_utils.smooth_spectra(flux, smooth_ker, smooth_size)
            if with_errors:
                plt.errobar(wl, flux,yerr = err, **kwargs) #label=labels[i]

            else:
                plt.plot(wl, flux, **kwargs) #label=labels[i],

        plt.ylabel("Flux [ADU]")

        if self.lambda_calibrated: #plot is not perfectly the same
            plt.xlabel("Wavelength [um]")
            plt.xlim([1.1,1.4]) #wavelength display range
        else:
            plt.xlabel("Wavelength [Arbitrary Unit]")
            plt.xlim([0,np.size(wl)]) #arbitrary unit wavelength display range

        plt.legend()
        plt.show()

   

