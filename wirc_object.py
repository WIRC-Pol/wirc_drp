import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import wirc_drp.utils.image_utils as image_utils
import wirc_drp.utils.spec_utils as spec_utils
import wirc_drp.utils.calibration as calibration
from wirc_drp import version # For versioning (requires gitpython 'pip install gitpython')
from wirc_drp.masks import * ### Make sure that the wircpol/DRP/mask_design directory is in your Python Path!

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

    """
    def __init__(self, raw_filename=None, wirc_object_filename=None, dark_fn = None, flat_fn = None, bp_fn = None, bkg_fn = None):

        ## Load in either the raw file, or the wircpol_object file
        if raw_filename != None and wirc_object_filename != None:
            print("Can't open both a raw file and wircpol_object, ignoring the raw file and loading the wirc_object_file ")
            print("Loading a wircpol_data object from file {}".format(wirc_object_filename))
            self.load_wirc_object(wirc_object_filename)

        elif wirc_object_filename != None:
            print("Loading a wirc_data object from file {}".format(wirc_object_filename))
            self.load_wirc_object(wirc_object_filename)

        elif raw_filename != None:
            print("Creating a new wirc_data object from file {}".format(raw_filename))
            self.raw_filename = raw_filename
            hdu = fits.open(raw_filename)
            self.full_image = hdu[0].data
            self.header = hdu[0].header

            self.header['RAW_FN'] = raw_filename

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
                print("Found a {}-band filter in the header of file {}".format(filter_name,raw_filename))
            self.filter_name = filter_name

            self.calibrated = False
            self.bkg_subbed = False

            self.n_sources = 0
            self.source_list = []

            self.dark_fn = dark_fn        
            self.flat_fn = flat_fn
            self.bkg_fn = bkg_fn
            self.bp_fn = bp_fn

    def calibrate(self, clean_Bad_Pix=True, replace_nans=True, mask_bad_pixels=False):
        '''
        Apply dark and flat-field correction
        '''

        #TODO Add checks to make sure the flatnames are not none

        #Open the master dark
        master_dark_hdu = fits.open(self.dark_fn)
        master_dark = master_dark_hdu[0].data
        dark_shape = np.shape(master_dark)
        print(("Subtracting {} from each file".format(self.dark_fn)))
        dark_exp_time = master_dark_hdu[0].header['EXPTIME']

        #Open the master flat
        master_flat_hdu = fits.open(self.flat_fn)
        master_flat = master_flat_hdu[0].data
        print(("Dividing each file by {}".format(self.flat_fn)))
        dark_exp_time = master_dark_hdu[0].header['EXPTIME']

        #Open the bad pixel map
        bp_map_hdu = fits.open(self.bp_fn)
        bad_pixel_map = bp_map_hdu[0].data
        bad_pixel_map_bool = np.array(bad_pixel_map, dtype=bool)
        print(("Using bad pixel map {}".format(self.bp_fn)))

        if self.bkg_fn != None:
            background_hdu = f.open(background_fname)
            background = background_hdu[0].data
            print("Subtracting background frame {} from all science files".format(self.bkg_fn))

        if dark_exp_time != self.header["EXPTIME"]:
            warnings.warn("The master dark file doesn't have the same exposure time as the flats. We'll scale the dark for now, but this isn't ideal", UserWarning)
            factor = self.header["EXPTIME"]/dark_exp_time
        else: 
            factor = 1. 

        #Subtract the dark, divide by flat
        redux = ((self.full_image - factor*master_dark)/master_flat)
        #get rid of crazy values at bad pixel
        redux = redux*~bad_pixel_map_bool

        if self.bkg_fn != None:
            redux -= background

        if clean_Bad_Pix:
            # plt.plot(bad_pixel_map_bool)
            redux = calibration.cleanBadPix(redux, bad_pixel_map_bool)
            #redux = ccdproc.cosmicray_lacosmic(redux, sigclip=5)[0]

            # redux = ccdproc.cosmicray_median(redux, mbox=7, rbox=5, gbox=7)[0]

        #Mask the bad pixels if the flag is set
        if mask_bad_pixels:
            redux *= ~bad_pixel_map_bool

        if replace_nans: 
            # nan_map = ~np.isfinite(redux)
            # redux = cleanBadPix(redux, nan_map)
            # plt.imshow(redux-after)
            nanmask = np.isnan(redux) #nan = True, just in case this is useful
            redux = np.nan_to_num(redux)


        #Put the cablibrated data back in place
        self.full_image = redux

        #Update the header
        #Add pipeline version and history keywords
        self.header['HISTORY'] = "Subtracting {} from each flat file".format(self.dark_fn)
        self.header['DARK_FN'] = self.dark_fn

        self.header['HISTORY'] = "Dividing each file by {}".format(self.flat_fn)
        self.header["FLAT_FN"] = self.flat_fn

        if self.bkg_fn != None:
            self.header['HISTORY'] = "Subtracted background frame {}".format(self.bkg_fn)
            self.header['BKG_FN'] = self.bkg_fn
            
        if mask_bad_pixels:
            self.header['HISTORY'] = "Masking all bad pixels found in {}".format(self.bp_fn)
            self.header['BP_FN'] = self.bp_fn

        if clean_Bad_Pix:
            self.header['HISTORY'] = "Cleaned all bad pixels found in {} using a median filter".format(self.bp_fn)
            self.header['CLEAN_BP'] = "True"

        #Turn on the calibrated flag
        self.calibrated = True


    def sub_background_image(self, scale_itime=True):
        """
        Subtr sact a background frame

        Args:
            scale_itime - If true then scale the background image to have the same integration time as the science frame. 

        """

        if self.bkg_fn != None:
            # try: 
            background_hdu = f.open(background_fname)
            background = background_hdu[0].data

            bkg_itime = f.open(background_img_fname)[0].header["EXPTIME"]
            print("Subtracting background frame {} from all science files".format(self.bkg_fn))

            if scale_itime:
                scitime = self.header["EXPTIME"]
                background = background * scitime/bkg_itime

            #Subtract a background image
            self.full_image -= background

            self.bkg_subbed = True
        else: 
            print("Background filename not set, please set wircpol_data.bkg_fn property to the filename of your background file")

    
    def save_wirc_object(self, wirc_object_filename, overwrite = True):
        #Save the object to a fits file   

        vers = version.get_version()
        self.header.set('PL_VERS',vers,'Version of pipeline used for processing')

        #TODO: Update the header keywords below to include a keyword description like PS_VERS above

        #These may not always be set by other function
        self.header["NSOURCES"] = self.n_sources
        self.header["DARK_FN"] = self.dark_fn
        self.header["FLAT_FN"] = self.flat_fn
        self.header["BP_FN"] = self.bp_fn
        self.header["BKG_FN"] = self.bkg_fn

        #Have the data been calibrated/background subtracted? 
        self.header["CALBRTED"] = self.calibrated
        self.header["BKG_SUBD"] = self.bkg_subbed

        hdu = fits.PrimaryHDU(self.full_image)
        hdu.header = self.header

        hdulist = fits.HDUList([hdu])

        #Now for each source, create a ImageHDU, this works even if the cutouts haven't been extracted
        for i in range(self.n_sources):

            #Create an ImageHDU for each of the sources
            # source_hdu = fits.ImageHDU(self.source_list[i].trace_images)
            source_hdu = fits.PrimaryHDU(self.source_list[i].trace_images)

            #Put in the source info
            source_hdu.header["XPOS"] = self.source_list[i].pos[0]
            source_hdu.header["XPOS_ERR"] = self.source_list[i].pos[2]

            source_hdu.header["YPOS"] = self.source_list[i].pos[1]
            source_hdu.header['YPOS_ERR'] = self.source_list[i].pos[3]

            source_hdu.header["SLIT_LOC"] = self.source_list[i].slit_pos

            #Append it to the hdu list
            hdulist.append(source_hdu)

            #TODO: Add a fits table extension (or a series of them) to contain the spectra


        print("Saving a wirc_object to {}".format(wirc_object_filename))
        hdulist.writeto(wirc_object_filename, overwrite=overwrite)


    def load_wirc_object(self, wirc_object_filename):
        '''
        Read in the wircpol_object file from a fits file
        '''

        #Open the fits file
        hdulist = fits.open(wirc_object_filename)

        #Read in the full image and the primary header
        self.full_image = hdulist[0].data
        self.header = hdulist[0].header

        #What are the calibration filenames?
        self.dark_fn = self.header["DARK_FN"]
        self.flat_fn = self.header["FLAT_FN"]
        self.bp_fn = self.header["BP_FN"]
        self.bkg_fn = self.header["BKG_FN"]

        self.filter_name = self.header['AFT'][0]

        #What's the calibration status?
        self.calibrated = self.header["CALBRTED"]
        self.bkg_subbed = self.header["BKG_SUBD"]

        #How many sources are there
        self.n_sources = self.header["NSOURCES"]

        #Create one source object for each source and append it to source_list
        self.source_list = []

        for i in range(self.n_sources):

            #Extract the source info from the header
            xpos = hdulist[i+1].header["XPOS"]
            ypos = hdulist[i+1].header["YPOS"]
            xpos_err = hdulist[i+1].header["XPOS_ERR"]
            ypos_err = hdulist[i+1].header["YPOS_ERR"]

            slit_loc = hdulist[i+1].header["SLIT_LOC"]

            #Creat the new wircpol_source object
            new_source = wircpol_source([xpos,ypos,xpos_err,ypos_err],slit_loc, i)
            new_source.trace_images = hdulist[i+1].data

            #Append it to the source_list
            self.source_list.append(new_source)


    def find_sources(self, direct_image_fn = None, threshold_sigma = 5, guess_seeing = 4, plot = False):
        """
        Find the number of sources in the image and create a wircpol_source objects for each one

        Args:
            direct_image_fn - The direct image with no mask or PG. If this is None then we find the sources with an as-of-yet to be determined method. 

        """
        
        if direct_image_fn != None:

            #Open the direct image
            direct_image = fits.open(direct_image_fn)[0].data
            
            #Get the focal plane mask. 
            mask = cross_mask_ns.astype('bool') #What does our mask look like? 

            #Find the sources 
            locations = image_utils.find_sources_in_direct_image(direct_image, mask, threshold_sigma = threshold_sigma, guess_seeing = guess_seeing, plot = plot)

            #How many sources are there? 
            self.n_sources = np.shape(locations[0,:])[0]+1

            #Append all the new objects
            for source in range(self.n_sources):
                self.source_list.append(wircpol_source(locations[source, 0], locations[source,1],source))

        else: 
            print("No direct image filename given. For now we can only find sources automatically in a direct image, so we'll assume that there's a source in the middle slit. If you wish you can add other sources as follows: \n\n > wirc_data.source_list.append(wircpol_source([y,x],slit_pos,wirc_data.n_sources+1) \
            #where slit_pos is '0','1','2' or slitless. \n > wirc_data.n_sources += 1")

            self.source_list.append(wircpol_source([1063,1027],'1',self.n_sources+1))
            self.n_sources = 1
            
        self.header['NSOURCES'] = self.n_sources

    
    def get_source_cutouts(self):
        """
        Get thumbnail cutouts for the spectra of for each source in the image. 
        """

        for source in range(self.n_sources):
            self.source_list[source].get_cutouts(self.full_image, filter_name = self.filter_name, sub_bar = True)



class wircpol_source(object):
    """
    A point-source in a a wircpol_data image    

    Args:
        pos - [x,y] - the location in the image of the source
        slit_pos - Is it in the slit with possible values of [0,1,2,'slitless']


    Attributes:
        trace_images - An array of size [4,N,N], where n is the width of the box, and there is one image for each trace
        trace_spectra - An array of size [4,3, m], where each m-sized spectrum as a wavelength, a flux and a flux error
        pol_spectra - An array of size [3,3, m], where each m-sized spectrum as a wavelength, a flux and a flux error
        calibrated_pol_spectra - An array of size [5,3, m], where each m-sized spectrum as a wavelength, a flux and a flux error

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
        self.pol_spectra = None
        self.Q = None
        self.U = None
        self.P = None
        self.theta = None
    
        self.lambda_calibrated = False

    def get_cutouts(self, image, filter_name, sub_bar=True):
        """
        Cutout thumbnails and put them into self.trace_images

        """
        self.trace_images = np.array(image_utils.cutout_trace_thumbnails(image, np.expand_dims([self.pos, self.slit_pos],axis=0), flip=False
                                    ,filter_name = filter_name, sub_bar = sub_bar)[0])

    def plot_cutouts(self, **kwargs):

        fig = plt.figure(figsize = (12,8))

        ax = fig.add_subplot(141)
        plt.imshow(self.trace_images[0,:,:], **kwargs)
        plt.text(5,145,"Top - Left", color='w')

        ax = fig.add_subplot(142)
        plt.imshow(self.trace_images[1,:,:], **kwargs)
        plt.text(5,145,"Bottom - Right", color='w')

        ax = fig.add_subplot(143)
        plt.imshow(self.trace_images[2,:,:], **kwargs)
        plt.text(5,145,"Top - Right", color='w')

        ax = fig.add_subplot(144)
        plt.imshow(self.trace_images[3,:,:], **kwargs)
        plt.text(5,145,"Bottom - Left", color='w')
        
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.90, 0.38, 0.03, 0.24])
        plt.colorbar(cax = cbar_ax)

        plt.show()


    def extract_spectra(self, sub_background = False, plot=False):
        print("Performing Spectral Extraction for source {}".format(self.index))
        spectra, spectra_std = spec_utils.spec_extraction(self.trace_images, self.slit_pos, sub_background = sub_background, plot=plot) 

        spectra_length = spectra.shape[1]

        self.trace_spectra = np.zeros((4,3,spectra_length))
        self.trace_spectra[:,0,:] = np.arange(spectra_length) #The wavelength axis, to be calibrated later. 
        self.trace_spectra[:,1,:] = spectra
        self.trace_spectra[:,2,:] = spectra_std

    def rough_lambda_calibration(self, filter_name="J", method=1):
        #Rough wavelength calibration. Will have to get better later!

        if method == 1:
            self.trace_spectra[0,0,:] = spec_utils.rough_wavelength_calibration_v1(self.trace_spectra[0,1,:], filter_name)
            self.trace_spectra[1,0,:] = spec_utils.rough_wavelength_calibration_v1(self.trace_spectra[1,1,:], filter_name)
            self.trace_spectra[2,0,:] = spec_utils.rough_wavelength_calibration_v1(self.trace_spectra[2,1,:], filter_name)
            self.trace_spectra[3,0,:] = spec_utils.rough_wavelength_calibration_v1(self.trace_spectra[3,1,:], filter_name)
        if method == 2:
            self.trace_spectra[0,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[0,1,:], filter_name)
            self.trace_spectra[1,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[1,1,:], filter_name)
            self.trace_spectra[2,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[2,1,:], filter_name)
            self.trace_spectra[3,0,:] = spec_utils.rough_wavelength_calibration_v2(self.trace_spectra[3,1,:], filter_name)

        self.lambda_calibrated = True

    def compute_polarization(self, cutmin=0, cutmax=160):


        wlQp, q, dq, wlUp,u, du = spec_utils.compute_polarization(self.trace_spectra, cutmin=cutmin, cutmax = cutmax)
        
        pol_spectra_length = q.shape[0]
        
        self.Q = np.zeros([3,pol_spectra_length])
        self.U = np.zeros([3,pol_spectra_length])
        
        self.Q[0,:] = wlQp
        self.Q[1,:] = q
        self.Q[2,:] = dq

        self.U[0,:] = wlUp
        self.U[1,:] = u
        self.U[2,:] = du

    def plot_trace_spectra(self, with_errors = False, **kwargs):

        fig = plt.figure(figsize=(7,7))
        labels = ["Top-Left", "Bottom-Right", "Top-Right", "Bottom-left"]

        for i in range(4):
            if with_errors:
                plt.errobar(self.trace_spectra[i,0,:], self.trace_spectra[i,1,:],yerr = self.trace_spectra[i,2,:], label=labels[i], **kwargs)

            else:
                plt.plot(self.trace_spectra[i,0,:], self.trace_spectra[i,1,:], label=labels[i], **kwargs)

        plt.ylabel("Flux [ADU]")

        if self.lambda_calibrated:
            plt.xlabel("Wavelength [um]")
        else:
            plt.xlabel("Wavelength [Arbitrary Unit]")
        
        plt.legend()
        plt.show()

    def plot_Q_and_U(self, with_errors = False, xlow=1.15, xhigh=1.35, ylow=-0.2, yhigh=0.2, **kwargs):

        fig = plt.figure(figsize=(7,7))

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

    # def subtract_pol_bias():

    # def wavelength_calibration():

    # def show_traces():



