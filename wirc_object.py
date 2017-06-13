import numpy as np
from astropy.io import fits
from wircpol.DRP.reduction import calibration
from DRP import version # For versioning (requires gitpython 'pip install gitpython')
from wircpol.DRP.reduction import coarse_regis
from wircpol_masks import * ### Make sure that the wircpol/DRP/mask_design directory is in your Python Path!
import matplotlib.pyplot as plt


class wirc_data(object):

    """
    A wirc+pol data file, that may include reduced data products

    Args:
        raw_filename: The filename of the raw image [optional]
        wircpol_object_filename: The filename of 

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
    def __init__(self, raw_filename=None, wircpol_object_filename=None, dark_fn = None, flat_fn = None, bp_fn = None, bkg_fn = None):

        ## Load in either the raw file, or the wircpol_object file
        if raw_filename != None and wircpol_object_filename != None:
            print("Can't open both a raw file and wircpol_object, ignoring the raw file and loading the wircpol_object_file ")
            print("Loading a wircpol_data object from file {}".format(wircpol_object_filename))
            self.load_wircpol_object(wircpol_object_filename)

        elif wircpol_object_filename != None:
            print("Loading a wircpol_data object from file {}".format(wircpol_object_filename))
            self.load_wircpol_object(wircpol_object_filename)

        elif raw_filename != None:
            print("Creating a new wircpol_data object from file {}".format(raw_filename))
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

    
    def save_wircpol_object(self, wircpol_object_filename, overwrite = True):
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


        print("Saving a wircpol_object to {}".format(wircpol_object_filename))
        hdulist.writeto(wircpol_object_filename, overwrite=overwrite)


    def load_wircpol_object(self, wircpol_object_filename):
        '''
        Read in the wircpol_object file from a fits file
        '''

        #Open the fits file
        hdulist = fits.open(wircpol_object_filename)

        #Read in the full image and the primary header
        self.full_image = hdulist[0].data
        self.header = hdulist[0].header

        #What are the calibration filenames?
        self.dark_fn = self.header["DARK_FN"]
        self.flat_fn = self.header["FLAT_FN"]
        self.bp_fn = self.header["BP_FN"]
        self.bkg_fn = self.header["BKG_FN"]

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
            locations = coarse_regis.coarse_regis(direct_image, mask, threshold_sigma = threshold_sigma, guess_seeing = guess_seeing, plot = plot)

            #How many sources are there? 
            self.n_sources = np.shape(locations[0,:])[0]+1

            #Append all the new objects
            for source in range(self.n_sources):
                self.source_list.append(wircpol_source(locations[source, 0], locations[source,1],source))

        else: 
            print("For now we can only find sources in a direct image, we'll assume that there's a source in the middle slit for now, \
                otherwise please enter a direct_image_fn as a keyword")
            #TODO put in the object right in the slit. 


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
        self.calibrated_pol_spectra = None

    def get_cutouts(self, image, filter_name, sub_bar):
        """
        Cutout thumbnails and put them into self.trace_images

        """
        self.trace_images = np.array(coarse_regis.extract_traces(image, np.expand_dims([self.pos, self.slit_pos],axis=0), flip=False
                                    ,filter_name = filter_name, sub_bar = sub_bar)[0])

    def plot_cutouts(self, **kwargs):

        fig = plt.figure(figsize = (12,8))

        ax = fig.add_subplot(141)
        ax.imshow(self.trace_images[0,:,:], **kwargs)

        ax = fig.add_subplot(142)
        ax.imshow(self.trace_images[1,:,:], **kwargs)

        ax = fig.add_subplot(143)
        ax.imshow(self.trace_images[2,:,:], **kwargs)

        ax = fig.add_subplot(144)
        ax.imshow(self.trace_images[3,:,:], **kwargs)

        plt.show()


    def extract_spectra(self,image, sub_background = False, plot=False):
        print("Performing Spectral Extraction for source {}".format(self.index))
        res_spec, res_stddev, thumbnails= spec_extraction(image, np.expand_dims([self.pos, self.slit_pos],axis=0), sub_background = sub_background, plot=plot) 



    # def compute_polarization():

    # def subtract_pol_bias():

    # def wavelength_calibration():

    # def show_traces():




