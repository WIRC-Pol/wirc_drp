
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 00:53:59 2017

@author: kaew
Basic Reduction Routines

Perform dark subtraction and flat field correction
"""
import astropy.io.fits as f
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from astropy.stats import sigma_clip
from scipy.stats import mode
from scipy.ndimage import median_filter, shift, rotate
from scipy import interpolate
import copy
import cv2
import os
import warnings
from image_registration import chi2_shift
#from wircpol.DRP.reduction.constants import *
from wirc_drp.constants import *
#from pyklip import klip
from wirc_drp.masks.wircpol_masks import * ### Make sure that the wircpol/DRP/mask_design directory is in your Python Path!
from wirc_drp import version # For versioning (requires gitpython 'pip install gitpython')
import copy
from astropy.stats import sigma_clipped_stats

def masterFlat(flat_list, master_dark_fname, normalize = 'median', local_sig_bad_pix = 3, \
                global_sig_bad_pix = 9, local_box_size = 11,  hotp_map_fname = None, verbose=False,
                output_dir = None):


    """
    Create a master normalized flat file given a list of fits files of flat fields from
    WIRC.

    flats are scaled with mode or median (in case that illumination change, like in twilight flat)
    and then median combined to reject spurious pixels.

    It also saves a bad pixel map of pixels that are further than sig_bad_pix sigma away from the median (mean?)


    flat_list: a list of file names for flat fields
    master_dark_fname: a file name of a combined dark frame of the same exposure time as these flats
    normalize: How to normalize the flat field, by 'median' or 'mode'
    local_sig_bad_pix: sigmas used to reject bad pixel based on local standard deviation in a box of size given by median_box_size
    global_sig_bad_pix: igmas used to reject bad pixel based on global distribution of the pixel-to-pixel variation
    local_box_size: the dimension of the size of the local box used to do median and standard deviation filters
    sig_bad_pix: we define bad pixels as pixels with value more than sig_bad_pix*sqrt(variance) away from the median of the frame
    hotp_map_fname: file name of the hot pixel map from the dark frame, will be deprecated and let calibrate function deal with combinding
                    two maps
    """

    #Open the master dark
    master_dark_hdu = f.open(master_dark_fname)
    master_dark = master_dark_hdu[0].data
    dark_shape = np.shape(master_dark)

    if verbose:
        print(("Subtracting {} from each flat file".format(master_dark_fname)))
    dark_exp_time = master_dark_hdu[0].header['EXPTIME']

    #Open all files into a 3D array
    #foo = np.empty((dark_shape[0],dark_shape[1],len(flat_list)))
    foo = []

    #Open first flat file to check exposure time and filter
    first_flat_hdu = f.open(flat_list[0])
    flat_exp_time = first_flat_hdu[0].header['EXPTIME']



    if dark_exp_time != flat_exp_time:
        print("The master dark file doesn't have the same exposure time as the flats. We'll scale the dark for now, but this isn't ideal", UserWarning)
        factor = flat_exp_time/dark_exp_time
    else:
        factor = 1.

    #We've already read it, so we'll stick it in foo

    print("Combining flat files")
    for i in range(0,len(flat_list)):
        try: 
            #subtract dark for each file, then normalize by mode
            hdu = f.open(flat_list[i],ignore_missing_end=True)
            d_sub = hdu[0].data  - factor*master_dark
            #normalize
            if normalize == 'mode':
                d_sub = d_sub/mode(d_sub, axis = None, nan_policy = 'omit')
            elif normalize == 'median':
                d_sub = d_sub/np.nanmedian(d_sub)
        #foo[:,:,i] = d_sub
            foo.append(d_sub)
        except:
            print("Some error. Skipping file {}".format(i))   
    #Median combine frames
    flat = np.median(foo, axis = 0)

    #Filter bad pixels
    #bad_px = sigma_clip(flat, sigma = sig_bad_pix) #old and bad
    ###Major update here: do sigma clipping on the pix-to-pix flat with the large scale vignette removed
    ###Also add local sigma clipping
    def stddevFilter(img, box_size):
        """ from
        https://stackoverflow.com/questions/28931265/calculating-variance-of-an-image-python-efficiently/36266187#36266187
        This function compute the standard deviation of an image in a
        moving box of a given size. The pixel i,j of the output is the
        standard deviation of the pixel value in the box_size x box_size box
        around the i,j pixel in the original image.
        """
        wmean, wsqrmean = (cv2.boxFilter(x, -1, (box_size, box_size), \
            borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
        return np.sqrt(wsqrmean - wmean*wmean)

    #median flat
    median_flat = median_filter(flat, local_box_size) #arbitrary size, shouldn't matter as long as it's big enough
    #standard deviation image
    stddev_im = stddevFilter(flat, local_box_size)

    #Local clipping
    local_bad_pix = np.abs(median_flat - flat) > local_sig_bad_pix*stddev_im

    #Global clipping here to reject awful pixels and dust, bad columns, etc
    pix_to_pix = flat/median_flat
    global_bad_px = sigma_clip(pix_to_pix, sigma = global_sig_bad_pix).mask #9 seems to work best

    #also set all 0 and negative pixels in flat as bad
    non_positive = flat <= 0

    #logic combine
    bad_px = np.logical_or(global_bad_px, local_bad_pix)

    #also add non_positive pixels
    bad_px = np.logical_or(bad_px, non_positive)

    #Normalize good pixel values
    if normalize == 'median':
        norm_flat = flat/np.nanmedian(flat[~bad_px])
    elif normalize == 'mode':
        norm_flat = flat/mode(flat, axis = None, nan_policy = 'omit')
    #Stick it back in the last hdu
    hdu[0].data = norm_flat

    #Add pipeline version and history keywords
    vers = version.get_version()
    hdu[0].header.set('PL_VERS',vers,'Version of pipeline used for processing')
    hdu[0].header['HISTORY'] = "############################"
    hdu[0].header['HISTORY'] = "Created master flat by median combining the following:"
    for i in range(len(flat_list)):
        hdu[0].header['HISTORY'] = flat_list[i]
    hdu[0].header['HISTORY'] = "Normalized to the median of the master flat"
    hdu[0].header['HISTORY'] = "Performed bad pixel local and global sigma clipping with {}, {}sigmas".format(local_sig_bad_pix, global_sig_bad_pix)
    hdu[0].header['HISTORY'] = "############################"

    #Parse the last fileanme
    if output_dir is not None:
        flat_outname = flat_list[-1].rsplit('.',1)[0]+"_master_flat.fits"
        flat_outname = flat_outname.rsplit('/',1)[-1]
        flat_outname = output_dir+flat_outname
    else:
        flat_outname = flat_list[-1].rsplit('.',1)[0]+"_master_flat.fits"

    #Write the fits file
    if verbose:
        print(("Writing master flat to {}".format(flat_outname)))
    hdu.writeto(flat_outname, overwrite=True)

    #If there's already a hot pixel map then we'll add to it.
    if hotp_map_fname != None:
        #read in the existing bp map
        #hdu = f.open(hotp_map_fname)
        #hdu[0].data += np.array(bad_px.mask, dtype=float)
        #hdu[0].data = np.logical_or(hdu[0].data.astype(bool), bad_px) #use logical or to combine bad pixel maps
        #bp_outname = flat_list[-1].rsplit('.',1)[0]+"_bp_map.fits"
        print("Will deal with hot pixel map from dark frames in the calibrate function")

    #else:
    #Parse the last fileanme
    if output_dir is not None:
        bp_outname = flat_list[-1].rsplit('.',1)[0]+"_bp_map.fits"
        bp_outname = bp_outname.rsplit('/',1)[-1]
        bp_outname = output_dir+bp_outname
    else:
        bp_outname = flat_list[-1].rsplit('.',1)[0]+"_bp_map.fits"

    ##### Now write the bad pixel map
    hdu[0].data = bad_px.astype(int)#np.array(bad_px.mask, dtype=float)
    #Parse the last fileanme
    # bp_outname = flat_list[-1].rsplit('.',1)[0]+"_bp_map.fits"

    #Add history keywords
    hdu[0].header['HISTORY'] = "############################"
    hdu[0].header['HISTORY'] = "Created bad pixel map by sigma clipping on pixel-to-pixel flat{}".format(flat_outname)
    hdu[0].header['HISTORY'] = "Bad pixel cutoffs: local sigma = {} and global sigma = {} for clipping".format(local_sig_bad_pix, global_sig_bad_pix)
   #hdu[0].header['HISTORY'] = "Bad pixel cutoff of {}sigma".format(sig_bad_pix)
    hdu[0].header['HISTORY'] = "A pixel value of 1 indicates a bad pixel"
    hdu[0].header['HISTORY'] = "############################"

    if verbose:
        print(("Writing bad pixel map to {}".format(bp_outname)))
    #Write the fits file
    hdu.writeto(bp_outname, overwrite=True)

    return flat_outname, bp_outname

def masterPGFlat(flat_list, master_dark_fname, normalize = 'median', local_sig_bad_pix = 3, \
                global_sig_bad_pix = 9, local_box_size = 11,  hotp_map_fname = None, verbose=False,
                output_dir = None, zeroth_order_flat_fname = None, zeroth_transmission_factor = 0.00016, offsets = [4,-1],
                normal_flat_fname = None, plot = False):


    """
    Create a master normalized PG flat file given a list of fits files of PG flat fields from
    WIRC. This function also removes the zeroth order transmission in the PG flat, provided a zeroth order flat (with mask in and PG out),
    a transmission factor, and the y, x offsets between the 0th order frame and the 0th order ghost in the PG frame.

    flats are scaled with mode or median (in case that illumination change, like in twilight flat)
    and then median combined to reject spurious pixels.

    It also saves a bad pixel map of pixels that are further than sig_bad_pix sigma away from the median (mean?)


    flat_list: a list of file names for flat fields
    master_dark_fname: a file name of a combined dark frame of the same exposure time as these flats
    normalize: How to normalize the flat field, by 'median' or 'mode'
    local_sig_bad_pix: sigmas used to reject bad pixel based on local standard deviation in a box of size given by median_box_size
    global_sig_bad_pix: igmas used to reject bad pixel based on global distribution of the pixel-to-pixel variation
    local_box_size: the dimension of the size of the local box used to do median and standard deviation filters
    sig_bad_pix: we define bad pixels as pixels with value more than sig_bad_pix*sqrt(variance) away from the median of the frame
    hotp_map_fname: file name of the hot pixel map from the dark frame, will be deprecated and let calibrate function deal with combinding
                    two maps
    output_dir: where to save the output flat file
    zeroth_order_flat_fname: a filename of the median combined zeroth order flat (mask in, PG out)
    zeroth_transmission_factor: this is a factor of the 0th order flux leaking into PG flat. The nominal measured figure is 0.012 (1.2%).
    offsets: the presence of PG offsets the 0th order image by some amout. This parameter moves the 0th order flat back so it subtracts
            the 0th order ghost in the PG flat cleanly.
    normal_flat_fname: for plotting, it is instructive to show PG flat/normal flat to show the zeroth order ghost. If this is not given, then
                       use an archival flat.
    """

    #Open the master dark
    master_dark_hdu = f.open(master_dark_fname)
    master_dark = master_dark_hdu[0].data
    dark_shape = np.shape(master_dark)

    if verbose:
        print(("Subtracting {} from each flat file".format(master_dark_fname)))
    dark_exp_time = master_dark_hdu[0].header['EXPTIME']

    #Open all files into a 3D array
    foo = np.empty((dark_shape[0],dark_shape[1],len(flat_list)))

    #Open first flat file to check exposure time
    first_flat_hdu = f.open(flat_list[0])
    flat_exp_time = first_flat_hdu[0].header['EXPTIME']
    filter_name = first_flat_hdu[0].header['AFT']

    #Open the zeroth order
    zeroth_order_flat = f.open(zeroth_order_flat_fname)[0].data


    if verbose:
        print("Subtracting zeroth order frame {} using transmission factor {} and offsets [{},{}]".format(zeroth_order_flat_fname, zeroth_transmission_factor, offsets[0],offsets[1]))


    if dark_exp_time != flat_exp_time:
        print("The master dark file doesn't have the same exposure time as the flats. We'll scale the dark for now, but this isn't ideal", UserWarning)
        factor = flat_exp_time/dark_exp_time
        print(factor)
    else:
        factor = 1.

    #scale the zeroth order image to the same exposure time
    zeroth_exp_factor = flat_exp_time/float(f.getheader(zeroth_order_flat_fname)['EXPTIME'])
    print(zeroth_exp_factor)
    zeroth_order_flat = zeroth_exp_factor*zeroth_order_flat #scale it to the same exposure time of PG flats

    #We've already read it, so we'll stick it in foo

    print("Combining flat files")
    for i in range(0,len(flat_list)):
        #subtract dark for each file, then normalize by mode
        hdu = f.open(flat_list[i])
        d_sub = hdu[0].data  - factor*master_dark

        #cleaned_d_sub = d_sub - ndimage.shift(zeroth_transmission_factor*zeroth_order_flat,offsets, order = 0) #full pixel shift

        #normalize
        if normalize == 'mode':
            d_sub = d_sub/mode(d_sub, axis = None, nan_policy = 'omit')
        elif normalize == 'median':
            d_sub = d_sub/np.nanmedian(d_sub)
        foo[:,:,i] = d_sub

    #Median combine frames
    uncleaned_flat = np.median(foo, axis = 2)

    #For PG_flat, subtract zeroth order flat

    flat = uncleaned_flat - shift(zeroth_transmission_factor*zeroth_order_flat,offsets, order = 0)

    ###Now, deal with bad pixel.

    #Filter bad pixels
    #bad_px = sigma_clip(flat, sigma = sig_bad_pix) #old and bad
    ###Major update here: do sigma clipping on the pix-to-pix flat with the large scale vignette removed
    ###Also add local sigma clipping
    def stddevFilter(img, box_size):
        """ from
        https://stackoverflow.com/questions/28931265/calculating-variance-of-an-image-python-efficiently/36266187#36266187
        This function compute the standard deviation of an image in a
        moving box of a given size. The pixel i,j of the output is the
        standard deviation of the pixel value in the box_size x box_size box
        around the i,j pixel in the original image.
        """
        wmean, wsqrmean = (cv2.boxFilter(x, -1, (box_size, box_size), \
            borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
        return np.sqrt(wsqrmean - wmean*wmean)

    #median flat
    median_flat = median_filter(flat, local_box_size) #arbitrary size, shouldn't matter as long as it's big enough
    #standard deviation image
    stddev_im = stddevFilter(flat, local_box_size)

    #Local clipping
    local_bad_pix = np.abs(median_flat - flat) > local_sig_bad_pix*stddev_im

    #Global clipping here to reject awful pixels and dust, bad columns, etc
    pix_to_pix = flat/median_flat
    global_bad_px = sigma_clip(pix_to_pix, sigma = global_sig_bad_pix).mask #9 seems to work best

    #also set all 0 and negative pixels in flat as bad
    non_positive = flat <= 0

    #logic combine
    bad_px = np.logical_or(global_bad_px, local_bad_pix)

    #also add non_positive pixels
    bad_px = np.logical_or(bad_px, non_positive)

    #Normalize good pixel values
    if normalize == 'median':
        norm_flat = flat/np.nanmedian(flat[~bad_px])
    elif normalize == 'mode':
        norm_flat = flat/mode(flat, axis = None, nan_policy = 'omit')
    #Stick it back in the last hdu
    hdu[0].data = norm_flat

    #Add pipeline version and history keywords
    vers = version.get_version()
    hdu[0].header.set('PL_VERS',vers,'Version of pipeline used for processing')
    hdu[0].header['HISTORY'] = "############################"
    hdu[0].header['HISTORY'] = "Created master flat by median combining the following:"
    for i in range(len(flat_list)):
        hdu[0].header['HISTORY'] = flat_list[i]
    if normalize == 'median':
        hdu[0].header['HISTORY'] = "Normalized to the median of the master flat"
    elif normalize == 'mode':
        hdu[0].header['HISTORY'] = "Normalized to the mode of the master flat"
    hdu[0].header['HISTORY'] = "Performed bad pixel local and global sigma clipping with {}, {}sigmas".format(local_sig_bad_pix, global_sig_bad_pix)
    hdu[0].header['HISTORY'] = "Zeroth order removed by {}, with factor {}, and offsets [{},{}]".format(zeroth_order_flat_fname, zeroth_transmission_factor, offsets[0], offsets[1])
    hdu[0].header['HISTORY'] = "############################"

    if plot:
	#Different limits for H and J
        print(filter_name)
        if filter_name == 'H__(1.64)':
            vmin, vmax = 1.1, 1.3
        else:
            vmin, vmax = 1.1, 1.17
        print(vmin, vmax, ' are the limits')
        if normal_flat_fname == None:
            normal_flat_fname = "/scr/data/calibrations/median_flat_J.fits"
            print("using archival normal flat")
        normal_flat = f.open(normal_flat_fname)[0].data
        fig, ax = plt.subplots(2,2,figsize = (20,20))
        ax0 = ax[0,0].imshow(uncleaned_flat/np.nanmedian(uncleaned_flat[~bad_px])/normal_flat, origin = 'lower', vmin = vmin, vmax = vmax)
        ax1 = ax[0,1].imshow(norm_flat/normal_flat, origin = 'lower', vmin =vmin, vmax = vmax)
        ax2 = ax[1,0].imshow(uncleaned_flat/np.nanmedian(uncleaned_flat[~bad_px])/normal_flat, origin = 'lower', vmin = vmin, vmax = vmax)
        ax3 = ax[1,1].imshow(norm_flat/normal_flat, origin = 'lower', vmin =vmin, vmax = vmax)
        ax[0,0].set_xlim([400,1600])
        ax[0,0].set_ylim([400,1600])
        ax[0,1].set_xlim([400,1600])
        ax[0,1].set_ylim([400,1600])
        ax[1,0].set_xlim([400,1000])
        ax[1,0].set_ylim([1100,1600])
        ax[1,1].set_xlim([400,1000])
        ax[1,1].set_ylim([1100,1600])
        ax[0,0].set_title('PG flat')
        ax[0,1].set_title('Zeroth order subtracted')
        plt.colorbar(ax0, ax = ax[0,0])
        plt.colorbar(ax1, ax = ax[0,1])
        plt.show()

    #Parse the last fileanme
    if output_dir is not None:
        flat_outname = flat_list[-1].rsplit('.',1)[0]+"_master_PG_flat.fits"
        flat_outname = flat_outname.rsplit('/',1)[-1]
        flat_outname = output_dir+flat_outname
    else:
        flat_outname = flat_list[-1].rsplit('.',1)[0]+"_master_PG_flat.fits"

    #Write the fits file
    if verbose:
        print(("Writing master flat to {}".format(flat_outname)))
    hdu.writeto(flat_outname, overwrite=True)

    #If there's already a hot pixel map then we'll add to it.
    if hotp_map_fname != None:
        #read in the existing bp map
        #hdu = f.open(hotp_map_fname)
        #hdu[0].data += np.array(bad_px.mask, dtype=float)
        #hdu[0].data = np.logical_or(hdu[0].data.astype(bool), bad_px) #use logical or to combine bad pixel maps
        #bp_outname = flat_list[-1].rsplit('.',1)[0]+"_bp_map.fits"
        print("Will deal with hot pixel map from dark frames in the calibrate function")

    #else:
    #Parse the last fileanme
    if output_dir is not None:
        bp_outname = flat_list[-1].rsplit('.',1)[0]+"_bp_map.fits"
        bp_outname = bp_outname.rsplit('/',1)[-1]
        bp_outname = output_dir+bp_outname
    else:
        bp_outname = flat_list[-1].rsplit('.',1)[0]+"_bp_map.fits"

    ##### Now write the bad pixel map
    hdu[0].data = bad_px.astype(int)#np.array(bad_px.mask, dtype=float)
    #Parse the last fileanme
    # bp_outname = flat_list[-1].rsplit('.',1)[0]+"_bp_map.fits"

    #Add history keywords
    hdu[0].header['HISTORY'] = "############################"
    hdu[0].header['HISTORY'] = "Created bad pixel map by sigma clipping on pixel-to-pixel flat{}".format(flat_outname)
    hdu[0].header['HISTORY'] = "Bad pixel cutoffs: local sigma = {} and global sigma = {} for clipping".format(local_sig_bad_pix, global_sig_bad_pix)
   #hdu[0].header['HISTORY'] = "Bad pixel cutoff of {}sigma".format(sig_bad_pix)
    hdu[0].header['HISTORY'] = "A pixel value of 1 indicates a bad pixel"
    hdu[0].header['HISTORY'] = "############################"

    if verbose:
        print(("Writing bad pixel map to {}".format(bp_outname)))
    #Write the fits file
    hdu.writeto(bp_outname, overwrite=True)

    return flat_outname, bp_outname

def masterDark(dark_list, bad_pix_method = 'MAD', sig_hot_pix = 5, output_dir = None):

    """
    Create a master dark file from the median of a list of fits files
    It also saves a bad pixel map of hot pixels and pixels that have a value of 0. in the dark.

    Bad pixels here are defined to be pixels with variance over some given cutoff.
    Options here are to use either median absolute deviation (MAD) or standard deviation.

    Inputs:
        dark_list: a list of names for dark frames with the same exposure time.
        bad_pix_method: criteria for bad pixel clipping, options are:
                - 'sigma_clipping': sigma clip on the median combined image
                - 'MAD': use median absolute deiviation of that pixel to determine a cutoff (MAD = median( abs(i - median(i))) )
                         then sigma clip the MAD frame
                - 'standard_deviation':  use std deviation of that pixel to determine a cutoff, then sigma clip the standard deviation frame
        sig_hot_pix: a cutoff for each bad_pix_method, recommend sigma_clipping < 9, MAD < 10, standard_deviation < ?.

    """
    #Open all files into a 3D array
    print("Creating a master dark")
    dark_cube = np.empty((len(dark_list),2048,2048))
    for i in range(len(dark_list)):
        try:
            hdu = f.open(dark_list[i])
            dark_cube[i,:,:] = hdu[0].data
            hdu.close()
        except:
            print('File Error; moving on to next file.')
            dark_cube[i,:,:] = [([0]*2048)]*2048
            continue

    #Create the master dark
    master_dark = np.median(dark_cube, axis = 0)

    if bad_pix_method == 'sigma_clipping':
        hot_px = sigma_clip(master_dark, sigma = sig_hot_pix)
    elif bad_pix_method == 'MAD':
        MAD = np.median(np.abs(dark_cube - master_dark ), axis = 0) #compute MAD
        hot_px = sigma_clip(MAD, sigma = sig_hot_pix)
    elif bad_pix_method == 'standard_deviation':
        SD = np.std(dark_cube, axis = 0)
        hot_px = sigma_clip(SD, sigma = sig_hot_pix)
    else:
        print('%s is in valid, use MAD instead'%bad_pix_method)
        MAD = np.median(np.abs(dark_cube - master_dark ), axis = 0) #compute MAD
        hot_px = sigma_clip(MAD, sigma = sig_hot_pix)


    #zero_px = master_dark == 0.

    bad_px = hot_px.mask #| zero_px

    #Stick it back in the last hdu
    hdu[0].data = master_dark

    #Add pipeline version and history keywords
    vers = version.get_version()
    hdu[0].header.set('PL_VERS',vers,'Version of pipeline used for processing')
    hdu[0].header['HISTORY'] = "############################"
    hdu[0].header['HISTORY'] = "Created master dark by median combining the following frames"
    for i in range(len(dark_list)):
        hdu[0].header['HISTORY'] = dark_list[i]
    hdu[0].header['HISTORY'] = "############################"

    #Parse the last fileanme
    if output_dir is not None:
        dark_outname = dark_list[-1].rsplit('.',1)[0]+"_master_dark.fits"
        dark_outname = dark_outname.rsplit("/",1)[-1]
        dark_outname = output_dir+dark_outname
    else:
        dark_outname = dark_list[-1].rsplit('.',1)[0]+"_master_dark.fits"

    print(("Writing master dark to {}".format(dark_outname)))
    #Write the fits file
    hdu.writeto(dark_outname, overwrite=True)

    #Stick it back in the last hdu
    #hdu[0].data = np.array(bad_px, dtype=float)*2
    hdu[0].data = np.array(bad_px, dtype=float) #this is for new version, separate maps from dark and flat

    #Add history keywords
    #Add history keywords
    hdu[0].header['HISTORY'] = "############################"
    hdu[0].header['HISTORY'] = "Created hot pixel map by {}: {}".format(bad_pix_method, dark_outname)
    hdu[0].header['HISTORY'] = "Bad pixel cutoff of {}sigma".format(sig_hot_pix)
    hdu[0].header['HISTORY'] = "A pixel value of 1 indicates a hot pixel"
    hdu[0].header['HISTORY'] = "############################"

    #Parse the last filename
    if output_dir is not None:
        bp_outname = dark_list[-1].rsplit('.',1)[0]+"_hp_map.fits"
        bp_outname = bp_outname.rsplit("/",1)[-1]
        bp_outname = output_dir+bp_outname
    else:
        bp_outname = dark_list[-1].rsplit('.',1)[0]+"_hp_map.fits" #hp map is from dark, as oppose to bp map from flat

    print(("Writing master dark to {}".format(bp_outname)))
    #Write the fits file
    hdu.writeto(bp_outname, overwrite=True)

    return dark_outname, bp_outname



def calibrate(science_list_fname, master_flat_fname, master_dark_fname, hp_map_fname, bp_map_fname, mask_bad_pixels = False,
                clean_Bad_Pix=True, replace_nans=True, background_fname = None, outdir = None):
    """
    Subtract dark; divide flat
    Bad pixels are masked out using the bad_pixel_map with 0 = bad and 1 = good pixels

    """

    #Get the list of science frames
    #science_list = np.loadtxt(science_list_fname, dtype=str)
    science_list = science_list_fname

    #Open the master dark
    master_dark_hdu = f.open(master_dark_fname)
    master_dark = master_dark_hdu[0].data
    dark_shape = np.shape(master_dark)
    print(("Subtracting {} from each flat file".format(master_dark_fname)))
    dark_exp_time = master_dark_hdu[0].header['EXPTIME']

    #Open the master flat
    master_flat_hdu = f.open(master_flat_fname)
    master_flat = master_flat_hdu[0].data
    print(("Dividing each file by {}".format(master_flat_fname)))
    dark_exp_time = master_dark_hdu[0].header['EXPTIME']

    #Open the bad pixel map from flat
    bp_map_hdu = f.open(bp_map_fname)
    bad_pixel_map = bp_map_hdu[0].data
    bad_pixel_map_bool = np.array(bad_pixel_map, dtype=bool)
    print(("Using bad pixel map {}".format(bp_map_fname)))

    #now if hot pixel map from dark is also given
    if hp_map_fname != None:
        hp_map_hdu = f.open(hp_map_fname)
        hot_pixel_map = hp_map_hdu[0].data
        bad_pixel_map_bool = np.logical_or(bad_pixel_map_bool, hot_pixel_map.astype(bool) )


    if background_fname != None:
        background_hdu = f.open(background_fname)
        background = background_hdu[0].data
        print("Subtracting background frame {} from all science files".format(background_fname))


    for fname in science_list:
        #Open the file
        print(("Calibrating {}".format(fname
            )))
        hdu = f.open(fname)
        data = hdu[0].data
        science_exp_time = hdu[0].header['EXPTIME']

        if dark_exp_time != science_exp_time:
            warnings.warn("The master dark file doesn't have the same exposure time as the data. We'll scale the dark for now, but this isn't ideal", UserWarning)
            factor = science_exp_time/dark_exp_time
        else:
            factor = 1.

        #Subtract the dark, divide by flat
        redux = ((data - factor*master_dark)/master_flat)
        #get rid of crazy values at bad pixel
        redux = redux*~bad_pixel_map_bool

        if background_fname != None:
            redux -= background

        if clean_Bad_Pix:
            # plt.plot(bad_pixel_map_bool)
            redux = cleanBadPix(redux, bad_pixel_map_bool)
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

        #Put the cablibrated data back in the HDU list
        hdu[0].data = redux

        #Add pipeline version and history keywords
        vers = version.get_version()
        hdu[0].header.set('PL_VERS',vers,'Version of pipeline used for processing')
        hdu[0].header['HISTORY'] = "Subtracting {} from each flat file".format(master_dark_fname)
        hdu[0].header['HISTORY'] = "Dividing each file by {}".format(master_flat_fname)

        if background_fname != None:
            hdu[0].header['HISTORY'] = "Subtracted background frame {}".format(background_fname)

        if mask_bad_pixels:
            hdu[0].header['HISTORY'] = "Masking all bad pixels found in {}".format(bp_map_fname)

        if clean_Bad_Pix:
            hdu[0].header['HISTORY'] = "Cleaned all bad pixels found in {} using a median filter".format(bp_map_fname)

        # #Append the bad pixel list to the HDU list
        # hdu.append(f.PrimaryHDU([bad_pixel_map]))
        # hdu[1].header['HISTORY'] = "Appending bad pixel map :{}".format(bp_map_fname)
        # hdu[1].header['HISTORY'] = "0 = good pixel"
        # hdu[1].header['HISTORY'] = "1 = bad pixel from flat fields"
        # hdu[1].header['HISTORY'] = "2 = hot pixel from darks"

        outname = fname.split('.')[0]+"_calib.fits"

        #if an output directory is specified we can write out to that directory instead
        #making sure to take only the stuff after the last '/' to avoid directory issues from fname
        if outdir:
            outname = outdir + fname.split('/')[-1]

        print(("Writing calibrated file to {}".format(outname)))
        #Save the calibrated file
        hdu.writeto(outname, overwrite=True)

        # f.PrimaryHDU(redux).writeto('redux_'+i, overwrite = True)

def replace_bad_pix_with_interpolation(image, bad_pixel_map, interpolation_type = 'linear'):
    """
    at each location of bad pixel, pick a box of given size around it, interpolate good pixels in
    that box, then replace the bad pixel with that.

    This function is intended for a small image since trying to interpolate a 2x2k image will not end well.

    Inputs:
        image: a 2D array representing a science image
        bad_pixel_map: a 2D array with 1 representing bad pixels
        interpolation_type: a choice of interpolations from scipy.ndimage.griddata. valid options are nearest,
                            linear, and cubic.
    Output:
        a 2D array with all bad pixels replaced with the given interpolation
    """
    if image.shape[0] > 500:
        print('This image is big; perhaps try cleanBadPix.')
    #make sure bad_pixel_map is booleen
    bad_pixel_map = bad_pixel_map.astype('bool')

    if bad_pixel_map.shape != image.shape:
        raise ValueError('Image and bad pixel map must be of the same size.')

    #create coordinate grids
    y, x = np.mgrid[:image.shape[0], :image.shape[1]]
    #valid data and coords
    valid_data = image[~bad_pixel_map].ravel()
    coords = np.vstack([y[~bad_pixel_map], x[~bad_pixel_map]]).T

    #now, replace bad pixels
    res = interpolate.griddata(coords, valid_data, (y,x), method = interpolation_type)

    return res


def cleanBadPix(redux_science, bad_pixel_map, method = 'median', replacement_box = 5, replace_constant = -99):
    """
    replace bad pixels by either median, interpolation, or a constant.

    Inputs:
            redux_science: the 2D array representing the reduced science image
            bad_pixel_map: the 2D map of the bad pixel locations. value 1 is bad pixel
            method: either 'median', 'interpolation', or 'constant'
                - median => bad pixel replaced by the median within the replacement_box
                - *interpolation => bad pixel replaced by the 2D linear interpolation within the replacement_box
                - contant => just replace bad pixel with some constant
            replacement_box: size of the box used in median filtering and interpolation
            replace_constant: if the method is constant, then replace all bad pixels with this constant

    Output: 2D array of a cleaned image

    """
    #add negative pixels to the bad pixel map
    bad_pixel_map = np.logical_or(bad_pixel_map, redux_science <= 0)
    # im = np.copy(redux_science)
    # im[np.where(bad_pixel_map)[1]] = 0.
    if method == 'median':
        med_fil = median_filter(redux_science, size = replacement_box)

        cleaned = redux_science*~bad_pixel_map + med_fil*bad_pixel_map

    #elif method == 'interpolate':

    # print('so clean')

    return cleaned

def sum_images(filelist):
    """
    Super simple sum of all the images in a list.
    """

    nfiles = np.size(filelist)

    print("Summing together {} files".format(nfiles))

    ims = []

    for fname in filelist:
        hdu = f.open(fname)
        ims.append(hdu[0].data)

    ims = np.array(ims)

    sum_im = np.nansum(ims, axis=0)
    hdu[0].data = sum_im

    #Add pipeline version and history keywords
    vers = version.get_version()
    hdu[0].header.set('PL_VERS',vers,'Version of pipeline used for processing')
    hdu[0].header['HISTORY'] = "Summed up the following images:"

    for fname in filelist:
        hdu[0].header['HISTORY'] = fname

    hdu[0].header['HISTORY'] = "######################"

    outname = filelist[-1].split('.')[0]+'_summed.fits'

    print("Writing out final file to {}".format(outname))

    hdu.writeto(outname, overwrite=True)

def get_relative_image_offsets(cutouts, plot = False, save_cutout = False):

    '''
    This function returns the relative x and y offsets between a set of images,
    determined through cross correlation (using the chi2_shift image_registration python packge)
    It really works best on either very bright sources or on sources that have been background subracted.

    Inputs:
        cutouts         -   an array of cutouts with dimensions [m,n,k,l,l] where m is the number of images,
                            n is the number of sources per image, k is the number of traces (probably always 4),
                            and l is the height and width of the cutout (usually 80 for J and bigger for H-band)

    Outputs:
        offsets -   an [m-1, n, 4] sized array, where the components of the third dimension are [x,y, xerr, yerr]
                    Note: xerr and yerr aren't currently working very well.
    '''

    #Get cutouts info
    sz = np.shape(cutouts)
    nfiles = sz[0]
    n_sources = sz[1]
    cutout_sz = sz[3]

    #The output
    offsets = []

    #Stack the first image horizontally:
    im0_stacks = []
    for j in range(n_sources):
        #Create the horizontal stack

        stack = np.concatenate((cutouts[0,j,0,:,:], cutouts[0,j,1,:,:], cutouts[0,j,2,:,:],cutouts[0,j,3,:,:]), axis=1)

        #Get rid of outlying pixels
        tmp = np.copy(stack)*0.
        stack = median_filter(stack, size=5, output=tmp)
        stack = tmp
        im0_stacks.append(stack)

    #Make the list a numpy array
    im0_stacks = np.array(im0_stacks)
    #plt.imshow(im0_stacks[0], origin = 'lower')
    #plt.show()

    #Step through the remaining files and calculate their relative offset compared to the first file.
    for i in np.arange(0,nfiles): #include the first frame as a sanity check
        img_offset = []
        for j in range(n_sources):
            #Stack this cutout
            horiz_stack = np.concatenate((cutouts[i,j,0,:,:], cutouts[i,j,1,:,:], cutouts[i,j,2,:,:],cutouts[i,j,3,:,:]), axis=1)

            #To get rid of bad pixels
            tmp = np.copy(horiz_stack)*0.
            horiz_stack = median_filter(horiz_stack, size=5, output=tmp) #To get rid of mixed
            horiz_stack = tmp

            #Calculate the image offsets
            #plt.imshow(horiz_stack, origin = 'lower')
            #plt.show()
            shifted = chi2_shift(im0_stacks[j,:,:],horiz_stack, zeromean=True, verbose=False, return_error=True)
            img_offset.append(shifted)

        offsets.append(img_offset)
        if plot:
            plt.figure(figsize = (12,3))
            plt.imshow(np.array(im0_stacks[0]) - shift(horiz_stack,[-img_offset[0][1]+0.5,-img_offset[0][0]], order = 4 ), origin = 'lower')
            plt.show()
        if save_cutout:
            f.PrimaryHDU(shift(horiz_stack,[-img_offset[0][1]+0.5,-img_offset[0][0]], order = 4 )).writeto(str(i)+'.fits',overwrite = True)
            f.PrimaryHDU(np.array(im0_stacks[0]) - shift(horiz_stack,[-img_offset[0][1]+0.5,-img_offset[0][0]], order = 4 )).writeto(str(i)+'_sub.fits', overwrite = True)
        #for debugging
        #print( img_offset[0][1]-0.5, img_offset[0][0]  )
    return offsets

def register_and_combine_raw(direct_image_fname, spec_list_fname, datadir = "", background_img_fname = None, locations= None, cutouts = None, quiet=True,
                            combine = 'median', save_fits=True, save_each = False, plot=False):
    #
    # This functions reads in a list of science frames, performs cross correlation and then shifts and combines them
    #
    # Inputs
    #    direct_image_fname      -   a string that holds the path and filename to the direct image, which is used to find the locations
    #                                of the sources in the image. If the keyword 'locations' is provided no direct image is read,
    #                                and instead the provided locations are used
    #    spec_list_fname         -   a string that holds the path and filename of a list of science images
    #    background_img_fname    -   (keyword) a string keyword that holds the path and filename of a background image to be subtracted before cross correlation
    #    locations               -   (keyword) an array of locations of the sources in the image. If this is provided then no direct image is read.
    #                                You might provide this if you've already read in the direct image and have already found the source locations, or if
    #                                 you want to determine them yourself
    #     cutouts                 -   (keyword) an array of cutouts with dimensopns [m,n,k,l,l] where m is the number of images,
    #                                 n is the number of sources per image, k is the number of traces (probably always 4),
    #                                 and l is the height and width of the cutout (usually 80 for J and bigger for H-band).
    #                                 If you provide this keyword cutouts will not be extracted from the science images and these cutouts will be used
    #                                 to determine image offsets. However the science images will still be read and shifted.
    #     save_fits               -   (keyword) if set to true then save the registered and combined images
    #     save_each               -   (keyword) if true then save the aligned version of each input image.

    # Outputs
    #     spec_image              -   the name of the output file where the combined image was saved

    #If locations == None then automatically find the source locations.
    if locations == None:
        #The mask - required to find the locations
        mask = cross_mask_ns.astype('bool')

        #### Read in the direct image to get the source locations
        direct_image = f.open(direct_image_fname)[0].data
        locations = coarse_regis.coarse_regis(direct_image, mask, threshold_sigma = 5, guess_seeing = 4, plot = plot)

    #The number of sources found
    n_sources = np.shape(locations[0,:])[0]+1

    #For the cross correlation to work reliably a background image should be supplied.
    if background_img_fname != None:
        bkg_img = f.open(background_img_fname)[0].data
        bkg_itime = f.open(background_img_fname)[0].header["EXPTIME"]

    #Get the list of spectral images
    spec_images = a.read(spec_list_fname, format = "fast_no_header")['col1']
    n_images = np.size(spec_images)

    #An array that will hold all the images
    spec_stack = np.zeros([n_images, detector_size, detector_size])

    cutouts = []

    #Step through all the images, save the traces cutouts and put the full image in spec_stack
    for j,i in enumerate(spec_images):

        if not quiet:
            print("\nReading in file {}, ({} from {})".format(i,j+1,len(spec_images)))
        spectral_hdulist = f.open(datadir+i)
        spectral_image = np.nan_to_num(spectral_hdulist[0].data)
        scitime = spectral_hdulist[0].header["EXPTIME"]

        #TODO: ADD CHECK TO MAKE SURE FILES HAVE SAME EXPOSURE TIME.

        #### Get the Filter Info ####
        aft_filter = spectral_hdulist[0].header['AFT']
        filter_name = aft_filter[0]
        if filter_name != 'J' and filter_name != 'H':
            print("The pipeline was expecting either a J- or H-band filter but found {} instead".format(aft_filter))
            print("Returning.\n")
            break
        else:
            if not quiet:
                print("Found a {}-band filter in the header of file {}".format(filter_name,i))
        #Getting info about the filter.
        lb,dlb,f0,filter_trans_int, central_wl = getFilterInfo(filter_name)

        #Status update
        if not quiet:
            print ("Cutting out the traces")

        #Subtract a background if present.
        if background_img_fname != None:
            if not quiet:
                print("Subtracting {} from the cutouts as background".format(background_img_fname))
            # plt.imshow(spectral_image-bkg_img*scitime/bkg_itime, vmin=0, vmax=50)
            cutouts.append(coarse_regis.extract_traces(np.copy(spectral_image-bkg_img*scitime/bkg_itime), locations, flip = False))
        else:
            cutouts.append(coarse_regis.extract_traces(np.copy(spectral_image), locations, flip = False))

        #Put the image in the stack
        spec_stack[j,:,:] = spectral_image

    #Make cutouts an array and get the size
    cutouts = np.array(cutouts)
    sz = cutouts.shape
    cutout_sz = sz[3]

    #Calculate the image offsets
    offsets = get_relative_image_offsets(cutouts, plot = plot, save_cutout = True)
    offsets = np.array(offsets)

    #A list of all the offsets to write to the header later
    dx_list = []
    dy_list = []

    #Now shift images using pyklip.rotate -- Can think about writing out own code, or copying this code so people don't have to get pyklip to use wircpol
    #print('Offsets length = ', len(offsets))
    #print(n_images - 1)
    for i in np.arange(0,n_images):


        #Calculate the mean offset of sources outside the slit
        where_slitless = np.where(locations[:,1] == 'slitless')[0]
        #print(offsets[i,where_slitless])
        dx = np.mean(offsets[i,where_slitless,0])
        dy = np.mean(offsets[i,where_slitless,1])-0.5 #for some reason...

        dx_list.append(dx)
        dy_list.append(dy)

        if not quiet:
            print("Registering frame {} with (dx,dy) = ({},{})".format(i,dx,dy))

        #The old and new centers
        old_center = np.array([cutout_sz/2., cutout_sz/2.])
        new_center = old_center-[dx,dy]

        #Now re-align the images
        #print('Max value ', np.max(spec_stack[i,:,:]))
        #spec_stack[i,:,:] = klip.align_and_scale(spec_stack[i,:,:], new_center, old_center=old_center, scale_factor=1,dtype=float)
        spec_stack[i,:,:] = shift(spec_stack[i,:,:], [-dy,-dx], order = 4)
        #print('NaNs',len(spec_stack[i,:,:][np.isnan(spec_stack[i,:,:])]))

        #if save_each, save the aligned version of each file by adding _aligned at the end of the name before .fits
        if save_each:
            #file name
            outname = spec_images[i].rsplit('.',1)[-2]+'_aligned.fits'
            #data
            spectral_hdulist[0].data = spec_stack[i,:,:]
            #Add pipeline version and history keywords
            vers = version.get_version()
            hdu[0].header.set('PL_VERS',vers,'Version of pipeline used for processing')
            spectral_hdulist[0].header['HISTORY'] = "######"
            spectral_hdulist[0].header['HISTORY'] = "register_and_combine_raw: Found relative offsets with respect to the image: "
            spectral_hdulist[0].header['HISTORY'] = "{} (dx,dy) = ({}, {})".format(spec_images[0], dx_list[0], dy_list[0])
            spectral_hdulist[0].header['HISTORY'] = "######"
            #write
            spectral_hdulist.writeto(outname, overwrite = True)

    #Collapse the image by it's sum, median, or mean based on 'combine' parameter. Default is median.
    if combine == 'sum':
        comb_spec = np.nansum(spec_stack, axis=0)
    elif combine == 'median':
        comb_spec = np.nanmedian(spec_stack, axis = 0)
    elif combine == 'mean':
        comb_spec = np.nanmean(spec_stack, axis = 0)
    else:
        print(combine+' is not an option. Use median instead.')
        comb_spec = np.nanmedian(spec_stack, axis = 0)


    dx_list = np.array(dx_list)
    dy_list = np.array(dy_list)

    #Save the final image to a fits file
    outname = spec_images[-1].rsplit('.',1)[-2]+'_combined.fits'
    if save_fits:
        outname = spec_images[-1].rsplit('.',1)[-2]+'_combined.fits'
        #these are spectral_hdulist from the last file in the spec_list
        spectral_hdulist[0].data = comb_spec

        #Add pipeline version and history keywords
        vers = version.get_version()
        spectral_hdulist[0].header.set('PL_VERS',vers,'Version of pipeline used for processing')
        spectral_hdulist[0].header['HISTORY'] = "######"
        spectral_hdulist[0].header['HISTORY'] = "register_and_combine_raw: Found relative offsets, reigstered the following images: "
        spectral_hdulist[0].header['HISTORY'] = "{} (dx,dy) = ({}, {})".format(spec_images[0], dx_list[0], dy_list[0])

        for i in np.arange(1,n_images-1):
            spectral_hdulist[0].header['HISTORY'] = "{} (dx,dy) = ({}, {})".format(spec_images[i], dx_list[i-1], dy_list[i-1])
        spectral_hdulist[0].header['HISTORY'] = "Combine files by {}".format(combine)
        spectral_hdulist[0].header['HISTORY'] = "Total files combined: {}".format(n_images)
        spectral_hdulist[0].header['HISTORY'] = "######"

        spectral_hdulist[0].header['NFILES'] = n_images

        if not quiet:
            print("Writing fits to {}".format(datadir+outname))

        spectral_hdulist.writeto(outname, overwrite=True)

    return outname

def shiftSub(image, slit_gap1, slit_gap2):
    """
    Create a full frame background subtracted image. The background image is an average between a frame shifted
    to +x by slit_gap1 pixel, and -x by slit_gap2 pixel. This is then subtracted off of the image.
    """
    bkg = (shift(image,(0,slit_gap1), order = 3) + shift(image,(0,-slit_gap2), order = 3))/2.
    return image - bkg

def destripe_raw_image(image):
    '''
    Destripe the detector by subtracting bias levels from each quadrant
    '''

    for i in range(1024):

        image[1024+i,:1024] = image[1024+i,0:1024] - np.median(image[1024+i,0:100])
        image[:1024,i] =     image[0:1024,i]- np.median(image[40:100,i])
        # image[1024:,1024+i] =     image[1024:,1024+i] - np.median(image[-5:,1024+i])
        image[i,1024:] = image[i,1024:] - np.median(image[i,-80:])

    #The top right qudrant is a real pain because there aren't many pixel from which to measure the bias, so here we use a workaround
    #that first subtracts from each row to take away the median values, and then subtracts column by column.
    tmp = copy.deepcopy(image)
    for i in range(1024):
        tmp[1024+i,1024:] = image[1024+i,1024:] - np.median(image[1024+i,:])

    for i in range(1024):
        image[1024:,1024+i] = image[1024:,1024+i] - np.median(tmp[1024:,1024+i])

    return image

def destripe_after_bkg_sub(image, sigma = 3, iters=5, mode = 'robust'):
    '''
    Destripe the detector by subtracting the median of each row/column from each sector.
    This will work best after you subtract a background sky image.

    #Robust method updated March 19, 2019
    #Simple is likely out of date. 

    Input parameters:
    image    -    A 2048x 2048 WIRC image.
    sigma    -    The number of sigmas to klip. The sigma that goes into the astropy sigma_clipped_stats function.
    iters    -    The number of times to run sigma clipping. The iters argument that goes into the astropy sigma_clipped stats fuction
    mode     -    The statistics to use. simple is faster, but more sensitive to outliers. Robust is slower but more robust.

    Outputs:
    image    - A destriped detector image.
    '''

    quad1 = image[:1024,:1024]
    quad2 = np.rot90(image[1024:,:1024], k=3, axes=(0, 1))
    quad3 = np.rot90(image[1024:,1024:], k=2, axes=(0, 1))
    quad4 = np.rot90(image[:1024,1024:], k=1, axes=(0, 1))

    mn_quad = np.median([quad1,quad2,quad3,quad4],axis=0)

    clean_imm = copy.deepcopy(image)

    if mode == 'robust':
        for i in range(1024):

            #Upper Left
            to_sub = sigma_clipped_stats(mn_quad[:,i],sigma=sigma,maxiters=iters)[1] #Returns mean, median, stddev (default parameters are 5 iterations of 3-sigma clipping)

            clean_imm[:1024,i]  -= to_sub
            clean_imm[1024:,-i] -= to_sub
            clean_imm[-i,:1024] -= to_sub
            clean_imm[i,1024:]  -= to_sub

        return clean_imm

    elif mode == 'simple':

        for i in range(1024):

            #Upper Left
            image[1024+i,:1024] = image[1024+i,0:1024] - np.nanmedian(image[1024+i,0:1024])

            #Lower Left
            image[:1024,i] =     image[:1024,i]- np.nanmedian(image[:1024,i])

            #Upper Right
            image[1024:,1024+i] =     image[1024:,1024+i] - np.nanmedian(image[1024:,1024+i])

            #Lower Right
            image[i,1024:] = image[i,1024:] - np.nanmedian(image[i,1024:])

        return image

    else:
        print("'mode' keyword note understood, please choose either mode='robust' or mode='simple'. Not doing any destriping.")
        return image

def remove_correlated_channel_noise(image,n_channels = 8, mask = None):
    '''
    This function attemps to remove detector bias drifts introduced in the readout procedue.
    It assumes that the same bias occurs in each read-out channel, so it medians the 8 readout channels in each quadrant
    and then subtracts the median from each channel.

    Note, it could be that using a median here is a bad idea when there is a source.
    A median might also mess with things because of the edges of the detector being screwy.
    One way to get around this is to mask things out.

    Run this after background frame subtraction.

    ;TODO: include functionality for mask.


    Inputs:
    image      -    The image to be subtracted
    n_channels -    The number of channels per quadrant. This is 8 for the WIRC detector.
    mask       -    An optional mask, where anything with a "True value" will be masked out when calculating the mean bias signal.

    Outputs:
    image_copy    -    The bias-corrected image.
    '''

    #Make the output image
    image_copy = copy.deepcopy(image)

    #If a mask is provided then set mask locations to be NANs.
    if mask is not None:
        image_copy[np.where(mask)] = np.nan

    #The size of each quadrant
    quad_size = detector_size//2

    #The width of each read out channel
    channel_width = quad_size//n_channels

    #Set up the arrays to hold the median channel data.
    ll_channel_median = np.zeros([n_channels,channel_width,quad_size])
    lr_channel_median = np.zeros([n_channels,channel_width,quad_size])
    ul_channel_median = np.zeros([n_channels,channel_width,quad_size])
    ur_channel_median = np.zeros([n_channels,channel_width,quad_size])

    for i in range(n_channels):
        #Lower Left
        ll_channel_median[i,:,:] = image_copy[channel_width*(i):channel_width*(i+1),0:quad_size]
        #Lower Right
        lr_channel_median[i,:,:] = image_copy[0:quad_size,quad_size+channel_width*(i):quad_size+channel_width*(i+1)].T
        #Upper Left
        ul_channel_median[i,:,:] = image_copy[quad_size:,channel_width*(i):channel_width*(i+1)].T
        #Upper Right
        ur_channel_median[i,:,:] = image_copy[quad_size+channel_width*(i):quad_size+channel_width*(i+1),quad_size:]


    #Now take the median across all channels
    ll_channel_median = np.nanmedian(ll_channel_median,axis=0)
    lr_channel_median = np.nanmedian(lr_channel_median,axis=0)
    ul_channel_median = np.nanmedian(ul_channel_median,axis=0)
    ur_channel_median = np.nanmedian(ur_channel_median,axis=0)

    #Copy the original again for the output image
    image_copy = copy.deepcopy(image)

    for i in range(n_channels):

        #Lower Left
        image_copy[channel_width*(i):channel_width*(i+1),0:quad_size] -= ll_channel_median

        #Lower Right
        image_copy[:quad_size,quad_size+channel_width*(i):quad_size+channel_width*(i+1)] -= lr_channel_median.T

        #Upper Left
        image_copy[quad_size:,channel_width*(i):channel_width*(i+1)] -= ul_channel_median.T

        #Upper Right
        image_copy[quad_size+channel_width*(i):quad_size+channel_width*(i+1),quad_size:] -= ur_channel_median

    return image_copy

def PCA_subtraction(im, ref_lib, num_PCA_modes):
    """
    Does PCA subtraction of science frames using KLIP algorithm described in Soummer et al. (2012)
    
    im: 2-D np.array
        2-D image to do PCA subtraction on
    ref_lib: list of str
        list of .fits files to serve as reference library for PCA subtraction
    num_PCA_modes: np.array of int
        1-D np.array listing number of PCA modes to calculate when doing subtraction
        
    return: 3-D np.array of fl
        3-D datacube of PCA subtracted images. Should have shape (k, N_y, N_x) where k is the number
        of different modes we calculated for the image (i.e. the size of num_PCA_modes) and N_y, N_x are the
        input image dimensions in the y and x axes respectively
    """
    print('Performing PCA background subtraction using {} modes'.format(num_PCA_modes))
    #concatenate input image into 1-D array
    im_x = im.shape[1]
    im_y = im.shape[0]
    
    im = im.ravel()
    
    # reads list of reference frames into data matrix by first concatenating the 2-D .fits images
    # into 1-D arrays and then row stacking these images into a 2-D np.array
    ref_frames = np.stack([fits.open(ref_lib[i])[0].data.ravel() for i in range(len(ref_lib))], axis=0)
    
    # subtracts the mean of each reference frame from each reference frame 
    ref_frames_mean_sub = ref_frames - np.nanmean(ref_frames, axis=1)[:, None]
    ref_frames_mean_sub[np.where(np.isnan(ref_frames_mean_sub))] = 0
    
    # creates covariance matrix from mean subtracted reference frames 
    covar_psfs = np.cov(ref_frames_mean_sub)
    tot_basis = covar_psfs.shape[0]
    
    num_PCA_modes = np.clip(num_PCA_modes - 1, 0, tot_basis-1)  # clip values, for output consistency we'll keep duplicates
    max_basis = np.max(num_PCA_modes) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate
    
    # calculates eigenvalues and eigenvectors of the covariance matrix, but only the ones we need (up to max basis)
    evals, evecs = la.eigh(covar_psfs, eigvals=(tot_basis-max_basis, tot_basis-1))
    
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1], order='F') 
    
    # calculates the PCA basis vectors
    basis_vecs = np.dot(ref_frames_mean_sub.T, evecs)
    basis_vecs = basis_vecs * (1. / np.sqrt(evals * (np.size(im) - 1)))[None, :]  #multiply a value for each row
    
    #subtract off the mean of the input frame
    im_mean_sub = im - np.nanmean(im)
    
    # duplicate science image by the max_basis to do simultaneous calculation for different number of PCA modes
    im_mean_sub_rows = np.tile(im_mean_sub, (max_basis, 1))
    im_rows_selected = np.tile(im_mean_sub, (np.size(num_PCA_modes), 1)) # this is the output image which has less rows
    
    # bad pixel mask
    # do it first for the image we're just doing computations on but don't care about the output
    im_nanpix = np.where(np.isnan(im_mean_sub_rows))
    im_mean_sub_rows[im_nanpix] = 0
    # now do it for the output image
    im_nanpix = np.where(np.isnan(im_rows_selected))
    im_rows_selected[im_nanpix] = 0
    
    inner_products = np.dot(im_mean_sub_rows, np.require(basis_vecs, requirements=['F']))
    # select the KLIP modes we want for each level of KLIP by multiplying by lower diagonal matrix
    lower_tri = np.tril(np.ones([max_basis, max_basis]))
    inner_products = inner_products * lower_tri
    
    # make a model background for each number of basis vectors we actually output
    model = np.dot(inner_products[num_PCA_modes,:], basis_vecs.T)
    
    # subtract model from input frame for each number of PCA modes chosen
    PCA_sub_images = (im_rows_selected - model).reshape(np.size(num_PCA_modes), im_y, im_x)
    
    if type(num_PCA_modes) is np.int64:
        return PCA_sub_images[0]

    else type(num_PCA_modes) is np.ndarray:
        return PCA_sub_images


       
def correct_nonlinearity(image, n_coadd, nonlinearity_arr):
    assert np.shape(nonlinearity_arr) == np.shape(image)

    image_copy = np.array(image, dtype = float) #copy image
    image_copy /= n_coadd
    image_copy = (-1 + np.sqrt(1 + 4*nonlinearity_arr*image_copy)) / \
                 (2*nonlinearity_arr) #quadratic formula with correct root
    return image_copy * n_coadd

def PCA_subtraction(im, ref_lib, num_PCA_modes):
    """
    Does PCA subtraction of science frames using KLIP algorithm described in Soummer et al. (2012)

    ARGS:
    im: 2-D np.array
        2-D image to do PCA subtraction on
    ref_lib: list of str
        list of .fits files to serve as reference library for PCA subtraction
    num_PCA_modes: int or np.array of int
        1-D np.array listing number of PCA modes to calculate when doing subtraction
        

    OUTPUT:
    if num_PCA_modes is an int:
        return: 2-D np.array of fl
            2-D PCA subtracted image
            
    elif num_PCA_modes is an np.array of int
        return: 3-D np.array of fl
            3-D datacube of PCA subtracted images. Should have shape (k, N_y, N_x) where k is the number
            of different modes we calculated for the image (i.e. the size of num_PCA_modes) and N_y, N_x are the
            input image dimensions in the y and x axes respectively
    """
    print('Performing PCA background subtraction using {} modes'.format(num_PCA_modes))
    #concatenate input image into 1-D array
    im_x = im.shape[1]
    im_y = im.shape[0]
    
    im = im.ravel()
    
    # reads list of reference frames into data matrix by first concatenating the 2-D .fits images
    # into 1-D arrays and then row stacking these images into a 2-D np.array
    ref_frames = np.stack([fits.open(ref_lib[i])[0].data.ravel() for i in range(len(ref_lib))], axis=0)
    
    # subtracts the mean of each reference frame from each reference frame 
    ref_frames_mean_sub = ref_frames - np.nanmean(ref_frames, axis=1)[:, None]
    ref_frames_mean_sub[np.where(np.isnan(ref_frames_mean_sub))] = 0
    
    # creates covariance matrix from mean subtracted reference frames 
    covar_psfs = np.cov(ref_frames_mean_sub)
    tot_basis = covar_psfs.shape[0]
    
    num_PCA_modes = np.clip(num_PCA_modes - 1, 0, tot_basis-1)  # clip values, for output consistency we'll keep duplicates
    max_basis = np.max(num_PCA_modes) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate
    
    # calculates eigenvalues and eigenvectors of the covariance matrix, but only the ones we need (up to max basis)
    evals, evecs = la.eigh(covar_psfs, eigvals=(tot_basis-max_basis, tot_basis-1))
    
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1], order='F') 
    
    # calculates the PCA basis vectors
    basis_vecs = np.dot(ref_frames_mean_sub.T, evecs)
    basis_vecs = basis_vecs * (1. / np.sqrt(evals * (np.size(im) - 1)))[None, :]  #multiply a value for each row
    
    #subtract off the mean of the input frame
    im_mean_sub = im - np.nanmean(im)
    
    # duplicate science image by the max_basis to do simultaneous calculation for different number of PCA modes
    im_mean_sub_rows = np.tile(im_mean_sub, (max_basis, 1))
    im_rows_selected = np.tile(im_mean_sub, (np.size(num_PCA_modes), 1)) # this is the output image which has less rows
    
    # bad pixel mask
    # do it first for the image we're just doing computations on but don't care about the output
    im_nanpix = np.where(np.isnan(im_mean_sub_rows))
    im_mean_sub_rows[im_nanpix] = 0
    # now do it for the output image
    im_nanpix = np.where(np.isnan(im_rows_selected))
    im_rows_selected[im_nanpix] = 0
    
    inner_products = np.dot(im_mean_sub_rows, np.require(basis_vecs, requirements=['F']))
    # select the KLIP modes we want for each level of KLIP by multiplying by lower diagonal matrix
    lower_tri = np.tril(np.ones([max_basis, max_basis]))
    inner_products = inner_products * lower_tri
    
    # make a model background for each number of basis vectors we actually output
    model = np.dot(inner_products[num_PCA_modes,:], basis_vecs.T)
    
    # subtract model from input frame for each number of PCA modes chosen
    PCA_sub_images = (im_rows_selected - model).reshape(np.size(num_PCA_modes), im_y, im_x)

    if type(num_PCA_modes) is np.int64:
        return PCA_sub_images[0], model.reshape(im_y, im_x)
    elif type(num_PCA_modes) is np.ndarray:
        return PCA_sub_images, model.reshape(np.size(num_PCA_modes), im_y, im_x)
    
    else:
        print('Unsupported datatype for variable: num_PCA_modes. Variable must be either int or 1-D np.ndarray')
