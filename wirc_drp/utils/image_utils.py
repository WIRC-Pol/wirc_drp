# -*- coding: utf-8 -*-
"""
Created on Fri June 2 2017

@author: Kaew Tinyanont, Max Millar-Blanchaer, Ricky Nilsson

Imaging Utilities for the WIRC+Pol DRP

This file contains functions used to extract spectra from a cutout. 

"""

import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter, label, find_objects
# from scipy.signal import fftconvolve
import scipy.signal
from wirc_drp.constants import *
from wirc_drp.masks.wircpol_masks import *
from scipy.ndimage import gaussian_filter as gauss
# from scipy.ndimage.filters import median_filter, shift
from scipy.ndimage import median_filter, shift, rotate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import copy
from image_registration import chi2_shift
from wirc_drp import constants

from photutils import RectangularAperture, aperture_photometry,make_source_mask
from astropy.stats import sigma_clipped_stats

# import ccdproc

import cv2
# import pyfftw
# from numba import jit

def shift_figure(x_figs,y_figs=0):
    '''
    This function sets the position of the current figure on the screen. 
    Currently it just works by shifting the figure by integer numbers (x_figs, y_figs) of the figure width/height. 
    '''

    mgr = plt.get_current_fig_manager()
    py = mgr.canvas.height()
    px = mgr.canvas.width()
    mgr.window.setGeometry(px*x_figs, py*y_figs, px, py)



# @jit
# @profile
def locate_traces(science, sky = None, sigmalim = 5, plot = False, verbose = False, brightness_sort=True, update_w_chi2_shift=True, im_package = 'cv2', max_sources=5, use_full_frame_mask=True, force_figures = False, seeing = 0.75):
    """
    This is a function that finds significant spectral traces in WIRC+Pol science images. Search is performed in the upper left quadrant of image, and location of corresponding traces (and 0th order) in other three quadrants are calculated from assumed fixed distances. The function saves trace locations and thumbnail cutouts of all traces.
    Input:
        science: science image, either as np array or full path+filename
        sky: sky (offset) image to subtract, either as np array or full path+filename
        sigmalim: sigma limit for detection threshold above background noise
        plot: if True, UL quadrant of sky subtracted science_image is shown along with locations of the
                    traces found. Thumbnail cutouts of all traces are also plotted in a separate figure.
        brightness_sort: if True, then sort the sources according to their brightness. 
        update_w_chi2_shift: if True, then update the positions using the chi2_shift algo. The main reason for this is to get sub-pixel resolution. 
        use_full_frame_mas: If True then mask out the areas covered by the focal plane mask and by the bars of doom. 
        force_figures: if True, then when plotting it forces the first plot to be in Figure 1 and the second in Figure 2. 
        seeing: An estimate of the seeing in arcseconds. We will blur the template trace by this value. NOTE: This will have to be an odd multiple of 0.25. TODO make a check for this. 
                    
    Output: Dictionary of objects found. Each item contains of five keys with pairs of x and y coordinates (index starts at 0) of upper left, upper right, lower right, and lower left trace, and 0th order locations, as well as a flag set to True if trace is very noisy or crossing quadrant limit. The last item in the dictionary always contains the central hole/slit and trace locations 
    """    

    # TODO:
    # - Reactivate sort after brightest trace
    # - Implement for H images too

    UL_slit_trace = (573, 1024+500) # This should always be the same. Added even if not found.

    # Distance of 0th order, UR, LR, and LL traces to UL traces
    UR_diff = np.asarray(constants.dUR) - np.asarray(constants.dUL)
    LR_diff = np.asarray(constants.dLR) - np.asarray(constants.dUL)
    LL_diff = np.asarray(constants.dLL) - np.asarray(constants.dUL)
    spot0_diff = -np.asarray(constants.dUL)

    ############

    # MAIN CODE ###########

    # Load cropped and centered trace template image
    template_fn = constants.wircpol_dir+'wirc_drp/masks/single_trace_template2.fits'
    
    if verbose:
        print("Loading Template from {}".format(template_fn))

    trace_template_hdulist = fits.open(template_fn)
    trace_template = trace_template_hdulist[0].data
    trace_template = ndimage.median_filter(trace_template,int(seeing/constants.plate_scale))
    # # Plot trace template image
    # fig = plt.figure()
    # plt.imshow(trace_template, origin='lower')
    # plt.title('Trace template')

    # Cross-correlate image with itself
    # scipy.fftpack = pyfftw.interfaces.scipy_fftpack #Dropping in the pyfftw
    trace_selfcorr = scipy.signal.fftconvolve(trace_template,trace_template, mode='same')
    # pyfftw.interfaces.cache.enable()

    # Find best_match value. Normalize correlation image with this later. 
    best_match_val = np.max(trace_selfcorr)

    # Load sky (offset) image, either from file or as np array
    if isinstance(sky, str):
        sky_image_hdulist = fits.open(sky) 
        sky_image = sky_image_hdulist[0].data
        if verbose:
            print('Loading sky background '+ sky + ' ...')
    elif type(sky) == np.ndarray: #if array
        sky_image = sky.copy()
        if verbose:
            print('Using supplied sky background array ...')
    else: # sky == None:
        if verbose:
            print('No sky background image given. Assuming it has already been subtracted.')
    # Filter sky image to remove bad pixels
    if (im_package =='scipy') and (type(sky)==np.ndarray):
        sky_image_filt = ndimage.median_filter(sky_image,3) 
    elif (im_package =='cv2') and (type(sky)==np.ndarray):
        sky_image_filt = cv2.medianBlur(np.ndarray.astype(sky_image,'f'),3)    
   # plt.imshow(sky_image_filt)
    # plt.show()

    # Load science image, either from file or as np array
    if isinstance(science, str):
        science_image_hdulist = fits.open(science)
        science_image = science_image_hdulist[0].data
        if verbose:
            print('Loading science image '+ science + ' ...')
    else:
        science_image = science.copy()
        if verbose:
            print('Using supplied science array ...')

    # Filter science image to remove bad pixels
    if im_package == 'scipy':
        science_image_filt = ndimage.median_filter(science_image,3)
    elif im_package == 'cv2': #use cv2.
        science_image_filt = cv2.medianBlur(np.ndarray.astype(science_image,'f'),3)    
    # # Plot science image
    # fig = plt.figure()
    # plt.imshow(science_image_filt, origin='lower')
    # plt.show()
    # plt.title('Science image')

     # # If use_full_frame_mask
    if use_full_frame_mask:
        ffmask = fits.open(constants.wircpol_dir+'wirc_drp/masks/full_frame_mask.fits')[0].data
        fftmask = np.ndarray.astype(ffmask,bool)
        fftmask = fftmask[::-1,:]
        # science_image_filt[np.where(~fftmask)] = 0.
        # sky_image_filt[np.where(~fftmask)] = 0.
        science_image_filt[np.where(~fftmask)] = np.median(science_image_filt)
        if type(sky)==np.ndarray:
            sky_image_filt[np.where(~fftmask)] = np.median(sky_image_filt)
    #     med_sci = np.nanmedian(science_image_filt[fftmask])
    #     med_sky = np.nanmedian(sky_image_filt[fftmask])

    # else:
    #     med_sci = np.nanmedian(science_image_filt)
    #     med_sky = np.nanmedian(sky_image_filt)
        
    # Subtract sky image from science image -> Scale the sky so the medians of the two images match.
    if type(sky)==np.ndarray:
        stars_image = science_image_filt - sky_image_filt #*med_sci/med_sky
        if verbose:
            print('Subtracting sky image from science image ...')
    else:
        stars_image = science_image_filt

    # Cut out upper left quadrant of stars_image
    # stars_image_UL = np.array(stars_image[1024::,0:1023], copy=True)
    ylow = 924
    yhigh=2048
    xlow = 0
    xhigh = 1123

    stars_image_UL = np.array(stars_image[ylow::,xlow:xhigh], copy=True)

    # Cross-correlate trace template image with stars_image_UL
    corr_image_UL = scipy.signal.fftconvolve(stars_image_UL, trace_template, mode='same')

    # Calculate median and standard deviation in corr_image_UL. Exclude very low and very high pixel values (indicating sources)    
    corr_image_UL_med = np.median(corr_image_UL[(corr_image_UL < 2000) & (corr_image_UL > -2000)])
    corr_image_UL_std = np.std(corr_image_UL[(corr_image_UL < 2000) & (corr_image_UL > -2000)])

    # Threshold and mask corr_image_UL
    corr_image_UL_threshold = corr_image_UL_med + sigmalim * corr_image_UL_std
    diff = (corr_image_UL > corr_image_UL_threshold)
    corr_image_UL_masked = np.array(corr_image_UL, copy=True)
    corr_image_UL_masked[diff == 0] = 0
    corr_image_UL_masked[diff != 0] = 1

    # Label features in masked array
    labeled, num_objects = ndimage.label(corr_image_UL_masked.astype(int))

    # Find "objects" in labeled array
    traces = ndimage.find_objects(labeled)

    # Lists for saving x and y coordinates of peaks in each correlation image
    x_locs, y_locs = [], []

    # Get trace coordinates and add to x and y lists
    for dy,dx in traces:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2 + ylow

        size = trace_template.shape[0]/2
        cutout = stars_image_UL[int(y_center-size-ylow):int(y_center+size-ylow),int(x_center-size):int(x_center+size)]
        
        if update_w_chi2_shift:
            try:
                shifts = chi2_shift(cutout,trace_template, zeromean=True, verbose=False, return_error=True, boundary='constant')

                #Sometimes if it's too big the whole thing gets shifted out and it breaks things. 
                if (np.abs(shifts[0]) < 10 and np.abs(shifts[1]) < 10):
                    x_center -= shifts[0]
                    y_center -= shifts[1]
                
                ## Debugging plots
                # fig = plt.figure(figsize=(7,7))
                # ax1 = fig.add_subplot(141)
                # plt.imshow(cutout)
                # ax2 = fig.add_subplot(142)
                # plt.imshow(trace_template)
                # ax3 = fig.add_subplot(143)
                # cutout2 = stars_image_UL[np.floor(y_center-size-1024).astype(int):np.floor(y_center+size-1024).astype(int),np.floor(x_center-size).astype(int):np.floor(x_center+size).astype(int)]
                # plt.imshow(cutout2,alpha=0.5,cmap='magma')
                # ax4 = fig.add_subplot(144)
                # plt.imshow(shift(cutout,(shifts[1],shifts[0])))

            except Exception as e:
                if verbose:
                    print(e)

        x_locs.append(x_center)
        y_locs.append(y_center)


    # Trace locations array with all coordinates
    locs_UL = np.array([x_locs, y_locs])

    # Add slit trace position to trace locations. RN: Not adding slit trace per default anymore. If source is there, locate_traces() should pick it up. 
    #locs_UL = np.append(locs_UL, np.swapaxes(np.array([UL_slit_trace]),0,1), 1)

    # Calculate location of corresponding traces (and 0th order) in other three quadrants
    locs_UR = locs_UL + np.swapaxes(np.array([UR_diff]),0,1)
    locs_LR = locs_UL + np.swapaxes(np.array([LR_diff]),0,1)
    locs_LL = locs_UL + np.swapaxes(np.array([LL_diff]),0,1)
    locs_spot0 = locs_UL + np.swapaxes(np.array([spot0_diff]),0,1)

    n_sources = len(locs_spot0.T)
    trace_diag_flag = [False] * n_sources

    if verbose:
        print('Found ' + str(n_sources) + ' sources')
    #source_ok = trace_checker(stars_image, source_list_pre, verbose = True)

    pix_vals_UL=[] 
    #Do we want to sort the sources by their brightness? 
    if brightness_sort: 
        # Now we'll calculate the pixel value at each x,y value
        for i in range(np.shape(locs_UL)[1]):
            pix_vals_UL.append(science_image_filt[np.floor(locs_UL[0,i]).astype('int'),np.floor(locs_UL[1,i]).astype('int')]+\
                science_image_filt[np.floor(locs_UR[0,i]).astype('int'),np.floor(locs_UR[1,i]).astype('int')]+\
                science_image_filt[np.floor(locs_LR[0,i]).astype('int'),np.floor(locs_LR[1,i]).astype('int')]+\
                science_image_filt[np.floor(locs_LL[0,i]).astype('int'),np.floor(locs_LL[1,i]).astype('int')])
        pix_vals_UL = np.array(pix_vals_UL)
        pix_vals_argsort = np.argsort(pix_vals_UL)[::-1]
        # Now reorder locs_UL so that it's according to pix_vals_UL
        locs_UL = np.array([[locs_UL[0,i],locs_UL[1,i]] for i in pix_vals_argsort]).T
        locs_UR = np.array([[locs_UR[0,i],locs_UR[1,i]] for i in pix_vals_argsort]).T
        locs_LL = np.array([[locs_LL[0,i],locs_LL[1,i]] for i in pix_vals_argsort]).T
        locs_LR = np.array([[locs_LR[0,i],locs_LR[1,i]] for i in pix_vals_argsort]).T
        locs_spot0 = np.array([[locs_spot0[0,i],locs_spot0[1,i]] for i in pix_vals_argsort]).T

    # Flag suspicious traces by checking mid-diagonals. This currently only checks diagonals of UL trace. Check UL, UR, LL, and LR by running trace_checker() separately instead!
    trace_diag_val = []
    trace_diag_flag = []
    
    for n in range(0,locs_UL.shape[1]):
        thumbn = stars_image[int(round(locs_UL[1,n]))-50:int(round(locs_UL[1,n]))+50, int(round(locs_UL[0,n]))-50:int(round(locs_UL[0,n]))+50]
        diag_val = []
        for diag_offset in range(-10,11):
            diag = np.diagonal(np.flipud(thumbn), diag_offset).copy()
            diag_val.append(np.sum(diag))
        opt_diag_offset = diag_val.index(max(diag_val)) - 10
        #print(opt_diag_offset)
        diag0 = np.diagonal(np.flipud(thumbn), opt_diag_offset).copy()
        diag_plus = np.diagonal(np.flipud(thumbn), opt_diag_offset+1).copy()
        diag_minus = np.diagonal(np.flipud(thumbn), opt_diag_offset-1).copy()
        full_diag = np.concatenate((diag0[20:-20], diag_plus[20:-20], diag_minus[20:-20]), axis=0)
        #norm_diag = full_diag / np.max(full_diag)
        trace_diag_val.append(np.median(full_diag))
        td_sig = 3
        td_thres = np.median(thumbn)+td_sig*np.std(thumbn)
        if (np.median(full_diag) > td_thres) & (np.median(np.flipud(thumbn[0:19,80:99]).diagonal(opt_diag_offset).copy()) < td_thres) & (np.median(np.flipud(thumbn[80:99,0:19]).diagonal(opt_diag_offset).copy()) < td_thres):
            trace_diag_flag.append(False)
        else:
            trace_diag_flag.append(True)
            
    print(trace_diag_val, '\n', trace_diag_flag)

    # Put all the good traces at the top.
    args = np.argsort(trace_diag_flag)
    locs_UL = locs_UL[:,args][:,:max_sources]
    locs_UR = locs_UR[:,args][:,:max_sources]
    locs_LL = locs_LL[:,args][:,:max_sources]
    locs_LR = locs_LR[:,args][:,:max_sources]
    locs_spot0 = locs_spot0[:,args][:,:max_sources]
    trace_diag_flag = np.array(trace_diag_flag)[args][:max_sources]


    # Gather all trace and 0th order locations in a dictionary
    locs = {'UL': locs_UL, 'UR': locs_UR, 'LR': locs_LR, 'LL': locs_LL, 'spot0': locs_spot0, 'flag': trace_diag_flag}

    # # Show location of all found traces in stars_image
    if plot == True:
        plt.ion() # Turning on interactive mode for plotting (shows figure and releases terminal prompt) 
        
        # Plot UL quadrant of sky subtracted science image with found traces labelled
        if force_figures:
            f = plt.figure(2, figsize=(6.4,4.8))
            plt.clf()
        else:
            f = plt.figure()

        plt.imshow(stars_image, origin='lower', clim=(0,np.median(stars_image_UL)+sigmalim*np.std(stars_image_UL)))
        plt.colorbar()
        plt.xlim(xlow,xhigh)
        plt.ylim(ylow,yhigh)
        for n in range(0,locs_UL.shape[1]):
            if trace_diag_flag[n] == True:
                #plt.scatter(locs_UL[0,n],locs_UL[1,n],color='white',marker='o')
                plt.annotate('Trace '+str(n+1), (locs_UL[0,n],locs_UL[1,n]),color='red')
            else:
                #plt.scatter(locs_UL[0,n],locs_UL[1,n],color='white',marker='o')
                plt.annotate('Trace '+str(n+1), (locs_UL[0,n],locs_UL[1,n]),color='white')
        # if trace_diag_flag[locs_UL.shape[1]-1] == True:
        #     #plt.scatter(locs_UL[0,n],locs_UL[1,n],color='white',marker='o')
        #     plt.annotate('Slit trace', (locs_UL[0,locs_UL.shape[1]-1],locs_UL[1,locs_UL.shape[1]-1]),color='red')
        # else:
        #     #plt.scatter(locs_UL[0,n],locs_UL[1,n],color='white',marker='o')
        #     plt.annotate('Slit trace', (locs_UL[0,locs_UL.shape[1]-1],locs_UL[1,locs_UL.shape[1]-1]),color='white')

        plt.draw()

    if verbose:
        print('\n')
        print('UL quadrant trace locations:\n',locs['UL'].T,'\n')
        print('UR quadrant trace locations:\n',locs['UR'].T,'\n')
        print('LR quadrant trace locations:\n',locs['LR'].T,'\n')
        print('LL quadrant trace locations:\n',locs['LL'].T,'\n')

    # Number of sources
    n_sources = len(locs['UL'][0])

    return locs

def update_location_w_chi2_shift(image, x, y, filter_name = 'J',seeing = 0.75, verbose = False, slit_pos = 'slitless', trace_template = None):
    """
    This function grabs the upper left cutout from given x,y location of the zeroth order, then uses chi2_shift to align it
    with a trace template, then spits out the new x, y location that will center the trace. 

    If template == None, use the default 150x150 px template. 
    """
    # Load cropped and centered trace template image

    if trace_template is None:
        template_fn = constants.wircpol_dir+'wirc_drp/masks/single_trace_template2.fits'
        
        if verbose:
            print("Loading Template from {}".format(template_fn))

        trace_template_hdulist = fits.open(template_fn)
        trace_template = trace_template_hdulist[0].data[:-1,:-1] #trim one to satisfy cutout_trace_thumbnails
        trace_template = ndimage.median_filter(trace_template,int(seeing/constants.plate_scale))
    else:
        pass

    # Grab top left trace cutout
    UL_trace = cutout_trace_thumbnails(image, np.expand_dims([[y,x], slit_pos],axis=0) , flip = False, filter_name = 'foo',
            cutout_size = int(trace_template.shape[0]/2), sub_bar = False, mode = 'pol', verbose = False)[0][0] #just take the first ones

    try:
        shifts = chi2_shift(median_filter(UL_trace,3),trace_template, zeromean=True, verbose=False, return_error=True, boundary='constant')
        if verbose:
            print("Shfits are x,y = ", shifts)
        #Sometimes if it's too big the whole thing gets shifted out and it breaks things. 
        if (np.abs(shifts[0]) < 10 and np.abs(shifts[1]) < 10):
            x -= shifts[0]
            y -= shifts[1]
        
        ## Debugging plots
        # fig = plt.figure(figsize=(7,7))
        # ax1 = fig.add_subplot(141)
        # plt.imshow(cutout)
        # ax2 = fig.add_subplot(142)
        # plt.imshow(trace_template)
        # ax3 = fig.add_subplot(143)
        # cutout2 = stars_image_UL[np.floor(y_center-size-1024).astype(int):np.floor(y_center+size-1024).astype(int),np.floor(x_center-size).astype(int):np.floor(x_center+size).astype(int)]
        # plt.imshow(cutout2,alpha=0.5,cmap='magma')
        # ax4 = fig.add_subplot(144)
        # plt.imshow(shift(cutout,(shifts[1],shifts[0])))

        return x, y

    except Exception as e:
        if verbose:
            print(e)
        return None





def check_traces(full_image, wo_source_list, verbose = False):

    '''
    Flag sources with suspicious traces by checking mid-diagonals.

        Input:
            full_image: full calibrated science (PG) image
            wo_source_list: wircpol_object source list OR locs list
        Output:
            (source_ok, source_brightness): tuple containing a list (source_ok) with True for sources with good traces and False for sources with bad traces, and a list (source_brightness) with a brighness metric for the sources.
    '''

    if isinstance(wo_source_list, dict):
            nsources = len(wo_source_list['spot0'].T)
    else:
            nsources = len(wo_source_list)
    quads_diff = [constants.dUL, constants.dUR, constants.dLR, constants.dLL] # offsets from zeroth order
    quads = ['UL', 'UR', 'LR', 'LL']
    trace_count = 0
    plt.ion()

    source_brightness = []
    source_ok = [] # True/False list for nsources
    for source in range(nsources):
        if isinstance(wo_source_list, dict):
            zeroth_loc = wo_source_list['spot0'][:,source]
        else:
            zeroth_loc = wo_source_list[source].pos
        if verbose:
            print('Checking traces for source '+ str(source+1) + ' at location ' + str(zeroth_loc))
            plt.figure(figsize=(8,8))
        # for UL
        trace_diag_val = [] # Value list for trace in each quad
        trace_diag_ok = [] # True/False list for trace in each quad
        for quad in range(len(quads)):
            trace_count += 1
            trace_loc = (zeroth_loc[0] + quads_diff[quad][0], zeroth_loc[1] + quads_diff[quad][1]) # trace locs with rounded offsets
            thumbn = full_image[int(round(trace_loc[1]-50)):int(round(trace_loc[1]+50)), int(round(trace_loc[0]-50)):int(round(trace_loc[0]+50))]
            if quads[quad] == 'UL':
                thumbn_orient = np.flipud(thumbn)
            elif quads[quad] == 'UR':
                thumbn_orient = np.fliplr(np.flipud(thumbn))
            elif quads[quad] == 'LR':
                thumbn_orient = np.fliplr(thumbn)
            elif quads[quad] == 'LL':
                thumbn_orient = thumbn
            diag_val = []
            for diag_offset in range(-10,11):
                diag = np.diagonal(thumbn_orient, diag_offset).copy()
                diag_val.append(np.sum(diag))
                opt_diag_offset = diag_val.index(max(diag_val)) - 10
                #print(opt_diag_offset)
                diag0 = np.diagonal(thumbn_orient, opt_diag_offset).copy()
                diag_plus = np.diagonal(thumbn_orient, opt_diag_offset+1).copy()
                diag_minus = np.diagonal(thumbn_orient, opt_diag_offset-1).copy()
                full_diag = np.concatenate((diag0[20:-20], diag_plus[20:-20], diag_minus[20:-20]), axis=0)
                #norm_diag = full_diag / np.max(full_diag)
                full_diag_med = np.median(full_diag)
                trace_diag_val.append(full_diag_med)
                td_sig = 3
                td_thres = np.median(thumbn)+td_sig*np.std(thumbn)
            if (full_diag_med > td_thres) & (np.median(thumbn_orient[0:19,80:99].diagonal(opt_diag_offset).copy()) < td_thres) & (np.median(thumbn_orient[80:99,0:19].diagonal(opt_diag_offset).copy()) < td_thres):
                trace_diag_ok.append(True)
                framecol = 'green'
                if verbose:
                    print('Looks alright!')
            else:
                trace_diag_ok.append(False)
                framecol = 'red'
                if verbose:
                    print('No good!')
            if verbose:
                print('Checking diagonal for '+ quads[quad] + ' trace ...')
                plt.subplot(nsources,5,trace_count)
                plt.imshow(thumbn, origin='lower')
                ax = plt.gca()
                #plt.setp(ax.spines.values(), color=framecol, linewidth=3)
                plt.title('Source ' + str(source+1) + ' ' + quads[quad])
        trace_count += 1
        thumb0 = full_image[int(round(zeroth_loc[1]-50)):int(round(zeroth_loc[1]+50)), int(round(zeroth_loc[0]-50)):int(round(zeroth_loc[0]+50))]
        if verbose:
            plt.subplot(nsources,5,trace_count)
            plt.imshow(thumb0, origin='lower')
            ax = plt.gca()
            #plt.setp(ax.spines.values(), color=framecol, linewidth=3)
            plt.title('Source ' + str(source+1) + ' 0th')
            print('\n')
        if all(trace_diag_ok):
            source_ok.append(True)
        else:
            source_ok.append(False)
        source_brightness.append(np.sum(trace_diag_val))
        
    return(source_ok, source_brightness)


def find_sources_in_direct_image(direct_image, mask, threshold_sigma, guess_seeing, plot = False):
    #Previously named coarse_regis
    """
    This is the main function to perform coarse registration to find all traces
    in the image by using a direct image (i.e. without the polarization grating)
    Input: direct_image: the array of direct image
            mask: the array representing the mask used in the observation
            threshold_sigma: the threshold of detection above background noise
            guess_seeing: the approximated seeing value in PIXEL (not arcsec)
            plot: if True, the direct_image is shown along with locations of the
                    objects found.
                    
    Output: An array whose length equals to the number of sources found
            Each element is a list with 2 elements. The first element is an array
            (index, x, y, x_err, y_err) and the second element tells the number 
            of slit the source is in or if it's in the slitless area. 
    """    

    locations = []
    if type(direct_image) == np.ndarray:
        
        #First find sources in the slitless area        
        
        #Crop direct_image to select the open part, then apply mask to avoid the slit
        #source in slit will be treated separately
        direct_image = np.nan_to_num(direct_image) #remove nans
        
        ####Commented out for now. For some examples, there's no need to do the slit
        ####and slitless sources separatedly.

    #    slitless_area = np.copy(direct_image)

    #    y_shift = int(slit_position_y - 1024)
    #    slitless_area[int(fov_size-fov_size/2+y_shift):int(fov_size+fov_size/2+y_shift),int(fov_size-fov_size/2):int(fov_size+fov_size/2)]\
    #                = slitless_area[int(fov_size-fov_size/2+y_shift):int(fov_size+fov_size/2+y_shift),int(fov_size-fov_size/2):int(fov_size+fov_size/2)] * mask #mask out area under spider and in slit
       
    #    #find sources in the direct image, assuming seeing of 4 pixel, 3sigma threshold, and no output file

    #    locations_slitless = sourcesInIm(slitless_area, seeing_pix = guess_seeing, threshold_sigma = threshold_sigma, \
    #                out_file = None)
       

       
    #    # print(locations)
    #    #locations is a list of (y,x) of sources detected in this image.    
    #    if len(locations_slitless) == 0:
    #        print('No sources found')
    #    print(len(locations_slitless),' sources in slitless area')
       
    #    #show the results
    #    if plot:
    #        plt.figure(1,figsize = (10,10))
    #        ax = plt.subplot()
    #        ax.imshow(slitless_area, origin = 'lower')
    #        for i in locations_slitless:
    #            source = plt.Circle((i[1],i[0]), 10, fill = False, color = 'w')
    #            ax.add_artist(source)
    #        plt.xlim([int(fov_size/2),int(1.5*fov_size)])
    #        plt.ylim([int(fov_size/2),int(1.5*fov_size)])
       
    #        plt.show()

    # Second, find sources inside the slits
        
        slit_area = np.copy(direct_image) #Just do all of the image.
       # slit_area[int(fov_size-fov_size/2+y_shift):int(fov_size+fov_size/2+y_shift),int(fov_size-fov_size/2):int(fov_size+fov_size/2)]\
       #             = slit_area[int(fov_size-fov_size/2+y_shift):int(fov_size+fov_size/2+y_shift),int(fov_size-fov_size/2):int(fov_size+fov_size/2)]  * ~mask #technically include area under spider. should be dark

       # bkg = np.median(slit_area[~mask])
       # bkg = 80
       # print(bkg)
       # #mask out area beyond the slit mask        
       # slit_area[0:int(slit_position_y-slit_hole_diameter),:] = bkg
       # slit_area[int(slit_position_y+slit_hole_diameter):,:] = bkg
       # slit_area[:, 0:int(slit_position_x-3*slit_hole_diameter)] = bkg
       # slit_area[:, int(slit_position_x+3*slit_hole_diameter): ] = bkg
        locations_slit = find_point_sources(slit_area, seeing_pix = guess_seeing, threshold_sigma = threshold_sigma, \
                    out_file = None, plot=plot)    
        
        slit_with_obj = [] #track slits with objects
        #see which one they are in. Hard code location for slits for now. Can modify later.
        for i in locations_slit:
            #note which slit the object is in by looking at the 
            if i[1] < slit_position_x + slit_hole_diameter/2 and i[1] > slit_position_x - slit_hole_diameter/2 \
                    and i[0] < slit_position_y - slit_hole_diameter and i[0] > slit_position_y +slit_hole_diameter:
                #print('slit 1')
                slit_with_obj += [1] #central slit
            elif i[1] < slit_position_x + slit_hole_gap + slit_hole_diameter/2 \
                    and i[1] > slit_position_x + slit_hole_gap - slit_hole_diameter/2 \
                    and i[0] < slit_position_y -slit_hole_diameter and i[0] > slit_position_y +slit_hole_diameter:
                #print('slit 0')
                slit_with_obj += [0] #top slit
            elif i[1] < slit_position_x - slit_hole_gap + slit_hole_diameter/2 \
                    and i[1] > slit_position_x - slit_hole_gap - slit_hole_diameter/2 \
                    and i[0] < slit_position_y -slit_hole_diameter and i[0] > slit_position_y +slit_hole_diameter:
                #print('slit 2')
                slit_with_obj += [2] #bottom slit 
            else:
                slit_with_obj += ['slitless']
                #print('nope')
        #remove bogus sources
        slit_with_obj = np.array(slit_with_obj)
        #locations_slit = np.array(locations_slit)[ slit_with_obj != -99]
        #slit_with_obj = slit_with_obj[slit_with_obj != 99]
        print(len(locations_slit), ' sources in slit.')
        #plotting
        if plot:
            plt.figure(1,figsize = (10,10))
            ax = plt.subplot() 
            ax.imshow(slit_area, origin = 'lower')
            
            for i in locations_slit:
                source = plt.Circle((i[1],i[0]), 10, fill = False, color = 'w')
                ax.add_artist(source)  

            plt.ylim([int(fov_size /2),int(1.5* fov_size )])
            plt.xlim([int(fov_size /2),int(1.5* fov_size )])
            # plt.show()
        #print('slit with obj', slit_with_obj, locations_slit)
            
        #put them together
        for i in range(len(locations_slit)):
            locations += [ np.array([locations_slit[i], slit_with_obj[i]]) ]
       # for i in range(len(locations_slitless)):
       #     locations += [ np.array([locations_slitless[i], 'slitless']) ]

    else:
        print("No Direct Image found. Need to find sources some other way!")

    locations = np.array(locations)
    ###Print some status    
    ###Now we have the locations of all sources in this fov 
    print( "Found %i point sources" % np.shape(locations[:,0])[0])
    # print "index\tx\ty\tx_stddev\ty_stddev"
    print( "{0:<8} {1:<13} {2:<13} {3:<13} {4:<13} {5:<8}".format("index", "x", "y", "x_stddev", "y_stddev", "slit_number"))



    for i,location in enumerate(locations[:,0]):
        # print "%i \t %3.3f \t %3.3f \t %1.2f \t %1.2f" % (i,location[1], location[0], location[2], location[3])
        print( '{0:<8} {1:<13.2f} {2:<13.2f} {3:<13.2f} {4:<13.2f} {5:<8}'.format(i,location[1], location[0], location[2], location[3],str(locations[i,1])))

    stddev_est = np.mean([np.vstack(locations[:,0])[:,2],np.vstack(locations[:,0])[:,3]])
    print( '\nSeeing FWHM Estimate: %0.2f"' % (2*stddev_est * sampling))

        
    return locations

def find_point_sources(image, seeing_pix = 4, threshold_sigma = 3, size_lim = 5, out_file = None, plot=False):
    """sourceInIm takes a direct image and find all sources above some detection
    threshold in it.

    Used by find_sources_in_direct_image

    Inputs:
        image: an array representing the direct image
        seeing_pix: seeing size in pixel
        threshold_sigma: detection threshold in sigmas away from the standard deviation in background fluctuation
        size_lim: the limit of stddev, in pixel, over which we don't accept the fit as a source        
        out_file: If defined then a .reg file for ds9 is created in the XY format. 
    """

    threshold = threshold_sigma*np.sqrt(np.var(np.nan_to_num(image)))
    
    #get all cutouts
    cutouts = pointFinder(image, seeing_pix, threshold)
    #list to collect results
    all_results = []
    #loop through cutouts, fit gaussian to get position of sources
    for i in cutouts:
        #make cutouts from indices, then run fit_gaussian_to_cutout
        if np.logical_and(len(np.ravel(image[i])) < 200, len(np.ravel(image[i])) > 10 ): #a legit cutout shouldn't be large
            
            if plot: 
                plt.imshow(image[i], interpolation = 'none')
            
            res = fit_gaussian_to_cutout(image[i], seeing_pix)
            #get x y
            x = (i[1].start)+res[0].x_mean.value
            y = (i[0].start)+res[0].y_mean.value
            x_stddev = res[0].x_stddev.value
            y_stddev = res[0].y_stddev.value
            #filter out bad fits. stddev should be of some reasonable number
            if x_stddev < size_lim and y_stddev < size_lim:
                all_results+=[(y,x, y_stddev, x_stddev)]            
        else:
            None
            #print(i, ' is invalid.' )
    #return cutouts
    if out_file != None:
        f = open(out_file, mode = 'w')
        for i in all_results:
            f.write(str(i[1]+1)+'  '+str(i[0]+1)+'\n')
        f.close()
    return all_results

def fit_gaussian_to_cutout(cutout, seeing_pix):
    """Fit a 2D gaussian to the cutout to determine the location of the maximum
    Inputs: 
        cutout: a small array representing a cutout from the direct image.
        seeing_pix: a guess of the seeing size in pixel for gaussian fit
    Output:
        res: result of the 2D polynomial (res[0]) and gaussian fit (res[1]). 
                The location of the source is determined by the gaussian part, of course.
                (MMB: Are these backwards? e.g. gaussian is res[0]?)
                """

    g_init = models.Gaussian2D(amplitude = np.max(cutout), \
                                x_mean = np.shape(cutout)[1]/2,y_mean = np.shape(cutout)[0]/2,\
                                x_stddev = seeing_pix, y_stddev = seeing_pix)
    #allow for some DC offset
    const_init = models.Polynomial2D(2)
    #Use LevMar LSQ fitter
    fitter = fitting.LevMarLSQFitter()
    #Just get x,y grid for the cutout
    y,x = np.mgrid[:np.shape(cutout)[0],:np.shape(cutout)[1]]   
    #fit
    res = fitter(g_init+const_init, x,y, cutout)
    #We want x,y location from the Gaussian part
    #print(res[0].amplitude.value, res[0].x_stddev.value)
    return res


def pointFinder(image, seeing_pix, threshold):
    """Take an image file and identify where point sources are. This is done by utilizing
    Scipy maximum and minimum filters.
    Inputs:
        image: an array of the image 
        seeing_pix: a guess of the seeing size in pixel used to define the size of max/min filters
        threshold: a threshold (in counts) used to determine a detection from noise fluctuation
    Output:
        cutouts: list of cutouts of sources found in this image."""
    #First, find maxima in the image
    #peaks = maximum_filter(image, footprint = pattern)#size = seeing_pix) #filter to find max
    peaks = maximum_filter(image,size = 0.1*seeing_pix)
    maxima = (peaks == image) #booleen array indicating locations of the maxima
    #now, compute the minimum for background subtraction. This will get rid of 
    #"maxima" that are really just noise
    troughs = minimum_filter(image, size = 4*seeing_pix) #to find min. make sure we erase the traces
    #now make sure that the real maxima are large enough from the backfround
    diff = (peaks-troughs) > threshold
    #label this such that every clump of '1's are labelled. Scipy magic!
    labeled, num_obj = label(diff)#, structure = pattern)
    cutouts = find_objects(labeled) #get cutouts of area with objects    
    return cutouts

def locationInIm(wl, location_in_fov, filter_name = 'J'):
    """compute for a source at the location_in_fov (y,x) in pixel scale, where
    4 traces would land in the actual detector.
    Outputs: A list of [x,y] location for [Q1, Q2, U1, U2], rounded

    Used by cutout_trace_thumbnails
    """
    #Use measured values from test data
    if filter_name == 'J':
        l0 = 1.2483
    elif filter_name == 'H':
        l0 = 1.6313
    dwl = wl-l0 #This compute the deviation from J band where the offsets were measured
    dpx = round(dwl/(wlPerPix))

    traceLocation = [ [ 453+location_in_fov[0]+dpx, -435 + location_in_fov[1]-dpx],\
                    [  -455+location_in_fov[0]-dpx,  455+ location_in_fov[1]+dpx], \
                    [   450+location_in_fov[0]+dpx,  455+location_in_fov[1]+dpx], \
                    [  -445+location_in_fov[0]-dpx, -455+location_in_fov[1]-dpx]]
    return np.array(traceLocation)

    #Functions for spectral image

def cutout_trace_thumbnails(image, locations, flip = True, filter_name = 'J', sub_bar = True, mode = 'pol', cutout_size = None, verbose=False):
    '''
    This function Extracts the thumbnails of each trace for a given image give a locations list. 
    image - the image where you want to extract the traces
    locations - the locations in the image that you want to use as a basis for extraction [y,x] format
    flip - An optional switch that allows you to flip all the trace thumbnails to be orientated in the same direction 
            (i.e. wavelength increasing in the same direction)
    filter_name  - the filter. This determines the cutout size.
    mode - use either 'pol' or 'spec'.  If set to spec, return cutouts at positions of input positions
    cutout_size - instead of auto-selecting cutout size, allow this as input. Leave as None if you want this auto-selected
    '''


    if mode == 'pol':
        if filter_name == 'J':
            if cutout_size == None:
                cutout_size = 80 #Make cutout of each trace. This has to chage for J/H bands: was 80, then 150, now 80 agian.
            lb = J_lam
        elif filter_name == 'H':
            if cutout_size == None:
                cutout_size = 200 #was 150
            lb = H_lam
        else:
            if verbose:
                print('Filter name %s not recognized, assuming J, and use given cutout_size' %filter_name)
            if cutout_size == None:
                cutout_size = 80
            lb = J_lam


    if mode == 'spec':
        if cutout_size is None:
            print('Need to define cutout size for wirc_spec object')

    cutouts = [] #A list of thumbnails lists

    for k,i in enumerate(locations[:,0]):

        thumbnails = [] #A thumbnail list of each traces
        
        if mode == 'pol':
            traceLocation = locationInIm(lb, i).astype(int) #These are locations in Q+,Q-,U+,U- respectively. 
            #print(traceLocation)        

        if mode == 'spec':
            traceLocation = np.array([i]) #These are locations in Q+,Q-,U+,U- respectively.
            trace_title='Source '+np.str(k+1)
            
        ###This part is just to get show 4 traces of a source
        ntraces = len(traceLocation) #how many traces? 4 for WIRC-POL

        for j in range(ntraces):

            ylow = int(traceLocation[j][0]-cutout_size)
            yhigh = int(traceLocation[j][0]+cutout_size+1)
            xlow = int(traceLocation[j][1]-cutout_size)
            xhigh = int(traceLocation[j][1]+cutout_size+1 )

            image_new = copy.deepcopy(image)

            #Checking and compensating for out of bounds
            if (ylow < 0) or (xlow < 0) or (yhigh > 2048) or (xhigh > 2048):
                
                pad_width = np.max([(0.-ylow),(0.-xlow), (yhigh-2048), (xhigh-2048)]).astype(int)
                image_new = np.pad(image_new, pad_width, 'constant')

                if True:
                    print("Cutout will slice outside of array, padding {} pixels with zeros".format(pad_width))

                # if (ylow < 0.):
                ylow += pad_width
                # if (xlow < 0):
                xlow += pad_width
                # if yhigh > 2048:
                yhigh += pad_width
                # if xhigh > 2048:
                xhigh += pad_width

            #The indices for the cutout
            cutout = np.s_[ylow:yhigh,xlow:xhigh]
        
            #cut the spectral image into a thumbnail containing the trace
            thumbnail = copy.deepcopy(image_new)[cutout]
                       
            #flip the thumbnail so that it's in the Q+ orientation (from top left to bottom right)
            if flip and mode=='pol': 
                trace_title = "Top Left (Q+)"
                if j == 1: #Q-
                    thumbnail = thumbnail[-1::-1, -1::-1] #flip y, x
                    trace_title = "Bottom Right (Q-)"
                    # print(np.shape(thumbnail))
                elif j == 2: #U+
                    thumbnail = thumbnail[:,-1::-1] #flip x
                    trace_title = "Top Right (U+)"
                    # print(np.shape(thumbnail))
                elif j == 3: #U-
                    thumbnail = thumbnail[-1::-1, :] #flip y 
                    trace_title = "Bottom Left (U-)"
                    # print(np.shape(thumbnail))

            if flip and mode=='spec':
                #rotate image by 90 degrees
                thumbnail = np.rot90(thumbnail[-1::-1, -1::-1])

            if filter_name == 'J' and mode=='pol':
                if sub_bar:
                    #Check to see if the traces hit the vertical bars of doom
                    if ((i[1] > slit_position_x+300) & (i[1] < slit_position_x+600)) | ((i[1] < slit_position_x-300) & (i[1] > slit_position_x-600)):
                        
                        if j == 0: 
                            print("Source {}'s traces will hit the vertical bar of doom, compensating by subtracting the median of the edges of each row".format(k+1))

                        for xind in range(np.shape(thumbnail)[1]):
                            sub_length = 20 #The number of pixels at the beginning and end to estimate the background
                            thumbnail[:,xind] -= np.nanmedian(np.concatenate([thumbnail[:sub_length-1,xind],thumbnail[-(sub_length):,xind]]))


                    
                    if ((i[0] > slit_position_y+300) & (i[0] < slit_position_y+600)) | ((i[0] < slit_position_y-300) & (i[0] > slit_position_y-600)):
                        if j == 0: 
                            print("Source {}'s traces will hit the horizontal bar of doom".format(k+1))

                        for yind in range(np.shape(thumbnail)[0]):
                            sub_length = 20 #The number of pixels at the beginning and end to estimate the background
                            thumbnail[yind,:] -= np.nanmedian(np.concatenate([thumbnail[yind,:sub_length-2],thumbnail[yind,-(sub_length):]]))


            thumbnails.append(thumbnail)

        cutouts.append(thumbnails)

    return cutouts

def shift_and_subtract_background(cutout, obj_slit = 1,  slit_gap = 21, masked_slit = None, plot = False):
    """
    Background subtraction for source in slit. This function averages two sky
    slits (empty), then subtract it from the object slit. The three slits are 
    number 0, 1, 2 from top to bottom.
    
    Inputs: cutout: the array containing the thumbnail of a trace under the slit mask
            obj_slit: number of the slit in which the object is. Default to 1, middle slit. 
                      This should be detected automatically in the coarse regis step. 
            slit_gap: the gap between slit in pixel. This should be determined from flat field
                      but is default to 21 pix.
            masked_slit: number of the slit that should be masked out due to other contaminating sources
    Output: sky_sub: an array of background subtracted trace shifted to the middle
            sky: the sky frame used to do subtraction. 
    """
    if int(obj_slit) not in [0,1,2]:
        print('Object slit must be 0 (top), 1 (middle), or 2 (bottom)')
        print('Object slit provided: {}'.format(obj_slit))
        return None
    else:
        #loop through the trace from right to left
        width = len(cutout[0]) #we have square cutout
        #buffer = round((width - trace_length -5)/2) #buffer area on either ends of the trace
        buffer = int(round(0.85*slit_length/2)) #imported from constant
        
        #dx0, 2 is here for place holder now. This is in case the three holes are not vertical
        dx0 = 0
        dx2 = 0
        #trace0 = shift(cutout, [-slit_gap,dx0], order = 4)
        trace0 = np.zeros(np.shape(cutout))
        trace0[:,slit_gap:] = cutout[:,:-slit_gap]
        #trace1 = cutout[slit_gap:-slit_gap]
        trace1 = cutout
        #trace2 = shift(cutout, [slit_gap, dx2], order = 4)
        trace2 = np.zeros(np.shape(cutout))
        trace2[:,:-slit_gap] = cutout[:,slit_gap:]
        
        if plot:
            plt.subplot(131)
            plt.imshow(trace0, origin = 'lower')
            plt.subplot(132)
            plt.imshow(trace1, origin = 'lower')
            plt.subplot(133)
            plt.imshow(trace2, origin = 'lower')
            plt.show()
        #package slit number and traces
        all_slit = [0,1,2]
        all_trace = [trace0, trace1, trace2]
        
        #sky slit contains no object, or contaminant. 
        sky_slit = all_slit
        sky_slit.remove(obj_slit) #sky slits
        if masked_slit != None:
            sky_slit.remove(masked_slit)
        
        ####background subtraction
        if len(sky_slit) == 1: #one open slit is contaminated
            sky = all_trace[sky_slit[0]]
        elif len(sky_slit) ==2: # no open slit contaminated
            sky = (all_trace[sky_slit[0]] + all_trace[sky_slit[1]])/2 #Average
        
        sky_sub = all_trace[obj_slit] - sky
        if plot:
            plt.subplot(131)
            plt.imshow(all_trace[obj_slit], origin = 'lower')
            plt.subplot(132)
            plt.imshow( sky, origin = 'lower')
            
            
            plt.subplot(133)
            plt.imshow( sky_sub, origin = 'lower')
            plt.show()
        #sky_sub = np.pad(sky_sub, ( (0, np.shape(cutout)[0]-np.shape(sky_sub)[0]) \
        #                            ,(0,np.shape(sky_sub[1]-np.shape(cutout)[1]) ) )\
        #                            ,mode = 'constant')
        #sky = np.pad(sky_sub, ( (0, np.shape(cutout)[0]-np.shape(sky_sub)[0]) \
        #                            ,(0,np.shape(sky_sub[1]-np.shape(cutout)[1]) ) )\
        #                            ,mode = 'constant')
        return sky_sub, sky

def fit_background_2d_polynomial(cutout, mask, polynomial_order = 2):
    """
    Takes a given 2d cutout of trace with the actual spectral trace masked out. 
    This mask is from makeDiagMask, or in the same format: 1 on trace and 0 off trace
    Then fits a 2d polynomial to estimate the sky background.
    """
    #first get x, y coordinates
    y, x = np.mgrid[:cutout.shape[0],:cutout.shape[1]]
    #call astropy fitter
    poly = models.Polynomial2D(polynomial_order)
    fitter = fitting.LinearLSQFitter()

    res = fitter(poly, y[~mask], x[~mask], cutout[~mask]) 

    sky = res(y,x)

    return cutout-sky, sky

def sub_bkg_shift_and_mask(source, masks):
    '''
    Cross correlate the thumbnails to a masks, then mask the trace to estimate the backgroud and subtract. 
    '''
    
    for i in range(4):
        trace = source.trace_images[i]
        
        mask = masks[i]
        shifted = chi2_shift(trace,mask, zeromean=True, verbose=False, return_error=True)
        
        new_image = shift(rld, (shifted[1],shifted[0]))
        
        bkg_med = np.median(new_image[~mask])
        
        souce.trace_images[i] = new_image - bkg_med

def fitFlux(flux_vec, seeing_pix = 4):
    """
    This function fits the flux value along a 1d cross section using a
    sum of polynomial and gaussian. This function is called by fitBkg
    
    Inputs:
    flux_vec: a vector (1d array) containing flux values
    seeing_pix: a guess of seeing size in pixel. Default = 4 (1").
    
    Output:
    res: result of the fit in astropy format. res[0] is the
    polynomial part. res[1] is the Gaussian part. 
    """
    x = range(len(flux_vec))
    ###guess amplitude
    source_amp = np.max(flux_vec)- np.min(flux_vec)
    #Defing fitting functions and fitter using Astropy fitting routine
    poly = models.Polynomial1D(2)
    gauss = models.Gaussian1D(amplitude = source_amp, mean = np.argmax(flux_vec) ,stddev = seeing_pix/2, \
                                    bounds = {'amplitude': [0, np.inf], 'stddev':[0.5,3] })
    fitter = fitting.LevMarLSQFitter()
    #Fit the data, assuming both polynomail and gaussian components
    res = fitter(poly+gauss, x, flux_vec)
    
    #if gaussian peak is smaller than 1sigma of data, use only polynomial
    if res[1].amplitude.value < 3*np.sqrt(np.var(flux_vec)):
        #print('no gaussian')
        res = fitter(poly, x, flux_vec)+ models.Gaussian1D(amplitude = 0)
    #print(res)
    return res

def fit_and_subtract_background(cutout, trace_length = 60, seeing_pix = 4, plotted = False):
    """
    This function fits the given cutout using a sum of polynomial and gaussian
    functions. It iterates the image from bottom right to top left, makes a
    vertical cross section, and calls fitFlux to do actual fitting. 
    
    Inputs:
    cutout: an array representing a cutout containing a trace
    trace_length: the length of the trace in pixel unit. This is depending on the band.
    seeing_pix: a guess of the seeing size in pixel unit. Default = 4 (1").
    
    Outputs:
    all_res: the list of results from all 1D fits across the trace
    background: an array with the same size of cutout, containing background value
    
    """
    #Define the coordinate such that x starts at the bottom right (small wavelength)
    #and increases toward the upper left. This is the dispersion direction
    
    #y is from bottom left to upper right. This is the cross dispersion direction. 
    #plt.imshow(cutout, origin = 'lower')
    #plt.show()
    width = len(cutout[0]) #we have square cutout
    #buffer = round((width - trace_length -5)/2) #buffer area on either ends of the trace
    buffer = int(round(0.85*slit_length/2)) #imported from constant
    
    #x = range(width-2*buffer)
    #y = range(buffer)

    #Stashed
    x = range(int(width-2*buffer))
    y = range(int(buffer))
    #blank background
    bkg = np.zeros(np.shape(cutout))
    
    all_res_even = []
    all_res_odd  = []
    
    flux = []
    var = []
    #fitBkgs = []
    for i in x: #Iterate from right to left of the thumbnail

        #cross_section_even = cutout[i:i+2*buffer, width-buffer-i]
        #print(len(cross_section_even))
        cross_section_even = np.zeros(buffer)
        cross_section_odd = np.zeros(buffer)
        for j in y: #diagonal iteration
            cross_section_even[j] =  cutout[i+j , width-buffer-i+j]
            cross_section_odd[j]  =  cutout[i+j+1 , width-buffer-i+j]
        
        
        #Compute the flux fit 
        res_even = fitFlux(cross_section_even)
        res_odd  = fitFlux(cross_section_odd)
        yy = np.arange( len(cross_section_even))
        #print(yy)
        fitBkg_even = res_even[0](yy) #res[0] is the polynomial component
        fitBkg_odd = res_odd[0](yy)
        
        for k in y:
        #Generate a background frame
            bkg[i+k   , width-buffer-i+k] = fitBkg_even[k]
            bkg[i+k+1 , width-buffer-i+k]           = fitBkg_odd[k]
        #print(res[xx])
            
        #Plot cross-section
        if plotted:
            plt.plot(yy,cross_section_even)
            plt.plot(yy,res_even(yy), 'r')
            plt.plot(yy, res_even[0](yy), 'b')
            plt.plot(yy, fitBkg_even, 'k')
            plt.show()
        #print(res_even)
        all_res_even = all_res_even + [ res_even ]
        all_res_odd = all_res_odd + [ res_odd ]
        #fitBkgs += [fitBkg]
        flux_even = np.sum(res_even[1](yy))
        flux_odd = np.sum(res_odd[1](yy))
        flux += [flux_even, flux_odd] #sum the gaussian componenta and add as flux
        var += [ np.sum( res_even(yy) * res_even[1](yy)/flux_even ), \
                 np.sum( res_odd(yy) * res_even[1](yy)/flux_odd )]
        
    #just for plotting
    #    flux = []
    #    for i in all_res:
    #        flux += [i(np.array(y))]
    #    flux = np.array(flux)
    #    print('size = ',np.shape(flux))
    #    plt.contour(y, x, flux)
            
    #    #Create a blank background array
    #    fitBkgs = np.array(fitBkgs)
    #    print(np.shape(fitBkgs))
    #    plt.imshow(fitBkgs);plt.colorbar()
    #    plt.show()
    #    background = np.zeros(np.shape(cutout))
    #    print('bkg size, fitBkg', np.shape(background), np.shape(fitBkgs))
    #    for i in x:
    #        for j in y:
    #            #print(i+j,  width - buffer - i +j)
    #            background[i+j, width - buffer - i +j] = fitBkgs[i,j]
        
        #return np.array(all_res_even), np.array(all_res_odd), bkg
        #print(all_res_even)
        #print(np.array(all_res_even))
        
        #calculate angle
        # angle = angCalc(all_res_even)
        #return all_res_even, bkg, flux, var
    return cutout - bkg, bkg

# @profile
def findTrace(thumbnail, poly_order = 1, weighted = False, plot = False, diag_mask=False,mode='pol',fractional_fit_type = None):

    """
    mode='pol' or 'spec'
    
    findTrace iterates through the diagonal of the image, finds the maximum, fits a
    polynomial of order poly_order to it, then returns the y value of the trace for 
    each pixel on the x axis.

    HK: seems like it iterates along rows and not diagnals?

    At the location of maximum flux, it calls traceWidth to get the stddev of 
    the gaussian fit to the trace at that location.
    """

    peaks = []
    peak_size = []
    
    #Estimate the background noise
    bkg = []
    bkg_length = 10
    
    # thumbnail = median_filter(thumbnail, 6)
    thumbnail = cv2.medianBlur(np.ndarray.astype(thumbnail,'f'),5)    


    if diag_mask and mode=='pol':
        mask = makeDiagMask(np.shape(thumbnail)[0],100)
        thumbnail[~mask] = 0.0
        # plt.imshow(thumbnail)
    
    for i in range(np.shape(thumbnail)[1]): #iterate through x
        peaks +=[ np.argmax(thumbnail[:,i]) ] 
        peak_size += [np.max(thumbnail[:,i])]
        bkg += [np.std(np.concatenate((thumbnail[:bkg_length,i],thumbnail[-bkg_length:,i])))]
      
    bkg = np.array(bkg)
    # print np.shape(bkg)
    # print np.shape(peak_size)

    #polynomial fit
    if weighted:
        xinds = np.arange(np.shape(thumbnail)[1])
        # xcen = np.shape(thumbnail)[1]/2.
        
        #Weight the peaks by the strength of the y-peak. 
        weights = np.array(peak_size)**2

        #If the peaks are less than 75% of the brightest peak, set their weight to zero. 
        # weights[weights < 0.75* np.max(weights)] = 0. 
        # weights[weights < 5*bkg] = 0.

        #Further scale the weights by their distance from the center of the image: this is hard coded!
        # weights *= 1/(np.abs(xinds-xcen))
        width = thumbnail.shape[1] #x size of thumbnail
        if mode=='pol':
            weights[(xinds < width/2 - 15) | (xinds > width/2+15)] = 0.
        if mode=='spec':
            #we shoot outward from the column with the maximal peak
            #the second we drop below 50% of that value, set a cutoff
            #then, only fit the trace between the cutoffs
            max_peak = np.max(peak_size)
            max_ind = np.argmax(peak_size)
            upper_x = width
            lower_x = 0
            for i in range(max_ind, width):
                if peak_size[i] < 0.5 * max_peak:
                    upper_x = i
                    break
            for i in range(max_ind, 0, -1):
                if peak_size[i] < 0.5 * max_peak:
                    lower_x = i
                    break
            weights[:lower_x] = 0
            weights[upper_x:] = 0
            #If the peaks are less than 10% of the brightest peak, set their weight to zero. 
            #weights[weights < 0.5* np.max(weights)] = 0.
            peaks_spline = copy.deepcopy(peaks)
            #getting the spline interpolated positions
            #fit each column from lower_x to upper_x with a spline and use the spline maxima for fitting peak_size
            for i in range(lower_x, upper_x):
                x = np.arange(0, len(thumbnail[:,i]), 1)
                y = thumbnail[:,i]
                #throwing out the negative values from shift and subtract to make interpolation better
                inds = np.where(y > 0.1 * np.max(y))
                x = x[inds]
                y = y[inds]
                xnew = np.arange(np.min(x), np.max(x), 0.1)
                ynew = np.zeros(len(xnew))
                if fractional_fit_type == 'spline':
                    f = interp1d(x, y, kind = 'cubic')
                    ynew = f(xnew)
                elif fractional_fit_type == 'cubic':
                    p = np.polyfit(x, y, 3)
                    ynew = np.polyval(p, xnew)
                elif fractional_fit_type =='gaussian':
                    def gauss(x, A, mu, sigma):
                        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
                    coeff, cov = curve_fit(gauss, x, y, p0=[np.max(y),i,5], maxfev = 15000)
                    ynew = gauss(xnew, *coeff)
                peaks_spline[i] = xnew[np.argmax(ynew)]
        if fractional_fit_type is not None:
            p = np.polyfit(range(np.shape(thumbnail)[1]), peaks_spline, poly_order, w = weights)
        else:
            p = np.polyfit(range(np.shape(thumbnail)[1]), peaks, poly_order, w = weights)
    else:
        weights = np.ones(len(peaks))
        p = np.polyfit(range(np.shape(thumbnail)[1]), peaks, poly_order)

    fit = np.polyval(p,range(np.shape(thumbnail)[1]))

    #now the angle
    #second to last element of p is the linear order.
   # print(p[-2])
    angle = np.degrees(np.arctan(p[-2]))

    #Now for the trace width, mask irrelevent area to prevent traceWidth trying to fit weird places in the image
    on_trace = np.abs(fit-peaks) < 5 #5 pixels
    x_bigpeak = np.nanargmax(peak_size*on_trace) #set "peaks" that are not on trace to zero
    y_bigpeak = peaks[x_bigpeak]
    width = traceWidth(thumbnail, (y_bigpeak, x_bigpeak), bkg_length)



    if plot:
        # to_plot = np.where(weights == 0, 0, 1)
        #print('Plotting')
        #plt.plot(peaks_spline*to_plot)
        plt.imshow(thumbnail, origin = 'lower')
        #plt.plot(to_plot)
        plt.plot(fit)
        plt.plot(peaks)
        try:
            plt.title('Width = %.2f, angle = %.2f'%(width, angle))
        except:
            plt.title('Error in width or angle')
        plt.show()


    return peaks, fit, width, angle

def trace_location_along_x(thumbnail, angle, template_width = 70, plot = 0):
    """
    find the location of the trace along the x axis. This is to automate
    the broadband aperture photometry in spec_utils
    Inputs: thumbnail: a 2D array of the image
    Output: x_loc: the x locaiton of the center of the trace
    """

    sum_im = np.sum(thumbnail, axis = 0) #sum the image along the y axis

    #create a template
    length= len(sum_im)
    width = template_width #* np.abs(np.cos(np.radians(angles)))
    template = np.zeros(length) 
    template[ int(length/2 - width/2) : int(length/2 + width/2)] = 1 #1's in the center
    # print(np.sum(template))

    #cross correlation
    corr = scipy.signal.fftconvolve(sum_im, template)


    if plot:
        fig,ax = plt.subplots(1,2)
        ax[0].plot(shift(sum_im/np.max(sum_im), -(np.nanargmax(corr) - int(length) +1)))
        ax[0].plot(template)
        ax[1].plot(np.arange(-length+1, length), corr)
	#ax[1].plot(corr)

        plt.show()

    return np.nanargmax(corr) - int(length/2) +1 #the x center of the trace



def fitFlux(flux_vec, seeing_pix = 4):
    """
    This function fits the flux value along a 1d cross section using a
    sum of polynomial and gaussian. This function is called by fitBkg
    
    Inputs:
    flux_vec: a vector (1d array) containing flux values
    seeing_pix: a guess of seeing size in pixel. Default = 4 (1").
    
    Output:
    res: result of the fit in astropy format. res[0] is the
    polynomial part. res[1] is the Gaussian part. 
    """
    x = range(len(flux_vec))
    ###guess amplitude
    source_amp = np.max(flux_vec)- np.min(flux_vec)
    #Defing fitting functions and fitter using Astropy fitting routine
    poly = models.Polynomial1D(2)
    gauss = models.Gaussian1D(amplitude = source_amp, mean = np.argmax(flux_vec) ,stddev = seeing_pix/2, \
                                    bounds = {'amplitude': [0, np.inf], 'stddev':[0.5,3] })
    fitter = fitting.LevMarLSQFitter()
    #Fit the data, assuming both polynomail and gaussian components
    res = fitter(poly+gauss, x, flux_vec)
    
    #if gaussian peak is smaller than 1sigma of data, use only polynomial
    if res[1].amplitude.value < 3*np.sqrt(np.var(flux_vec)):
        #print('no gaussian')
        res = fitter(poly, x, flux_vec)+ models.Gaussian1D(amplitude = 0)
    #print(res)
    return res

def sub_bkg_shift_and_mask(source, plot=False):
    '''
    Cross correlate the thumbnails to the masks, then mask 
    '''
    if plot:
        fig = plt.figure(figsize=(7,7))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
    
    for i in range(4):
        trace = source.trace_images[i]
        
        xlow  = 30
        xhigh = 130
        ylow  = 30
        yhigh = 130
        mask = np.ndarray.astype(trace_masks[i],bool)
        
        shifted = chi2_shift(trace,mask, zeromean=True, verbose=False, return_error=True)

        # print(shifted)
        
        new_image = shift(trace, (shifted[1],shifted[0]))[ylow:yhigh,xlow:xhigh]
        
        n_mask = mask[ylow:yhigh,xlow:xhigh]
        bkg_med = np.median((new_image)[~n_mask])
        # bkg_med = np.median((new_image[ylow:yhigh,xlow:xhigh])[~mask])
        
        source.trace_images[i] = trace - bkg_med

        if plot:
            ax1 = fig.add_subplot(2,2,i+1)
            ax1.imshow(new_image)
            ax1.imshow(~n_mask, alpha=0.3)
        
    return source

def mask_and_sub_bkg(thumbnail, index, plot=False, xlow=30,xhigh=130,ylow=30,yhigh=130):
    '''
    Cross correlate the thumbnails to a mask then measure the background. 
    '''
    
    #Grab the appropriate mask
    mask = np.ndarray.astype(trace_masks[index],bool)
    
    #The the shift between the trace and the mask
    shifted = chi2_shift(thumbnail,mask, zeromean=True, verbose=False, return_error=True)

    # print(shifted)
    
    #Shift the thumbnail
    new_image = shift(thumbnail, (shifted[1],shifted[0]))[ylow:yhigh,xlow:xhigh]
    
    #Measure the background .
    n_mask = mask[ylow:yhigh,xlow:xhigh]
    bkg_med = np.median((new_image)[~n_mask])

    if plot:
        fig = plt.figure(figsize=(7,7))
        ax1 = fig.add_subplot(111)
        ax1.imshow(new_image)
        ax1.imshow(~n_mask, alpha=0.3)
        
    #Return and image of the same size as thumbnail, only containing the measured background level
    return thumbnail*0.+bkg_med

def mask_and_2d_fit_bkg(thumbnail, index, polynomial_order = 2, plot=False, xlow=30,xhigh=130,ylow=30,yhigh=130):
    '''
    Cross correlate the thumbnails to a mask then measure the background. 
    '''
    
    #Grab the appropriate mask
    mask = np.ndarray.astype(trace_masks[index],bool)
    
    #The the shift between the trace and the mask
    shifted = chi2_shift(thumbnail,mask, zeromean=True, verbose=False, return_error=True)

    # print(shifted)
    
    #Shift the thumbnail
    new_image = shift(thumbnail, (shifted[1],shifted[0]))[ylow:yhigh,xlow:xhigh]
    
    #Measure the background .
    n_mask = mask[ylow:yhigh,xlow:xhigh]

    #2d polynomial fit 
    poly_fit = models.Polynomial2D(degree = polynomial_order)
    fitter = fitting.LevMarLSQFitter()
    #define x,y
    y, x = np.mgrid[:thumbnail.shape[0], :thumbnail.shape[1]]

    #masked out
    masked_y = y[~n_mask]
    masked_x = x[~n_mask]
    masked_im = new_image[~n_mask]

    #fit
    res = fitter(poly_fit, masked_y,masked_x, masked_im)

    #reconstructed bkg
    bkg_est = res(y,x)


    #bkg_med = np.median((new_image)[~n_mask])

    if plot:
        fig = plt.figure(figsize=(7,7))
        ax1 = fig.add_subplot(111)
        ax1.imshow(new_image)
        ax1.imshow(~n_mask, alpha=0.3)
        
    #Return and image of the same size as thumbnail, only containing the measured background level
    return thumbnail*0.+bkg_est

def traceWidth(trace, location, fit_length):
    """
    traceWidth fits a Gaussian across the trace (in the spatial direction) at the given location 
    to find the width of the trace.

    Input:      trace       -- a 2D array containing an image of the trace
                location    -- a tuple (y,x) of the extracting location. 
                fit_length      -- an integer, for which we fit from y-length to y+length and similarly in x
    Output:     standard deviation of the resulting Gaussian fit. 
    """
    #First check if the given length will fall out of the trace
    y_trace, x_trace = trace.shape
    if (location[0] - fit_length < 0) or (location[0] + fit_length > y_trace):
        print('Given location and fit_length fall of the trace image.')
        return None
    elif (location[1] - fit_length < 0) or (location[1] + fit_length > x_trace):
        print('Given location and fit_length fall of the trace image.')
        return None
    else:
        #create a flux vector
        ypos = []
        xpos = []
        flux = np.zeros(2*fit_length)
        for i in range(2*fit_length):
            ypos.append(location[0] - fit_length + i)
            xpos.append(location[1] - fit_length + i)
            flux[i] = trace[location[0] - fit_length + i , location[1] - fit_length + i ]
    #    plt.show()
    #    plt.imshow(trace, origin = 'lower')
    #    plt.plot(xpos, ypos, 'r-')
    #    plt.show()
        #fit parameters
        x = range(len(flux))
        gauss = models.Gaussian1D(mean = np.argmax(flux), stddev = 4, amplitude = np.max(flux))#, bounds = {'stddev':[-5,5]}) 
        poly = models.Polynomial1D(2)  
        f = fitting.LevMarLSQFitter()

        # res = f(gauss+poly, x, flux)
        res = f(gauss, x, flux)

        # return res[0].stddev.value
        return res.stddev.value


def traceWidth_after_rotation(trace, fitlength = 10):
    collapsed = np.sum(trace, axis = 1)              #collapsing trace along x axis
    x = range(len(collapsed))
    
    gauss = models.Gaussian1D(mean = np.argmax(collapsed), stddev = 3, amplitude = np.max(collapsed))
    f = fitting.LevMarLSQFitter()
    res = f(gauss, x, collapsed)                     #fitting collapsed trace to a gaussian
    return res.stddev.value                          #returning standard deviation
       
def clean_thumbnails_for_cosmicrays(thumbnails, method='lacosmic',thumbnails_dq=None, nsig=3):
    '''
    Tries to identify cosmic rays, by looking for pixels 5-sigma about the background, after masking out the trace. 
    It then adds them to the DQ frame
    Note, this doesn't do anything for cosmic rays very close to the trace. 

    Inputs: 
        thumbnails - a [4,n,n]  array that has 4 thumbnails, each of dimension n x n pixels. 
        thumbnails_dq  - a [4,n,n] shaped array that has 4 thumbnails representing the dataquality frame of each thumbnail
        nsig - pixels above this many sigma away from the median will be rejected. 
    '''
    if method == 'old':
    
        bp_masks = []
        for i in range(4):
            mask = make_source_mask(thumbnails[i,:,:],snr=nsig,npixels=5,dilate_size=5)
            mean,median,std = sigma_clipped_stats(thumbnails[i,:,:],sigma=3.0,mask=mask)

            bpmask = (np.abs(thumbnails[i,:,:]-median) > nsig*std) & ~(mask)

            for bpx,bpy in [(np.where(bpmask)[0],np.where(bpmask)[1])]:
                thumbnails[0,bpx,bpy] = np.nanmedian(thumbnails[0,bpx[0]-2:bpx[0]+2,bpy[1]-2:bpy[1]+2])

            # if thumbnails_dq is not None:
                # bp_mask = thumbnails_dq[i] | bpmask
            thumbnails_dq[i][np.where(bpmask)] = 4

            # bp_masks.append(bpmask)

        return thumbnails, thumbnails_dq

    if method == 'lascosmic':
        import ccdproc
        # A downside to this method is that it doesn't update the DQ frame. 
        
        for i in range(4):
            thumbnails[i,:,:] = ccdproc.cosmicray_lacosmic(thumbnails[i,:,:], sigclip=nsig)[0]
        return thumbnails, thumbnails_dq


def smooth_cutouts(thumbnails,method='gaussian',width=3):
    '''
    A function to smooth the thumbnails
    '''
    if method != "gaussian" and method != "median":
        print('Only "gaussian" thumbnail smoothing is implemented')
        return

    if method == "gaussian":
        filter_type = gauss
    elif method == 'median':
        filter_type = median_filter
    else:
        print('Only "gaussian" and "median" thumbnail smoothing are implemented')
        print('Returning')
        return        

    if len(thumbnails.shape) == 3:
        for i in range(thumbnails.shape[0]):
            thumbnails[i] = filter_type(thumbnails[i],width)
    elif len(thumbnails.shape) == 2:
        thumbnails = filter_type(thumbnails,width)
    else:
        print("Your thumbnails shape is weird and needs to be checked. It should either me [n_images,x,y] or just x,y")

    return



