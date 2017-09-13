# -*- coding: utf-8 -*-
"""
Created on Fri June 2 2017

@author: Kaew Tinyanont, Max Millar-Blanchaer

Spectral Extraction Utilities for the WIRC-POL DRP

This file contains functions used to extract spectra from a cutout. 

"""
import sys, os
import numpy as np
from matplotlib import pyplot as plt
from wirc_drp.constants import *

from scipy.optimize import basinhopping
from scipy.ndimage import shift, median_filter
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

from skimage.measure import profile_line

from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel, Box1DKernel, convolve
from astropy.io import fits as f
from astropy import stats

#From other packages
from wirc_drp.utils.image_utils import locationInIm, shift_and_subtract_background, fit_and_subtract_background, findTrace
from wirc_drp.masks.wircpol_masks import *

#Import for vip functions
import warnings
try:
    import cv2
    no_opencv = False
except ImportError:
    msg = "Opencv python binding are missing (consult VIP documentation for "
    msg += "Opencv installation instructions). Scikit-image will be used instead."
    warnings.warn(msg, ImportWarning)
    no_opencv = True

from skimage.transform import rotate


#debugging
import pdb

#path
import sys, os, time

import copy
#import pyklip.klip

wircpol_dir = os.environ['WIRC_DRP']


def frame_rotate(array, angle, imlib='opencv', interpolation='bicubic', cxy=None):
    """ Rotates a frame.
    
    Parameters
    ----------
    array : array_like 
        Input frame, 2d array.
    angle : float
        Rotation angle.
    imlib : {'opencv', 'skimage'}, str optional
        Library used for image transformations. Opencv is faster than ndimage or
        skimage.
    interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
        'nneighbor' stands for nearest-neighbor interpolation,
        'bilinear' stands for bilinear interpolation,
        'bicubic' for interpolation over 4x4 pixel neighborhood.
        The 'bicubic' is the default. The 'nearneig' is the fastest method and
        the 'bicubic' the slowest of the three. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
    cxy : float, optional
        Coordinates X,Y  of the point with respect to which the rotation will be 
        performed. By default the rotation is done with respect to the center 
        of the frame; central pixel if frame has odd size.
        
    Returns
    -------
    array_out : array_like
        Resulting frame.
        
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    array = np.float32(array)
    y, x = array.shape
    
    if not cxy:  
        cy, cx = frame_center(array)
    else:
        cx, cy = cxy

    if imlib not in ['skimage', 'opencv']:
        raise ValueError('Imlib not recognized, try opencv or ndimage')

    if imlib=='skimage' or no_opencv:
        if interpolation == 'bilinear':
            order = 1
        elif interpolation == 'bicubic':
            order = 3
        elif interpolation == 'nearneig':
            order = 0
        else:
            raise TypeError('Interpolation method not recognized.')

        min_val = np.min(array)
        im_temp = array - min_val
        max_val = np.max(im_temp)
        im_temp /= max_val

        array_out = rotate(im_temp, angle, order=order, center=cxy, cval=np.nan)

        array_out *= max_val
        array_out += min_val
        array_out = np.nan_to_num(array_out)

    else:
        if interpolation == 'bilinear':
            intp = cv2.INTER_LINEAR
        elif interpolation == 'bicubic':
            intp= cv2.INTER_CUBIC
        elif interpolation == 'nearneig':
            intp = cv2.INTER_NEAREST
        else:
            raise TypeError('Interpolation method not recognized.')

        M = cv2.getRotationMatrix2D((cx,cy), angle, 1)
        array_out = cv2.warpAffine(array.astype(np.float32), M, (x, y), flags=intp)
             
    return array_out

def fitAcrossTrace_aligned(cutout, stddev_seeing = 4, box_size = 1, plot =  False, return_residual = False, \
                            trace_angle = -45, fitfunction = 'Moffat', sum_method = 'model_sum', poly_order = 4):
    """This function iterates the cutout from bottom right to top left, makes
        a diagonal cut (perpendicular to the trace) and fits gaussian along that cut.
        
        Input: cutout: an array representing the image. This should be background
                        subtracted.




               note: assume the cutout is subtracted by global background model. Will be derotated
                     with the spectral trace aligned with x axis grid and close to the center in y
                     ------------
                     |          |
                     |          |
                     |**********|
                     |          |
                     |          |
                     ------------

        Output: a 1D array containing the background-subtracted spectrum.    

    ###TO DO: add trace back in and compute angle             
    
    """
    lowcut = 55
    highcut = 100

    width = len(cutout[0]) #we have square cutout
    #print(width//box_size)
    #print(box_size)
    x = range(width//box_size) #number of bins here
    #print(x)
    #y = range(width)
    
    #rotate the image
    cutout_rot = frame_rotate(cutout, trace_angle, cxy=[width/2,width/2])

    #plt.imshow(cutout_rot, origin = 'lower')
    #plt.show()

    #vector to contain results
    results = [] #gaussian fits
    poly_results = [] #background polynomial fit
    if return_residual: #compute the background image from the fit. This is from the polynomial fit.
        residual = np.zeros(cutout_rot.shape)
    
    for i in x:
        #print(box_size*i, (box_size)*(i+1))
        if box_size > 1:
            cross_section=cutout_rot[lowcut:highcut,box_size*i:(box_size)*(i+1)] #hard coded! change this 
        else:
            cross_section=cutout_rot[lowcut:highcut,i]
        #print(cross_section.shape)
        #if box_size > 1, median combine cross_section
        if box_size > 1:
            cross_section = np.median(cross_section, axis = 1) 
        #print(cross_section.shape)
        #fit models

        y = range(len(cross_section))
        poly = models.Polynomial1D(poly_order)

        smooth_cross = median_filter(cross_section, 5) #use this to find actual peak, rejecting bad pixels
        #bad_pix = stats.sigma_clip((cross_section - smooth_cross)/smooth_cross, sigma = 5) 
        if plot:
            plt.subplot(122)
            plt.plot(cross_section - smooth_cross)
            plt.plot(bad_pix.mask*100)

        #cross_section = cross_section*~bad_pix.mask + smooth_cross*bad_pix.mask #fill bad pixels

        if fitfunction == 'Moffat':
            psf_moffat1d = models.Moffat1D(x_0 = np.argmax(smooth_cross), gamma = stddev_seeing, alpha = 1,  amplitude = np.max(smooth_cross))
            model = psf_moffat1d + poly
        elif fitfunction == 'Gaussian':
            psf_gauss1d = models.Gaussian1D(mean = np.argmax(smooth_cross), stddev = stddev_seeing, amplitude = np.max(smooth_cross))  
            model = psf_gauss1d + poly

        f = fitting.LevMarLSQFitter()
        #f = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(), stats.sigma_clip)
        all_res = f(model, y, cross_section)
        res = all_res[0]
        poly_res = all_res[1]
        
        #plotting
        if plot:
            plt.subplot(131)
            plt.plot(y,res(y) + poly_res(y),'--r')
            #plt.text(10,max(cross_section)/2, str(np.argmax(cross_section)),color ='r')
            plt.plot(y, cross_section,'b')
            plt.subplot(132)
            plt.plot(y, cross_section- (res(y) + poly_res(y) ))
            plt.subplot(133)
            plt.plot(y, cross_section - poly_res(y))
            plt.plot(y, res(y))
            plt.show()

        if return_residual:
            if box_size > 1:
                foo =res(y) +poly_res(y)
                residual[lowcut:highcut,box_size*i:(box_size)*(i+1)] = cutout_rot[lowcut:highcut,box_size*i:(box_size)*(i+1)] - np.tile(foo, (box_size,1)).T
            else:
                residual[lowcut:highcut,i] = cutout_rot[lowcut:highcut,i] - res(y) - poly_res(y)
        
        #flux += np.sum(res(y))     
        #flux += [res.amplitude.value * res.stddev.value *np.sqrt(2*np.pi) ]
        results += [res] #just a list of results
        poly_results += [poly_res]
    if fitfunction == 'Gaussian':
        stddev_vector  = [i.stddev.value for i in results] #all stddev, only use results with stddev within 3 sigmas from median
        loc_vector = [i.mean.value for i in results]
    elif fitfunction == 'Moffat':
        stddev_vector  = [i.gamma.value for i in results] #all stddev, only use results with stddev within 3 sigmas from median
        loc_vector = [i.x_0.value for i in results]       
    #valid = np.logical_and(np.abs(stddev_vector - np.median(stddev_vector)) < 1*np.sqrt(np.var(stddev_vector)) , np.abs(loc_vector - np.median(loc_vector)) < 1*np.sqrt(np.var(loc_vector)))
    valid = np.abs(loc_vector - np.median(loc_vector)) < 1*np.sqrt(np.var(loc_vector))

    if fitfunction == 'Gaussian':
        if sum_method =='model_sum':
            flux = [i.stddev.value*i.amplitude.value*np.sqrt(2*np.pi) for i in results]*valid
        elif sum_method == 'weighted_sum':
            for i in x: 
                results[i].amplitude=1.
            cutout_rot_subbkg = [np.mean(cutout_rot[lowcut:highcut,box_size*i:(box_size)*(i+1)],axis=1) -  poly_results[i](y) for i in x]#Subtract the background
            flux = [np.sum( cutout_rot_subbkg * results[i](y))/np.sum(results[i](y)**2) for i in x]*valid #sum(data*model)/sum(model)

        else:
            raise ValueError(sum_method+' is invalid. Choose from model_sum or weighted_sum')
    
    elif fitfunction == 'Moffat':
        if sum_method == 'model_sum':
            flux = [np.sum(i(y)) for i in results]*valid
        elif sum_method == 'weighted_sum':
            for i in x: 
                results[i].amplitude=1./np.sum(results[i](y))
            
            cutout_rot_subbkg = [np.mean(cutout_rot[lowcut:highcut,box_size*i:(box_size)*(i+1)],axis=1) -  poly_results[i](y) for i in x]#Subtract the background
            flux = [np.sum( cutout_rot_subbkg * results[i](y))/np.sum(results[i](y)**2) for i in x]*valid #sum(data*model)/sum(model)
            # flux = [np.sum( np.median(cutout_rot_subbkg[lowcut:highcut,box_size*i:(box_size)*(i+1)], axis = 1) * results[i](y))/np.sum(results[i](y)) for i in x]*valid #sum(data*model)/sum(model)
        else:
            raise ValueError(sum_method+' is invalid. Choose from model_sum or weighted_sum')
    else:
        raise ValueError(fitfunction+' is invalid. Choose from Moffat or Gaussian.')

    flux[~valid] = 0.
    flux[flux != flux] = 0.
    #plt.plot(valid, 'c')
    #plt.plot(stddev_vector, 'r')
    #plt.plot(loc_vector, 'k')
    #plt.plot(flux/np.max(flux),'b')
    #plt.show()  

    #if box_size > 1:
    #    flux = median_filter(flux, box_size)
    
    if return_residual:
        return np.array(flux[::-1]), np.array(flux[::-1])**2, residual #flip flux so the array index increases as a function of wavelength
    else:
        return np.array(flux[::-1]), np.array(flux[::-1])**2 #fake variance for now. 


def weighted_sum_extraction(cutout, trace, psf, ron = 12, gain = 1.2):
    """
    weightedSum takes the original, background cutouts, and fit_result from
    fitBkg as inputs. It iterates the cutout the same way fit_result did. 
    
    (i) For each column of pixels (x), it goes to the y location given by trace
    
    (ii) It sums pixel values around that location weighted by the given PSF. 

    Inputs:     cutout  -- a 2D array from which the spectrum is extracted
                trace   -- a 1D array with length equal to cutout.shape[1]
                            it represents y location of the peak at each x location
                psf     -- a square 2D array of odd width representing a normalized psf 
    Output:     spec    -- a 1D array containing extracted flux
                var     -- a 1D array containing estimated flux uncertainties provided read noise

    """
    ###NEW VERSION BELOW
    # width = len(cutout[0]) #we have square cutout
    # #buffer area on either ends of the trace
    # buffer = int(round(0.85*slit_length/2)) #imported from constant

    #width = len(cutout[0])
    spec = []
    var = []
    for i in range(len(trace)): #loop through x
        #print(i)
        #put psf at this location
        dim = np.array(cutout.shape) + np.array(psf.shape)
        #print(dim)
        weight = np.zeros( dim) #padded, to be cropped later

        #case where trace i is not in the image
        if trace[i] < 0 or trace[i] > cutout.shape[0]:
            spec += [0]
            var += [0]
        else:
            x = i + int(psf.shape[1]//2)
            #print(trace[i], psf.shape[0]//2)
            y = int(trace[i] + psf.shape[0]//2)
            #print(i, x, y - int(psf.shape[0]//2), y + int(psf.shape[0]//2)+1, np.shape(weight[y - int(psf.shape[0]//2): y + int(psf.shape[0]//2)+1, x - int(psf.shape[1]//2): x + int(psf.shape[1]//2)+1]))
            weight[y - int(psf.shape[0]//2): y + int(psf.shape[0]//2)+1, x - int(psf.shape[1]//2): x + int(psf.shape[1]//2)+1] = psf
            weight = weight[ int(psf.shape[0]//2): int(-psf.shape[0]//2), int(psf.shape[1]//2): int(-psf.shape[1]//2)]
            #print(weight.shape, cutout.shape)
            #plt.imshow(weight*cutout,origin = 'lower')
            #plt.show()

            spec += [np.sum(weight * cutout)/np.sum(weight)]
            #TODO: Is the variance calculation correct? Might need another weighted term. 
            var += [np.sum(weight * (cutout/gain + (ron/gain)**2))] #variance assuming readout noise and photon noise

    return np.array(spec[::-1]), np.array(var[::-1]) #flip so long wavelenght is to the right

def sum_across_trace(data, variance, extraction_range):
    """
    extract spectrum by simply summing in the spatial direction
    This also serves as a helper function for optimal_extraction
    Input:
        data: 2D numpy array of the data, background subtracted
        variance: 2D numpy array of the variance of the data
        extraction_range: a list of [y_begin, y_end], assuming that y is the spatial direction. 
                (TO BE IMPLEMENTED:
                If y_range == None, sum the trace in the spectral direction to get a total profile, fit a gaussian 
                and use the result as a limit.) 
    Output:
        flux
        flux_var
        
    """
    #if y_range == None:
    #    vert_profile = np.sum(data, axis = 1)
    #    plt.plot(vert_profile)
    #    plt.show()
    #    y_range = [83,94]
    return np.sum(data[extraction_range[0]:extraction_range[1],:], axis = 0), \
                    np.sum(variance[extraction_range[0]:extraction_range[1],:], axis = 0)   

def determine_extraction_range(thumbnail, trace_width, spatial_sigma = 3):
    """helper function for optimal_extraction and sum_across_trace extraction. This function sums
    the given rotated thumbnail in the spectral direction, find the peak (assume one trace only),
    and return the range based on the given width of the trace (from findTrace) and the given 'sigmas'.

    Inputs:
        thumbnail: a 2D numpy array of the trace thumbnail. This must be rotated so that the spectral trace
                    is along x (axis 1).
        width: this is the trace width determined by findTrace
        sigma_spatial: how may widths away you want to extract. 
    Output:
        extraction_range: a 2-element list of [peak - sigma*width, peak + sigma*width]

    """
    spatial_profile = np.sum(thumbnail, axis = 1) #sum in the spectral direction to get a net spatial profile
    vert_max = np.argmax(spatial_profile) #locate the peak in this profile
    #define lower and upper boundaries of the extraction area. Remember to multiply the trace_width with cos(rotation angle)
    #because the pixel width changes as we rotate the image 
    lower = int(np.floor(vert_max - spatial_sigma*trace_width)) #is this LISP or what?
    upper = int(np.ceil(vert_max + spatial_sigma*trace_width))

    return [lower, upper]


def optimal_extraction(data, background, extraction_range, gain = 1.2, read_out_noise = 12, plot = 0):
    """
    This is Horne 1986 optimal extraction algorithm. This function assumes that background estimation
    is done prior to calling this function. In this implementation, no bad pixel finding is in place
    
    Inputs: 
        data: 2D numpy array of the data, *before background subtraction.
        background: 2D numpy array of the background used for background subtraction
        extraction_range: a 2-element list of the lower and upper limit in spatial (y) direction to extract the spectrum 
        gain: detector gain in electron/ADU, 1.2 for WIRC
        read_out_noise: standard deviation of read-out noise in electron. 12e- for WIRC
    Outputs:
        flux
        variance of the measured flux
    """

    #background = median_filter(background, 11) #assume no fine structure in background
    #First construct the variance estimate (eq 12, Horne 1986)
    variance = (read_out_noise/gain)**2 + data/gain
    #Compute a "standard" spectrum estimator by summing across trace
    flux_0, var_0 = sum_across_trace(data-background, variance, extraction_range)
    sky_flux, sky_var = sum_across_trace(background, variance, extraction_range)
    
    #Profile image; first pass, eq 14, then smooth with a median filter
    P_0 = np.copy(data)
    P_0 = (data - background)/flux_0 #this is dividing each column (x) in data-background by the sum in that column
    #smooth with a median filter only in the dispersion direction (x); note that the index is (y,x) here
    P_0 = median_filter(P_0, size = (1,10))
    #enforce non negativity and unity sum in the spatial direction
    P_0[P_0 < 0] = 0
    P_sum, var_Psum = sum_across_trace(P_0, variance, extraction_range)
    P_0 = P_0/P_sum
    
    #for i in range(P_0.shape[1]):
    #    plt.scatter(i, np.sum(P_0[extraction_range[0]:extraction_range[1],i]))
    if plot:
        plt.figure()
        for i in range(extraction_range[0], extraction_range[1]):
            plt.plot(P_0[i,:])
            #plt.plot(median_filter(P_0[i,:], 10))
            plt.ylim([-0.05,0.2])
            plt.xlim([30,140])
            plt.ylabel('P')
            plt.xlabel('Spectral pixel')
        plt.title('Spectral Profile')
        plt.show()

    #Now, optimization step. This version makes no attempts to deal with bad pixel
    #print(flux_0.shape, P_0.shape)
    
    #optimal 
    variance_opt = (read_out_noise/gain)**2 + (flux_0*P_0 + background)/gain
    #plt.imshow(variance_opt,origin ='lower', norm = LogNorm())
    #sum P(D-S)/V
    sum1, foo = sum_across_trace( P_0*(data-background)/variance_opt, variance, extraction_range) 
    #sum P
    sumP, foo = sum_across_trace(P_0,variance, extraction_range) 
    #sum P**2/V
    sum3, foo = sum_across_trace(P_0**2/variance_opt,variance, extraction_range) 
    
    #plt.plot(sum_across_trace(P_0*(data-background), variance, extraction_range)[0])
    #plt.plot(sum_across_trace(variance_opt, variance, extraction_range)[0],'b')
    
    
    
    #plt.plot(sum_across_trace(P_0*(data-background), variance, extraction_range)[0],'c')
    #plt.plot(sum_across_trace(variance_opt, variance, extraction_range)[0],'r')
    #plt.plot(sum1 ,'r')
    #plt.plot(sumP,'g')
    #plt.plot(sum3,'b')
    #plt.show()
    flux_opt_final = np.nan_to_num(sum1/sum3)
    variance_opt_final = np.nan_to_num(sumP/sum3 )
    
    #plt.show()
    return flux_opt_final, variance_opt_final

def spec_extraction(thumbnails, slit_num, filter_name = 'J', plot = True, output_name = None, sub_background=True, \
    bkg_sub_shift_size = 21, method = 'weightedSum', skimage_order=4, width_scale=1., diag_mask = False, trace_angle = -45,\
     fitfunction = 'Moffat', sum_method = 'weighted_sum', box_size = 1, poly_order = 4, mode = 'pol', spatial_sigma = 3,\
     verbose = True, quiet=False):
    """
    This is the main function to perform spectral extraction on the spectral image
    given a set of thumbnails.

    ##########
    Parameters
    ##########

    thumbnails:     a cube containing 4 spectral traces Qp, Qm, Up, Um.
    slit_num:       the number of slit the source is in | (0) (1) (2) |, or 'slitless'
    filter_name:    a string of which filter the source is in.
    plot:           whether or not to plot results here. Should plot later using plot_trace_spectra
    output_name:    name of the fits file to write the result. If None, no file is written 
    sub_background: whether or not to run a background subtraction routine by shift and subtracting
    *method:        method for spectral extraction. Choices are
                        (i) skimage: this is just the profile_line method from skimage. Order for interpolation 
                                        is in skimage_order parameter (fast).
                        (ii) weightedSum: this is 2D weighted sum assfuming Gaussian profile. Multiply the PSF with data
                                        and sum for each location along the dispersion direction (fast). The width of the Gaussian
                                        is based on the measured value by 'findTrace'. One can adjust this using the parameter 'width_scale'.
                        (iii) fit_across_trace: this method rotates the trace, loops along the dispersion direction, and fit a profile in the 
                                        spatial direction. The fit function is either 'Moffat' or 'Gaussian'. One can also
                                        select how to extract flux: by summing the fitted model, or the data weighted by the model.
                                        ('model_sum' vs 'weighted_sum'). These are in 'fitfunction' and 'sum_method' parameters.
                                        box_size determine how many columns of pixel we will use. poly_order is the order of polynomial used to
                                        fit the background. 
                        (iv) optimal_extraction: This is a preferred method. 
    diag_mask:      if True, the area away from the trace will be masked out 

    skimage_order, width_scale, fitfunction, sum_method, see method above.
    mode - use either 'pol' or 'spec'

    #######
    Outputs
    #######

    spectra:        flux in ADU 
    spectra_std:    standard deviation of flux

    """

    # Define lists to collect results.
    wavelengths = []
    spectra = [] #This vector collects extracted spectra from 4 traces
    spectra_std = []

    if mode=='pol':
        ntraces = 4 #how many traces? 4 for WIRC-POL
        if plot:
            fig = plt.figure(figsize=(11,4))
        
    if mode=='spec':
        ntraces = 1 #how many traces? 1 for WIRC-SPEC
        if plot:
            fig = plt.figure(figsize=(11,4))

    thumbnails_copy = copy.deepcopy(thumbnails)

    #Flip some of the traces around.
    if mode=='pol': 
        thumbnails_copy[1,:,:] = thumbnails_copy[1,-1::-1, -1::-1] #flip y, x. Bottom-right
        thumbnails_copy[2,:,:] = thumbnails_copy[2,:,-1::-1] #flip x #Top-right
        thumbnails_copy[3,:,:] = thumbnails_copy[3,-1::-1, :] #flip y #Bottom-left

        trace_titles=["Top-Left", "Bottom-Right", "Top-Right", "Bottom-left"]

    if mode=='spec':
        trace_titles=['Extracted Spectrum']

    #lists to collect widths and angles of the traces
    widths = []
    angles = []
        
    for j in range(ntraces):    
        trace_title = trace_titles[j]

        thumbnail = thumbnails_copy[j,:,:]
        if mode=='pol' and verbose:
            print("Extracting spectra from trace {} of 4".format(j))
        if mode=='spec' and verbose:
            print("Extracting spectrum".format(j))

        #Should we subtrack the background? 
        if sub_background:        
            #############################################
            ######If data is in the slit mode, perform shift and subtract to remove background
            #############################################

            #if slit_num != 'slitless':

             #   bkg_sub, bkg = shift_and_subtract_background(thumbnail, obj_slit = slit_num)

                #Mask out the area outside of the slit hole.
             #   thumb_mask = makeDiagMask(len(bkg_sub[0]), slit_hole_diameter+3)
             #   bkg_sub = bkg_sub * thumb_mask
             #   bkg = bkg * thumb_mask
        
            #############################################
            ######For each thumbnail, fit for background
            #############################################
            #else:
             #   bkg_sub, bkg = fit_and_subtract_background(thumbnail)

        #For now, do shift and subtract always
            # bkg = (shift( thumbnail, [0,-21] ) + shift( thumbnail, [0,21] ))/2
            bkg_stack = np.dstack((shift( thumbnail, [0,-bkg_sub_shift_size ]),shift( thumbnail, [0,bkg_sub_shift_size ] ),thumbnail))
            bkg = np.nanmedian(bkg_stack, axis=2)
            bkg_sub = thumbnail - bkg

            #Mask out the area outside of the slit hole.
            # thumb_mask = makeDiagMask(len(bkg_sub[0]), slit_hole_diameter+3)
            # bkg_sub = bkg_sub * thumb_mask
            # bkg = bkg * thumb_mask

        else: 
            bkg_sub = np.copy(thumbnail)
            bkg = thumbnail*0.

        ###############################
        #locate the trace in the data##
        ###############################
        #trace is the vector of the y location of the trace for each x location in the frame
        #width is the width of the trace at its brightest point. 
        start = time.time()            

        if diag_mask:
            mask = makeDiagMask(np.shape(bkg_sub)[0], 25)
            bkg_sub[~mask] = 0.

        raw, trace, trace_width, measured_trace_angle = findTrace(bkg_sub, poly_order = 1, weighted=True, plot = 0, diag_mask=diag_mask,mode=mode) #linear fit to the trace
        if verbose:
            print("Trace width {}".format(trace_width))

        weight_width = trace_width*width_scale

        widths += [trace_width]
        angles += [measured_trace_angle]
        
        #raw, trace, trace_width, measured_trace_angle = findTrace(bkg_sub, poly_order = 1, weighted=True, plot = 0, diag_mask=diag_mask,mode=mode) #linear fit to the trace
        #weight_width = trace_width*width_scale
        #if verbose:
        #    print("Trace width {}".format(trace_width))

        ######################################
        ######Call spectral extraction routine
        ######################################
                

        ##skimage profile_line trying different interpolation orders
        if method == 'skimage':
            if verbose:
                print("Extraction by skimage")
            #linewidth = 20 #This should be adjusted based on fitted seeing.
            linewidth = 2*trace_width #use the measured trace width
            spec_res = profile_line(bkg_sub, (0,trace[0]), (len(bkg_sub[1]),trace[-1]), linewidth = linewidth,order =  skimage_order)                
            spectra.append(spec_res)
            spectra_std.append((gain*spec_res+linewidth * sigma_ron**2)/gain**2) #poisson + readout

        elif method == 'weightedSum':
            #define PSF (Gaussian for now)
            psf = np.zeros((21,21))
            xx,yy = np.mgrid[0:np.shape(psf)[0], 0:np.shape(psf)[1]]

            psf = models.Gaussian2D(amplitude = 1, y_mean = psf.shape[0]//2, x_mean = psf.shape[1]//2, \
                                   y_stddev = weight_width, x_stddev = weight_width)(yy,xx)
            #psf = models.Gaussian2D(amplitude = 1, y_mean = psf.shape[0]//2, x_mean = psf.shape[1]//2, \
            #                       y_stddev = 2, x_stddev = 2)(yy,xx)

            #extract
            spec_res, spec_var = weighted_sum_extraction(bkg_sub, trace, psf)
            spectra.append(spec_res)
            spectra_std.append(np.sqrt(spec_var))

        elif method == 'fit_across_trace':
            if verbose:
                print("trace angle is ", measured_trace_angle," deg")
            if trace_angle == None:
                rotate_spec_angle = measured_trace_angle #use the measured angle
            else:
                rotate_spec_angle = trace_angle #use the given value
                if verbose:
                    print("use given ", trace_angle," instead. change this by setting trace_angle to None")
            #start = time.time()
            spec_res, spec_var , residual= fitAcrossTrace_aligned(bkg_sub, stddev_seeing = weight_width, plot =  False, return_residual = 1, \
                                                                        fitfunction = fitfunction, box_size = box_size, poly_order = poly_order,
                                                                        sum_method = sum_method, trace_angle = rotate_spec_angle) #Do not use variance from this method
            #plt.imshow(residual, origin = 'lower',vmin = -200, vmax = 200)
            #plt.colorbar()
            #plt.show()
            #spec_res, spec_var = fitAcrossTrace_aligned(bkg_sub, stddev_seeing = weight_width, plotted =  0, return_residual = 0) #Do not use variance from this method

            if verbose:
                print('fit_across_trace takes {} s'.format(time.time()-start))
            spectra.append(spec_res)
            spectra_std.append(np.sqrt(spec_var)) #again, don't rely on the variance here yet.
        elif method == 'sum_across_trace':
            #First, determine the angle to rotate the spectrum, this can either be given or measured by findTrace
            if verbose:
                print("trace angle is ", measured_trace_angle," deg")
            if trace_angle == None: #if the trace angle is not given, use the measured angle
                rotate_spec_angle = measured_trace_angle 
            else: #otherwise, use the given value
                rotate_spec_angle = trace_angle 
                if verbose:
                    print("using given angle of ", trace_angle," deg. change this by setting trace_angle to None")

            #rotate the spectrum here. rotation axis is the middle of the image
            width_thumbnail = bkg_sub.shape[0]
            sub_rotated = frame_rotate(bkg_sub, rotate_spec_angle+180,cxy=[width_thumbnail/2,width_thumbnail/2])
            rotated = frame_rotate(thumbnail, rotate_spec_angle+180,cxy=[width_thumbnail/2,width_thumbnail/2])

            #determine the extraction range based on the width parameter
            #first, find the peak
            ext_range = determine_extraction_range(sub_rotated, trace_width/np.abs(np.cos(np.radians(rotate_spec_angle))), spatial_sigma = 3)



            #call the optimal extraction method, remember it's sum_across_trace(bkg_sub_data, bkg, extraction_range, etc)
            spec_res, spec_var = sum_across_trace( sub_rotated, rotated , ext_range) 

            spectra.append(spec_res)
            spectra_std.append(np.sqrt(spec_var)) 
             
        elif method == 'optimal_extraction':
            #First, determine the angle to rotate the spectrum, this can either be given or measured by findTrace
            if verbose:
                print("trace angle is ", measured_trace_angle," deg")
            if trace_angle == None: #if the trace angle is not given, use the measured angle
                rotate_spec_angle = measured_trace_angle 
            else: #otherwise, use the given value
                rotate_spec_angle = trace_angle 
                if verbose:
                    print("using given angle of ", trace_angle," deg. change this by setting trace_angle to None")

            #rotate the spectrum here. rotation axis is the middle of the image
            width_thumbnail = bkg_sub.shape[0]
            sub_rotated = frame_rotate(bkg_sub, rotate_spec_angle+180,cxy=[width_thumbnail/2,width_thumbnail/2])
            rotated = frame_rotate(thumbnail, rotate_spec_angle+180,cxy=[width_thumbnail/2,width_thumbnail/2])

            #determine the extraction range based on the width parameter
            #first, find the peak
            # spatial_profile = np.sum(sub_rotated, axis = 1) #sum in the spectral direction to get a net spatial profile
            # vert_max = np.argmax(spatial_profile) #locate the peak in this profile
            # #define lower and upper boundaries of the extraction area. Remember to multiply the trace_width with cos(rotation angle)
            # #because the pixel width changes as we rotate the image 
            # lower = int(np.floor(vert_max - spatial_sigma*trace_width/np.abs(np.cos(np.radians(rotate_spec_angle))))) #is this LISP or what?
            # upper = int(np.ceil(vert_max + spatial_sigma*trace_width/np.abs(np.cos(np.radians(rotate_spec_angle)))))

            ext_range = determine_extraction_range(sub_rotated, trace_width/np.abs(np.cos(np.radians(rotate_spec_angle))), spatial_sigma = 3)
            #call the optimal extraction method, remember it's optimal_extraction(non_bkg_sub_data, bkg, extraction_range, etc)
            spec_res, spec_var = optimal_extraction( rotated, rotated - sub_rotated, ext_range, 1.2, 12, plot = 0) 

            spectra.append(spec_res)
            spectra_std.append(np.sqrt(spec_var)) 

    
        else:
            print("method keyword not understood, please choose method='weightedSum', 'fit_across_trace', or method='skimage'")
            return None, None
        
        #Plotting
        if plot:
            ax = fig.add_subplot(2,ntraces,j+1)
            ax.set_title(trace_title)
     
            # We'll keep the scaling the same for all the traces
            # So we'll grab the min and max from the first trace
            # Let's also use log scaling and a better colomap

            if j == 0: 
                #For display purposes we'll get the min and max from where the trace is (in the first ) (Do this later)
                

                #Where is the trace? (It's roughly diagonal with some height)
                y, x = np.indices(np.shape(thumbnail))
                trace_height = 20. #This height will depend on the a few things, but we'll start with 20
                x_buffer = 22 #The thumbnail will be oversized a bit
                trace_inds = np.where( (y > - x + np.max(x) - trace_height) & ( y < -x + np.max(x) + trace_height) & (x > x_buffer) & (x < np.max(x)-x_buffer))

                vmin = np.nanmin(thumbnail[trace_inds])
                vmax = np.nanmax(thumbnail[trace_inds])

                vmin_bkg = np.nanmin( (thumbnail - bkg)[trace_inds] )
                vmax_bkg = np.nanmax( (thumbnail - bkg)[trace_inds] )


            im1 = ax.imshow(thumbnail, origin = 'lower', cmap='YlGnBu', vmin=vmin, vmax=vmax,)# norm=LogNorm(vmin=vmin, vmax=vmax))
            


        if plot:
            ax2 = fig.add_subplot(2,ntraces,j+1+ntraces)    

            im2 = ax2.imshow(bkg_sub, origin = 'lower', cmap='YlGnBu', vmin = vmin_bkg, vmax = vmax_bkg)#norm=LogNorm(vmin=vmin_bkg, vmax=vmax_bkg))
            #ax2.plot(loc[:,0],loc[:,1])
            # if j == ntraces-1:
            #     plt.colorbar(im2, ax=ax2)

            #f.PrimaryHDU(thumbnail - bkg).writeto('sub1.fits')

            fig.subplots_adjust(right=0.9)
            cbar_ax=fig.add_axes([0.9,0.535, 0.015, 0.37])
            cbar = fig.colorbar(ax.images[0], cax=cbar_ax)

            cbar_ax2=fig.add_axes([0.9,0.085, 0.015, 0.38])
            cbar2 = fig.colorbar(ax2.images[0], cax=cbar_ax2)
            #over plot traces                
            ax2.plot(raw,'k')
            ax2.plot(trace,'w')
    
    if plot: 
        plt.show()
    #print(spectra)
    #print(np.array(spectra).shape)
    #pdb.set_trace()

    if mode=='pol':
        min_len = min(len(spectra[0]),len(spectra[1]),len(spectra[2]),len(spectra[3]))
        for i in range(4):
            spectra[i] = spectra[i][0:min_len]
            spectra_std[i] = spectra_std[i][0:min_len]

    return np.array(spectra), np.array(spectra_std), np.array(widths), np.array(angles)  #res_spec is a dict, res_stddev and thumbnails are list

def rough_wavelength_calibration_v1(trace, filter_name):
    """
    roughWaveCal does rough wavelength calibration by comparing the extracted profile
    to the filter transmission function. It is assumed that the continuum trace 
    is detected, so this only works on sources with bright continuum.
    
    This is done by fitting a function that translate pixel index to wavelength,
    for which the trace coincide best with the transmission function. 
    
    Input:  trace: a 1d vector with the extracted spectral trace
            filter_name: string of filter name ('J' or 'J' for now)
    Output: a vector of wavelength corresponding to each pixel value
    """
    #First, call getFilterInfo for the transmissions curve
    lb,dlb,f0,filter_trans_int, central_wl_pix = getFilterInfo(filter_name) 
    
    wla = np.linspace(lb-dlb, lb+dlb, 200)
   # plt.plot(wl,filter_trans_int(wl))
   # plt.show()

   # plt.plot(trace)
   # plt.show()
    #flatten the top of the trace to remove spectral features
    trace = np.array( [  max( min(x, 0.5*np.max(trace)), 0.05*np.max(trace)) for x in trace] )
    x = np.arange(len(trace)) #pixel index
    # wl0 = lb-dlb+0.05
    def pixToWl(y):
        """dummy function to fit pixel index to wavelength
        Take m and wl0 in a linear relation scaling pixel index and wavelength,
        then maximize transmission * trace
        """
        m = y[0] #y is [slope, intercept]
        wl0 = y[1]
        
        wl = wl0 + m*x #linear relation between pixel index and wavelength
        #plt.plot(wla, filter_trans_int(wla))

        #print(wl)
        transmission = filter_trans_int(wl) #vector of transmission in the trace
        transmission = np.array( [ max(min(x, 0.5*np.max(transmission)), 0.05*np.max(transmission)) for x in transmission] )
        #plt.plot(wl, transmission/np.max(transmission),'b')
        #plt.plot(wl, trace/np.max(trace),'r')
        #plt.show()
        #print( m/wlPerPix, wl0, -np.log(np.sum(transmission*trace)))
        #now compute transmission
        #return -np.log(np.sum(transmission*trace))
        return 1/(np.max(np.correlate(transmission/np.max(transmission),trace/np.max(trace)) ))
    #print(pixToWl([wlPerPix,lb-dlb]))
    
    #res = basinhopping(pixToWl, [wlPerPix,lb-dlb], stepsize = 0.01*wlPerPix)#, bounds = [(0.9*wlPerPix,1.1*wlPerPix),\
                                                   #     (lb-dlb -0.5, lb-dlb +0.5)])
    #res = minimize(pixToWl, 0.5*wlPerPix, bounds = [(0.4*wlPerPix, 0.8*wlPerPix)])    
    res = basinhopping(pixToWl, [1*wlPerPix,lb-dlb+0.05], minimizer_kwargs = dict(bounds = [(0.5*wlPerPix,1.1*wlPerPix),\
                                                        (lb-dlb-0.2, lb-dlb+0.2)]\
                                                        , tol = 1e-6))
    # print(res)
    #Plotting   
    transmission = filter_trans_int(wla)
    transmission = np.array( [max(min(x, 0.5*np.max(transmission)), 0.05*np.max(transmission)) for x in transmission] )
    plt.plot(wla, transmission/np.max(transmission),'b')
    plt.plot(res.x[1] + res.x[0]*x, trace/np.max(trace),'r')
    # plt.show()    
    return res.x[1] + res.x[0]*x
   # 

def rough_wavelength_calibration_v2(trace, filter_name, lowcut=0, highcut=-1):
    """
    """

    #Make a copy
    trace_copy = copy.deepcopy(trace)

    trace = trace[lowcut:highcut]

    lb,dlb,f0,filter_trans_int, central_wl_pix = getFilterInfo(filter_name) 
    wla = np.linspace(lb-dlb, lb+dlb, len(trace))
    x = np.arange(len(trace))
    # print((trace))
    grad = np.gradient(trace) #min and max arer the steep slopes at the filter cuts
    
    trans = filter_trans_int(wla)
    grad_trans = np.gradient(trans)
    #for location of peak gradients in the transmission function, we know the wl
    wl_up = wla[np.argmax(grad_trans)]
    wl_down = wla[np.argmin(grad_trans)]
    
    slope = (wl_down - wl_up )/(np.argmin(grad) - np.argmax(grad))

    x = np.arange(len(trace_copy))
    return slope*(x - np.argmax(grad)) + wl_up

def align_set_of_traces(traces_cube, ref_trace):
    """
    align_set_of_traces takes a cube of traces with dimension (number_of_traces, length_of_each_trace) 
    and align them with respect to the reference trace of the same length. 
    """
    new_cube = np.zeros(traces_cube.shape)
    ref = ref_trace
    #fig, (ax, ax2) = plt.subplots(2,4, figsize = (20,10))
    for i, j in enumerate(traces_cube):
        #print(np.max(ref), np.max(j))
        #corr = fftconvolve(ref/np.nanmax(ref), (j/np.nanmax(j))[::-1] )
        corr = fftconvolve(np.nan_to_num(ref/np.nanmax(ref)), np.nan_to_num((j/np.nanmax(j))[::-1]) )

        shift_size = np.nanargmax(corr) - len(ref) +1
        #print(shift_size)
        new_cube[i] = shift(traces_cube[i], shift_size)
            
    return new_cube

def smooth_spectra(spectra, kernel = 'Gaussian', smooth_size = 3):
    """
    Convolve the spectra with either Gaussian or Box kernel of the specified size, using astropy.
    Spectra can be either a 1-d array or a 2d array of spectra
    """
    if smooth_size > 1: #only go through all this if smooth_size > 1
        if kernel == 'box':
            smooth_ker = Box1DKernel(smooth_size)
        elif kernel == 'Gaussian':
            smooth_ker = Gaussian1DKernel(smooth_size)
        else:
            raise ValueError('Kernel can be either box or Gaussian')
        #one spectrum or a cube of spec
        if len(spectra.shape) ==1 : #just one spectrum
            out_spectra = convolve(spectra, smooth_ker)

        else:
            out_spectra = np.zeros(spectra.shape)
            for i in range(spectra.shape[0]):
                out_spectra[i] = convolve(spectra,smooth_ker)
    else:
        out_spectra = spectra

    return out_spectra #same dimension as spectra


def compute_stokes_from_traces(trace_plus, trace_plus_err,trace_minus, trace_minus_err, plotted = False):
    """
    Possibly a temporary function used to compute Q and U by taking a plus and 
    a minus traces, align them in frequency, then compute q = Qp - Qm / Qp + Qm
    and u = Up - Um / Up + Um
    
    This should be replaced when wavelength calibration is done.     
    
    Input: trace_plus, trace_minus: vectors of flux along diagonal
    Output: q or u, scale factor m, and the shift ds
    
    """
    #First we find maximum correlation with different scale factor
    
    def shiftSpec(ds):
        """helper function, compute correlation between the 2 traces given shift"""
        #print(ds)
        res = -np.log(sum(trace_plus*shift(trace_minus, ds[0], order =4) ))
        #print(ds, res)
        return res
    
    #First, find shift that maximizes correlation between 2 traces
    #this is assuming that polarization is modest 
    
    res = basinhopping(shiftSpec, 0.0, stepsize = 0.1)#, tol = 1e-20)#, bounds = [(-5,5)] )#, tol = 1e-20)
    #res = brute(shiftSpec, (-5,5) )#, ((-5,5)), finish = minimize)
    #This is the shift we need to apply to trace_minus    
    ds = res.x[0]
    #print(ds)
    if ds != 0: 
        trace_minus = shift(trace_minus, ds)
    if plotted:
        plt.plot(trace_plus,'r')
        plt.plot(trace_minus,'b')
        # plt.show()
    #Compute stoke paremeters q or u
    stoke = (trace_plus - trace_minus)/(trace_plus + trace_minus)
    #Compute uncertainty
    stoke_err = np.abs(2/(trace_plus+trace_minus)**2) * \
                np.sqrt((trace_minus*trace_plus_err)**2+(trace_plus*trace_minus_err)**2)
    return stoke, stoke_err, ds

def compute_p_and_pa( q, q_err, u, u_err):
    """
    Computes degree and angle of polarization with associated uncertainties
    from given q and u. These should be corrected for instrumental polarization 
    Input:
        vectors q, u, and their uncertainties
    Output: 
        vector p, theta and their uncertainties
    Formulae used:
        p = sqrt(q^2 + u^2)
        dp = 1/p sqrt( (q dq)^2 + (u du)^2)
        theta = 1/2 atan(u/q)
        dtheta = (28.65 dp)/p 
    """
    #Compute deg of polarization
    p = np.sqrt(q**2 + u**2)
    dp = 1/p * np.sqrt( (q * q_err)**2 + (u * u_err)**2)
    p = np.sqrt(p**2 - dp**2) #debiased deg of polarization
    
    #Compute angle of polarization
    theta = 1/2.*np.arctan2(u,q)
    dtheta = 28.65*dp/p
    
    return p, dp, theta, dtheta

def compute_polarization(trace_spectra, filter_name = 'J', plot=False, cutmin=0, cutmax=190):
    """
    compute_polarization takes results from spec_extraction, then compute normalized stoke parameters
    q, u, and equivalently p and chi along with the corresponding uncertainties. 

    Input:
        trace_spectra   -   An array of size [4,3, m], where each m-sized spectrum as a wavelength, a flux and a flux error. 
                            The first dimension is for each trace, the second holds, wavelength, flux and flex_std. 
        cutmin          -   Chop off this many elements from the beginning of each spectra
        cutmin          -   Chop off all the indices after this number from each spectra
    """

    Qp_out = []
    Qp_var_out = []
    Qm_out = []
    Qm_var_out = []
    Up_out = []
    Up_var_out = []
    Um_out = []
    Um_var_out = []
    Q_out = []
    Q_var_out = []
    U_out = []
    U_var_out = []
    P_out = []
    P_var_out = []
    Theta_out = []
    Theta_var_out = []

    Qp = trace_spectra[0,1,:]
    Qp_var = trace_spectra[0,2,:]
    Qm = trace_spectra[1,1,:]
    Qm_var = trace_spectra[1,2,:]
    Up = trace_spectra[2,1,:]
    Up_var = trace_spectra[2,2,:]
    Um = trace_spectra[3,1,:]
    Um_var = trace_spectra[3,2,:]
    
    #Clip the heads and tails of spectra. Magic numbers for now! 
    #TODO: This whole process needs to be refined. 
    Qp = Qp[cutmin:cutmax]
    Qp_var = Qp_var[cutmin:cutmax]
    Qm = Qm[cutmin:cutmax]
    Qm_var = Qm_var[cutmin:cutmax]
    
    Up = Up[cutmin:cutmax]
    Up_var = Up_var[cutmin:cutmax]
    Um = Um[cutmin:cutmax]
    Um_var = Um_var[cutmin:cutmax]
    
    wlQp = trace_spectra[0,0,cutmin:cutmax]
    wlQm = trace_spectra[1,0,cutmin:cutmax]
    wlUp = trace_spectra[2,0,cutmin:cutmax]
    wlUm = trace_spectra[3,0,cutmin:cutmax]

    # if plot: 
    #     fig=plt.figure()

    #     #Plot Q+ and Q-
    #     fig.add_subplot(231) 
    #     #Wavelength calibration now working great right now
    #     # plt.errorbar(wlQp, Qp , color = 'r', label = 'Q+')
    #     # plt.errorbar(wlQm, Qm, color = 'b', label = 'Q-')
    #     plt.errorbar(range(len(Qp)), Qp , yerr = np.sqrt(Qp_var), color = 'r', label = 'Q+')
    #     plt.errorbar(range(len(Qm)), Qm , yerr = np.sqrt(Qm_var), color = 'b', label = 'Q-')
    #     plt.legend()

    #     #Plot Q+ - Q-
    #     fig.add_subplot(232)
    #     plt.errorbar(range(len(Qp)), Qp-Qm, yerr=np.sqrt(Qp_var + Qm_var), color = 'purple', label = 'Q')
    #     plt.legend()

    #     #Plot (Q+ - Q-)/(Q+ + Q-)
    #     fig.add_subplot(233)
    #     plt.errorbar(range(len(Qp)), (Qp-Qm)/(Qp+Qm), color = 'green', label = 'Q/I')
    #     plt.ylim(-0.1,0.1)
    #     plt.legend()

        
    #     fig.add_subplot(234)
    #     #Wavelength calibration now working great right now
    #     # plt.errorbar(wlUp, Up , color = 'r', label = 'U+')
    #     # plt.errorbar(wlUm, Um, color = 'b', label = 'U-')
    #     plt.errorbar(range(len(Up)), Up , yerr = np.sqrt(Up_var), color = 'r', label = 'U+')
    #     plt.errorbar(range(len(Um)), Um , yerr = np.sqrt(Um_var), color = 'b', label = 'U-')
        
    #     plt.legend()

    #     fig.add_subplot(235)
    #     plt.errorbar(range(len(Up)), Up-Um, yerr=np.sqrt(Up_var + Um_var), color = 'purple', label = 'U')

    #     fig.add_subplot(236)
    #     plt.errorbar(range(len(Up)), (Up-Um)/(Up+Um), color = 'green', label = 'U/I')
    #     plt.ylim(-0.1,0.1)
    #     plt.legend()

    #     plt.savefig('Q+U'+str(i)+'.pdf')
    # plt.show()
    
    #smooth_ker = Gaussian1DKernel(stddev = np.nanmedian(res_stddev[i]))
    
    q, dq, dsq = compute_stokes_from_traces(Qp, np.sqrt(Qp_var), Qm, np.sqrt(Qm_var), plotted = False)
    u, du, dsu = compute_stokes_from_traces(Up, np.sqrt(Up_var), Um, np.sqrt(Um_var), plotted = False)
    
    #shift traces so aligned traces can be returned
    # Qm = shift(Qm, dsq)
    # Qm_var = shift(Qm_var, dsq)

    # Um = shift(Um, dsu)
    # Um_var = shift(Um_var, dsu)        
    
    return wlQp, q, dq, wlUp, u, du
    #smooth at seeing size
    #q = convolve(q, smooth_ker)
    #u = convolve(u, smooth_ker)   
    
    # plt.figure() 
    # plt.errorbar(range(len(q)),q,dq, color = 'r', label = 'q')
    # plt.errorbar(range(len(u)),u,du, color = 'b', label = 'u')
    # #compute degree/angle of polarization
    # p, dp, theta, dtheta = compute_p_and_pa(q,dq,u,du)
    
    # plt.errorbar(range(len(p)),p ,dp, color = 'k', label = 'p')
    # plt.legend()
    # plt.ylim([-0.01,0.02])
    # plt.savefig('qup'+str(i)+'.pdf')
    # plt.show()
    
    # plt.figure()
    # plt.errorbar(range(len(theta)), theta, dtheta)
    # plt.ylabel('$\theta$')
    # plt.savefig('theta'+str(i)+'.pdf')

    # Qp_out += [[Qp]]
    # Qp_var_out += [[Qp_var]]
    # Qm_out += [[Qm]]
    # Qm_var_out += [[Qm_var]]
    # Up_out += [[Up]]
    # Up_var_out += [[Up_var]]
    # Um_out += [[Um]]
    # Um_var_out += [[Um_var]]
    # P_out += [[p]]
    # P_var_out += [[dp]]
    # Theta_out += [[theta]]
    # Theta_var_out += [[dtheta]]


    # ############################################
    # ######### MAKE THE PLOT ###################
    # ###########################################

    # if plot:
    #     fig=plt.figure()

    #     #Colors from Tableau 10 Medium
    #     x_color=(114/255.,158/255.,206/255.) #Blue
    #     y_color=(237/255.,102/255.,92/255.) #Red
    #     gpi_color=(173/255.,139/255.,201/255.) #Purple
    #     chauvin_color=(103/255.,191/255.,92/255.) #Green
    #     nielsen_color=(255/255.,158/255.,74/255) #Orange

    #     ax1 = fig.add_subplot(511,  ylabel="I [ADU]")
    #     # plt.xlim(0.430,0.920)
    #     # plt.xlim(500,900)
    #     # plt.ylim(0,5000)
    #     ax2 = fig.add_subplot(512,  ylabel="Q/I")
    #     # plt.xlim(0.430,0.920)
    #     # plt.xlim(500,900)
    #     # plt.ylim(-0.2,0.2)
    #     ax3 = fig.add_subplot(513,  ylabel="U/I")
    #     # plt.xlim(0.430,0.920)
    #     # plt.xlim(500,900)
    #     # plt.ylim(-0.2,0.2)
    #     ax4 = fig.add_subplot(514, ylabel = 'P/I')
    #     # plt.xlim(0.430,0.920)
    #     # plt.xlim(500,900)
    #     # plt.ylim(0.,0.1)
    #     ax5 = fig.add_subplot(515,  ylabel=r"PA [$^\circ$]",  xlabel="Wavelength [nm]")
    #     # plt.xlim(0.430,0.920)
    #     # plt.xlim(500,900)
    #     # plt.ylim(170,190)

    
    #     #Plot Stokes I
    #     ax1.plot(range(len(Qp)), Qp+Qm, color='k')
    #     ax1.plot(range(len(Up)), Up+Um, color='k')
    #     # ax1.fill_between(lamb, stokesI-stokesI_err, stokesI+stokesI_err, color=x_color, alpha=0.7)
    #     # ax1.fill_between(lamb, stokesI-stokesI_err, stokesI+stokesI_err, color=x_color, alpha=0.7)
    #     # ax1.set_ylim(0,4000)
    #     # ax1.set_yticks([0,1000,2000,3000,4000])
    #     ax1.set_xticklabels([])
    #     ax1.grid()
    #     plt.locator_params(nbins=5)

    #     #PLot Stokes Q
    #     # ax2.plot(lamb, stokesQ/stokesI)
    #     ax2.errorbar(range(len(q)), q, dq, color='b')
    #     # ax2.fill_between(range(len(q)), q-dq, q+dq, color=x_color, alpha=0.7)
    #     # ax2.set_ylim(-1.,1.)
    #     # ax2.set_yticks([-0.03,0,0.03,0.06])
    #     ax2.set_xticklabels([])
    #     ax2.grid()
    #     plt.locator_params(nbins=5)

    #     #Plot Stokes U
    #     # ax3.plot(lamb, stokesU/stokesI)
    #     ax3.errorbar(range(len(u)), u, du, color='purple')
    #     # ax3.fill_between(lamb, stokesU-stokesU_err, stokesU+stokesU_err, color=gpi_color, alpha=0.7)
    #     # ax3.set_ylim(-1,1)
    #     # ax3.set_yticks([-0.03,-0.01,0.01,0.03])
    #     ax3.set_xticklabels([])
    #     ax3.grid()
    #     plt.locator_params(nbins=5)

    #     # Plot Stokes P
    #     ax4.errorbar(range(len(p)), p, dp, color='red')
    #     # ax4.set_yticks([0,0.02,0.04,0.06])
    #     # ax4.set_ylim(0.,1.)
    #     ax4.set_xticklabels([])
    #     ax4.grid()
    #     plt.locator_params(nbins=5)

    #     # Plot the Position Angle
    #     ax5.errorbar(range(len(theta)), np.degrees(theta), dtheta, color='g')
    #     # ax5.fill_between(lamb, pa+180-pa_err, pa+180+pa_err, color=chauvin_color, alpha=0.7)
    #     ax5.grid()
    #     plt.locator_params(nbins=5)

    #     plt.savefig("plot_stokes_spectrum_debias"+str(i)+".pdf", bbox_inches="tight")
    #     # plt.show()

    # # return Qp, Qp_var, Qm, Qm_var, Up, Up_var, Um, Um_var, q, dq, u, du, p, dp, theta, dtheta
    # return Qp_out, Qp_var_out, Qm_out, Qm_var_out, Up_out, Up_var_out, Um_out, Um_var_out, Q_out, Q_var_out, U_out, U_var_out, P_out, P_var_out, Theta_out, Theta_var_out
    
