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
from scipy.optimize import least_squares
from scipy.optimize import basinhopping
from scipy.ndimage import shift, median_filter
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from skimage.measure import profile_line

from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel, Box1DKernel, convolve
from astropy.io import fits as f
from astropy import stats

from photutils import RectangularAperture, aperture_photometry,make_source_mask

#From other packages
from wirc_drp.utils.image_utils import locationInIm, shift_and_subtract_background, fit_and_subtract_background, findTrace
from wirc_drp.masks.wircpol_masks import *
from wirc_drp.utils import image_utils

from astropy.stats import sigma_clipped_stats

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

# @profile
def optimal_extraction(data, background, extraction_range, bad_pixel_mask = None, bad_pix_masking = 1, gain = 1.2, read_out_noise = 12, \
                        verbose = 0, plot = 0, niter = 1, sig_clip = 5,spatial_smooth=1,spectral_smooth=10):

    """
    This is Horne 1986 optimal extraction algorithm. This function assumes that background estimation
    is done prior to calling this function. In this implementation, no bad pixel finding is in place
    
    Inputs: 
        data: 2D numpy array of the data, *before background subtraction.
        background: 2D numpy array of the background used for background subtraction
        extraction_range: a 2-element list of the lower and upper limit in spatial (y) direction to extract the spectrum 
        bad_pixel_mask: if the bad pixel mask has been calculated before, pass it here. *** 1 is good, 0 is bad  ***
        bad_pix_masking: if False, do not compute bad pixel rejection.
        gain: detector gain in electron/ADU, 1.2 for WIRC
        read_out_noise: standard deviation of read-out noise in electron. 12e- for WIRC
        niter: the number of iteration loop to reject bad pixel. 1 should be enough
    Outputs:
        flux
        variance of the measured flux
    """
    #If bad_pixel_mask is not given
    if bad_pixel_mask is None:
        bad_pixel_mask = np.ones(data.shape)

    #background = median_filter(background, 11) #assume no fine structure in background
    #First construct the variance estimate (eq 12, Horne 1986)
    variance = (read_out_noise/gain)**2 + np.abs(data)/gain
    variance = median_filter(variance,6) #Smooth this - try to lower significance of bad pixels
    #Compute a "standard" spectrum estimator by summing across trace
    flux_0, var_0 = sum_across_trace(data-background, variance, extraction_range)
    sky_flux, sky_var = sum_across_trace(background, variance, extraction_range)
    Mask_old = np.ones(data.shape) #this is the initial bad pixel mask, assuming that all is good
    
    while niter > 0:
    
        #Profile image; first pass, eq 14, then smooth with a median filter
        P_0 = np.copy(data)
        P_0 = (data - background)/flux_0 #this is dividing each column (x) in data-background by the sum in that column
        
        #smooth with a median filter only in the dispersion direction (x); note that the index is (y,x) here
        P_0 = median_filter(P_0, size = (spatial_smooth,spectral_smooth))

        ##enforce non negativity and unity sum in the spatial direction
        P_0[P_0 < 0] = 0
        P_sum, var_Psum = sum_across_trace(P_0, variance, extraction_range)
        P_0 = P_0/P_sum
        

        
        #Now, optimization step. This version makes no attempts to deal with bad pixel
        #print(flux_0.shape, P_0.shape)
        
        #optimized variance 
        variance_opt = (read_out_noise/gain)**2 + (flux_0*P_0 + background)/gain

        if plot: #what are relavent diagnostic plots from here? P image is one, the actual extraction range is another. 
            fig, ax = plt.subplots(3,1)
            ax[0].imshow(data - background, origin = 'lower')
            ax[0].plot([0,1],extraction_range, '-')
            ax[0].suptitle('Data - background')
            ax[1].imshow(variance_opt, origin = 'lower')
            ax[1].plot([0,1],extraction_range, '-')
            ax[1].suptitle('Optimized variance')
            ax[2].imshow(P_0, origin = 'lower')
            ax[2].plot([0,1],extraction_range, '-')
            ax[2].suptitle('Profile image')
            # for i in range(extraction_range[0], extraction_range[1]):
            #     plt.plot(P_0[i,:])
            #     #plt.plot(median_filter(P_0[i,:], 10))
            #     plt.ylim([-0.05,0.2])
            #     plt.xlim([30,140])
            #     plt.ylabel('P')
            #     plt.xlabel('Spectral pixel')
            # plt.title('Spectral Profile')
            # plt.show()

        #compare data to model and reject spurious pixels, this is 1 for good pixels
        if bad_pix_masking:
            Mask = (data - background - flux_0*P_0)**2 < sig_clip**2*variance_opt
            Mask = np.logical_and(Mask.astype(int), bad_pixel_mask) #make integer
        else:
            Mask = np.logical_and(np.ones(data.shape), bad_pixel_mask) #use all pixels, this will quit the while loop automatically

        #Summation terms
        #sum P(D-S)/V
        sum1, foo = sum_across_trace(Mask*P_0*(data-background)/variance_opt, variance, extraction_range) 
        #sum P
        sumP, foo = sum_across_trace(Mask*P_0,variance, extraction_range) 
        #sum P**2/V
        sum3, foo = sum_across_trace(Mask*P_0**2/variance_opt,variance, extraction_range) 
        
        #plt.plot(sum_across_trace(P_0*(data-background), variance, extraction_range)[0],'c')
        #plt.plot(sum_across_trace(variance_opt, variance, extraction_range)[0],'r')
        #plt.plot(sum1 ,'r')
        #plt.plot(sumP,'g')
        #plt.plot(sum3,'b')

        #compute optimized flux and variance spectra
        flux_opt_final = np.nan_to_num(sum1/sum3)
        variance_opt_final = np.nan_to_num(sumP/sum3 )

        #update flux_0 and var_0
        flux_0 = flux_opt_final
        var_0 = variance_opt_final
        #subtract the number of iteration by 1
        niter -= 1

        if np.all(Mask == Mask_old): #the set of rejected pixel converges
            if verbose:
                print("Bad pixel mask converge at niter = {}".format(niter))
            break #break from while loop immediately 
        else:
            Mask_old = Mask #update Mask_old, and reiterate
    
    #for debugging, show the mask at the end
    # plt.figure(figsize = (20,5))
    # plt.subplot(141)
    # plt.imshow(data-background, origin = 'lower')
    # plt.xlim([130,160])
    # plt.ylim([130,170])
    # #plt.colorbar()
    # plt.subplot(142)
    # plt.imshow(Mask.astype('int'), origin = 'lower')
    # plt.xlim([130,160])
    # plt.ylim([130,170])
    # #plt.colorbar()
    # plt.subplot(143)
    # plt.imshow(Mask.astype('int')*(data-background), origin = 'lower')
    # plt.xlim([130,160])
    # plt.ylim([130,170])
    # plt.subplot(144)
    # plt.imshow(P_0, origin = 'lower', vmin = 0, vmax = 0.5)
    # plt.xlim([130,160])
    # plt.ylim([130,170])
    # plt.colorbar()
    # plt.tight_layout()
    
    # plt.show()

    return flux_opt_final, variance_opt_final

# @profile
def spec_extraction(thumbnails, slit_num, filter_name = 'J', plot = True, output_name = None, sub_background=True, shift_dir = 'diagonal',
    bkg_sub_shift_size = 21, bkg_poly_order = 2, method = 'optimal_extraction', niter = 2, sig_clip = 5, bad_pix_masking = 0,skimage_order=4, width_scale=1., 
    diag_mask = False, trace_angle = None, fitfunction = 'Moffat', sum_method = 'weighted_sum', box_size = 1, poly_order = 4, mode = 'pol', 
    spatial_sigma = 3,verbose = True, DQ_thumbnails = None, use_DQ=True, debug_DQ=False,spatial_smooth=1,spectral_smooth=10,fractional_fit_type=None,
    plot_optimal_extraction = False, plot_findTrace = False):
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
    sub_background: None to not run background subtraction, 
                    'shift_and_subtract' to run a background subtraction routine by shift and subtracting
                    '2D_polynomial' to run 2D polynomial fitting to subtract backgrond. If this is selected, bkg_poly_order is the order 
                    of polynomial used
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
                        (iv) optimal_extraction: This is Horne 1986 optimal extraction method. 
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
    
    if DQ_thumbnails is not None:
        DQ_copy = copy.deepcopy(DQ_thumbnails)
    
    #Flip some of the traces around.
    if mode=='pol': 
        thumbnails_copy[1,:,:] = thumbnails_copy[1,-1::-1, -1::-1] #flip y, x. Bottom-right
        thumbnails_copy[2,:,:] = thumbnails_copy[2,:,-1::-1] #flip x #Top-right
        thumbnails_copy[3,:,:] = thumbnails_copy[3,-1::-1, :] #flip y #Bottom-left

        if DQ_thumbnails is not None:
            DQ_copy[1,:,:] = DQ_copy[1,-1::-1, -1::-1] #flip y, x. Bottom-right
            DQ_copy[2,:,:] = DQ_copy[2,:,-1::-1] #flip x #Top-right
            DQ_copy[3,:,:] = DQ_copy[3,-1::-1, :] #flip y #Bottom-left

        trace_titles=["Top-Left", "Bottom-Right", "Top-Right", "Bottom-left"]

    if mode=='spec':
        trace_titles=['Extracted Spectrum']

    #lists to collect widths and angles of the traces
    widths = []
    angles = []
    #if the given angle is only one, make it into a list
    try:
        len(trace_angle) #if pass then angle is a list
    except:
        trace_angle = [trace_angle]*ntraces #just repeat the given nubmer 4 times

    thumbnails_to_extract = [] #This is to collect the thumbnails that will actually be extracted (e.g. bkg subtracted/rotated)
    for j in range(ntraces):    
        trace_title = trace_titles[j]

        thumbnail = thumbnails_copy[j,:,:]
        if mode=='pol' and verbose:
            print("Extracting spectra from trace {} of 4".format(j))
        if mode=='spec' and verbose:
            print("Extracting spectrum".format(j))

        #Should we subtract the background? 
        #for first round, we have to do shift and subtract just to find the trace
        if sub_background != None and mode == 'pol':        
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

            if slit_num != 'slitless' or shift_dir == 'horizontal':
                bkg_stack = np.dstack((shift( thumbnail, [0,-bkg_sub_shift_size ], order = 0),shift( thumbnail, [0,bkg_sub_shift_size ], order = 0 )))

                bkg = np.nanmean(bkg_stack, axis=2)

                #if median filter background
                #bkg = median_filter(bkg, 3)

            elif shift_dir == 'vertical':
                bkg_stack = np.dstack((shift( thumbnail, [-bkg_sub_shift_size,0 ], order = 0),shift( thumbnail, [bkg_sub_shift_size ,0], order = 0)))
                bkg = np.nanmean(bkg_stack, axis=2)

            elif shift_dir =='diagonal': #for slitless data, shift in diagonal
                bkg_stack = np.dstack((shift( thumbnail, [-bkg_sub_shift_size,-bkg_sub_shift_size ], order = 0),\
                            shift( thumbnail, [bkg_sub_shift_size,bkg_sub_shift_size ], order = 0 )))
                bkg = np.nanmean(bkg_stack, axis=2)
                
         


            #if median filter background
            #bkg = median_filter(bkg, 3)

            bkg_sub = thumbnail - bkg

            #Mask out the area outside of the slit hole.
            # thumb_mask = makeDiagMask(len(bkg_sub[0]), slit_hole_diameter+3)
            # bkg_sub = bkg_sub * thumb_mask
            # bkg = bkg * thumb_mask
         
        elif sub_background != None and mode == 'spec':
            medval = np.median(thumbnail.flatten())
            bkg = np.ones(np.shape(thumbnail))*medval
            bkg_sub = thumbnail - bkg

        else: #if not background subtraction, set bkg to 0.
            bkg_sub = np.copy(thumbnail)
            bkg = thumbnail*0.

        ###############################
        #locate the trace in the data##
        ###############################
        #trace is the vector of the y location of the trace for each x location in the frame
        #width is the width of the trace at its brightest point. 
        start = time.time()            
        
        if trace_angle[j] is None:
            raw, trace, trace_width, measured_trace_angle = findTrace(bkg_sub, poly_order = 1, weighted=True, plot = plot_findTrace, diag_mask=diag_mask, mode=mode,
                                                                  fractional_fit_type=fractional_fit_type) #linear fit to the trace
            widths += [trace_width]
            angles += [measured_trace_angle]
        else:
            angles += [trace_angle[j]] #use the given angle
            raw, trace, trace_width, measured_trace_angle = findTrace(bkg_sub, poly_order = 1, weighted = True, plot = plot_findTrace, diag_mask = diag_mask, mode = mode,
                                                          fractional_fit_type = None) #for quickly getting trace width, which is needed to determine extraction range
            widths += [trace_width]
        
        #After findTrace is run, we can do 2D polynomial background subtraction
        if sub_background == '2D_polynomial':
            #update background frame with a 2D fitted background. 

            #first mask out the trace using results from findTrace
            if trace_angle is None:
                mask = make_mask_from_findTrace(trace, 3*trace_width, measured_trace_angle)
            else:
                mask = make_mask_from_findTrace(trace, 3*trace_width, trace_angle[j])

            #then run the 2d polynomial function, update bkg_sub and bkg
            del bkg
            del bkg_sub
            bkg_sub, bkg = image_utils.fit_background_2d_polynomial(thumbnail, mask, polynomial_order = bkg_poly_order)
           
        if diag_mask:
            mask = makeDiagMask(np.shape(bkg_sub)[0], 25)
            bkg_sub[~mask] = 0.

        if verbose:
            print("Trace width {}".format(trace_width))

        weight_width = trace_width*width_scale


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

            thumbnails_to_extract.append(bkg_sub) #add background subtracted image

        elif method == 'fit_across_trace':
            if verbose:
                print("trace angle is ", measured_trace_angle," deg")
            if trace_angle == None:
                rotate_spec_angle = measured_trace_angle #use the measured angle
            else:
                rotate_spec_angle = trace_angle[j] #use the given value
                if verbose:
                    print("use given ", trace_angle[j]," instead. change this by setting trace_angle to None")
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
            thumbnails_to_extract.append(bkg_sub) #add background subtracted image
        elif method == 'sum_across_trace':
            #First, determine the angle to rotate the spectrum, this can either be given or measured by findTrace
            if verbose:
                print("trace angle is ", measured_trace_angle," deg")
            if trace_angle == None: #if the trace angle is not given, use the measured angle
                rotate_spec_angle = measured_trace_angle 
            else: #otherwise, use the given value
                rotate_spec_angle = trace_angle[j] 
                if verbose:
                    print("using given angle of ", trace_angle[j]," deg. change this by setting trace_angle to None")

            #rotate the spectrum here. rotation axis is the middle of the image
            width_thumbnail = bkg_sub.shape[0]
            sub_rotated = frame_rotate(bkg_sub, rotate_spec_angle+180,cxy=[width_thumbnail/2,width_thumbnail/2])
            rotated = frame_rotate(thumbnail, rotate_spec_angle+180,cxy=[width_thumbnail/2,width_thumbnail/2])

            #determine the extraction range based on the width parameter
            #first, find the peak
            ext_range = determine_extraction_range(sub_rotated, trace_width/np.abs(np.cos(np.radians(rotate_spec_angle))), spatial_sigma = spatial_sigma)



            #call the optimal extraction method, remember it's sum_across_trace(bkg_sub_data, bkg, extraction_range, etc)
            spec_res, spec_var = sum_across_trace( sub_rotated, rotated , ext_range) 

            spectra.append(spec_res)
            spectra_std.append(np.sqrt(spec_var)) 
            thumbnails_to_extract.append(sub_rotated)#add background subtracted, rotated, image
             
        elif method == 'optimal_extraction':
            #First, determine the angle to rotate the spectrum, this can either be given or measured by findTrace
            if verbose:
                print("trace angle is ", measured_trace_angle," deg")
            if trace_angle[j] == None: #if the trace angle is not given, use the measured angle
                rotate_spec_angle = measured_trace_angle 
            else: #otherwise, use the given value
                rotate_spec_angle = trace_angle[j] 
                if verbose:
                    print("using given angle of ", trace_angle[j]," deg. change this by setting trace_angle to None")

            #rotate the spectrum here. rotation axis is the middle of the image
            width_thumbnail = bkg_sub.shape[0]
            sub_rotated = frame_rotate(bkg_sub, rotate_spec_angle+180,cxy=[width_thumbnail/2,width_thumbnail/2])
            rotated = frame_rotate(thumbnail, rotate_spec_angle+180,cxy=[width_thumbnail/2,width_thumbnail/2])

            if mode == 'spec':
                real_width = image_utils.traceWidth_after_rotation(sub_rotated)
                

            #plt.imshow(sub_rotated, origin = 'lower')
            #plt.show()

            if DQ_thumbnails is not None and use_DQ:
                DQ_rotated = frame_rotate(DQ_copy[j,:,:], rotate_spec_angle+180,cxy=[width_thumbnail/2,width_thumbnail/2])
                DQ_final = copy.deepcopy(DQ_rotated)
                DQ_final[DQ_rotated > 1e-2] = 0 #Note, in the definition below, bad pixels have a value of 0 and good pixels 1
                DQ_final[DQ_rotated <= 1e-2] = 1
            else:
                DQ_final = rotated*0 + 1

            if debug_DQ and DQ_thumbnails is not None:
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                plt.imshow(DQ_copy[j,:,:])
                ax2 = fig.add_subplot(122)
                plt.imshow(DQ_final)

            #determine the extraction range based on the width parameter
            #first, find the peak
            # spatial_profile = np.sum(sub_rotated, axis = 1) #sum in the spectral direction to get a net spatial profile
            # vert_max = np.argmax(spatial_profile) #locate the peak in this profile
            # #define lower and upper boundaries of the extraction area. Remember to multiply the trace_width with cos(rotation angle)
            # #because the pixel width changes as we rotate the image 
            # lower = int(np.floor(vert_max - spatial_sigma*trace_width/np.abs(np.cos(np.radians(rotate_spec_angle))))) #is this LISP or what?
            # upper = int(np.ceil(vert_max + spatial_sigma*trace_width/np.abs(np.cos(np.radians(rotate_spec_angle)))))
            
            bkg = rotated-sub_rotated
            
            if mode == 'spec':
                ext_range = determine_extraction_range(sub_rotated, real_width, spatial_sigma = spatial_sigma)
                widths += [real_width]
            else:
                ext_range = determine_extraction_range(sub_rotated, trace_width/np.abs(np.cos(np.radians(rotate_spec_angle))), spatial_sigma = spatial_sigma)
            

            #call the optimal extraction method, remember it's optimal_extraction(non_bkg_sub_data, bkg, extraction_range, etc)
            spec_res, spec_var = optimal_extraction(rotated, bkg, ext_range, bad_pix_masking = bad_pix_masking, \
                gain = 1.2, read_out_noise = 12, plot = plot_optimal_extraction, niter = niter, sig_clip = sig_clip, verbose = verbose, bad_pixel_mask=DQ_final,
                spatial_smooth=spatial_smooth,spectral_smooth=spectral_smooth) 

            spectra.append(spec_res)
            spectra_std.append(np.sqrt(spec_var)) 
            thumbnails_to_extract.append(sub_rotated)#add background subtracted, rotated, image

    
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

    return np.array(spectra), np.array(spectra_std), np.array(widths), np.array(angles), np.array(thumbnails_to_extract)  #res_spec is a dict, res_stddev and thumbnails are list

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

def rough_lambda_and_filter_calibration(spectra, widths, xpos, ypos, band = "J", off0 = 0.93, verbose=False, 
    plot_alignment=False, offset_method=2, tilt_angle = 45, source_compensation = False):
    #Note this function uses a bunch of things from constants.py

    if band == "J":
        lambda0 = J_lam
    elif band == "H":
        lambda0 = H_lam
    else:
        print("You have selected an unsupported bandpass '{}', returning.".format(band))
        return spectra


    pix_angle_at_pupil = np.radians(J_lam * angular_magnification/3600) #radians/pixel
    #What is the angle of the incident source
    source_angle_x = np.degrees( (xpos-1027)*pix_angle_at_pupil)
    source_angle_y = np.degrees( (ypos-1060)*pix_angle_at_pupil)

    total_tilt = 7
    tilt_angle = np.radians(tilt_angle)

    angles = [total_tilt*np.sin(tilt_angle),total_tilt*np.sin(tilt_angle),
              total_tilt*np.cos(tilt_angle),total_tilt*np.cos(tilt_angle)]

    if verbose: 
        print("Tilt angles = {}".format(angles))
    angles = [3.5,3.5,3.5,3.5]

    if source_compensation:
        angles += np.array([-source_angle_y-source_angle_x, -source_angle_y-source_angle_x,
                           -source_angle_y-source_angle_x,-source_angle_y-source_angle_x])/np.sqrt(2)


    # These values were determined by trial and error, not by because I know the orientation of 
    # things in the instrument
    angles = [3.5,3.5,3.5,3.5]
    signs = [1.,-1,1,-1]

    for i in range(4):

        #We use the below equations to correct for the filter throughput wavelength shift

        # d_beta/d_lambda = m/(D*cos beta)
        # beta = arcsin(m*lambda/D - sin(alpha))
        #
        # alpha = 3.5 - angle of incidence 
        # m = +/-1          - Grating Order
        # D = 17um          - grating period
        
        # -> d_lambda/d_beta =  D/M * cos beta

        #First, what is the expected angle of each trace leaving the PG (relative to the optical axis)
        beta = np.degrees( np.arcsin( signs[i]*lambda0/pg_period) - np.sin(np.radians(angles[i])))

        if verbose: 
            print("The diffraction angle for trace {} is {}".format(i,beta))

        # Now what is the angular dispersion: 
        dbdl = pg_period * np.cos(np.radians(beta)) #Micron/Radian

        #How big is one pixel at the pupil?
        pix_angle_at_pupil = np.radians(plate_scale*angular_magnification/3600) #radians/pixel

        # What is the d_lambda per pixel (i.e. the linear dispersion)
        ld = dbdl*pix_angle_at_pupil

        if verbose: 
            print("The linear dispersion for trace {} is {} um per pixel".format(i,ld))

        #Read in the filter profile
        filt_tp = np.genfromtxt(wircpol_dir+"wirc_drp/specification/J_WIRC.csv", delimiter=',')
        filt_tp[:,1] = filt_tp[:,1]/100 #The filter trasmission is given as percent. Let's normalize it to 1. 

        #The wavelength correction factor for the filter. Ghinassi et al. 2002 paper gives the below equations, with n_eff = 2 for MKO filters. 
        n_eff = 2. #This value is given by 
        factor = np.sqrt ( 1 - np.sin(np.radians(beta))**2/n_eff**2)
        filt_tp[:,0] = filt_tp[:,0]*(factor)

        if verbose: 
            print("The wavelength correction factor for the filter for trace {} is {}".format(i,factor))

        
        #TODO: Read the width parameters from the spectral extraction to get this value
        seeing = widths[i] #In pixels
        # seeing = 1.
        #Smooth by the seeing
        filt_tp[:,1] = gaussian_filter(filt_tp[:,1],seeing)
        #Let's do this twice because we'll do this to smooth out the data as well. 

        #Apply the linear dispersion calculated above to the raw spectra plus a fiducial offset
        off0 = 0.93
        spec_wl = ld*spectra[i,0,:] + off0
        spec_f  = spectra[i,1,:]
        # spec_f  = gaussian_filter(spectra[i,1,:],seeing)

        if offset_method == 1:
            ##OFFSET FINDING METHOD 1
            #We'll cross correlate the filter profile with the spectra to get a precise shift in wavelength. 

            #A range of possible offset values. It could be that this is too large. 
            offs = np.arange(-0.1,0.1,0.0001)

            #A list that will hold the correlation values
            corr = []
            
            for off in offs:
                #Shift the filter profile by the offset
                filt_i = np.interp(spec_wl+off,filt_tp[:,0],filt_tp[:,1])
                #Correlate with the spectrum
                corr.append(np.correlate(filt_i,spec_f))

            #Find the offset with the highest correlation
            best_off = offs[np.where(corr == np.max(corr))[0]]
        
        elif offset_method == 2:
            ##OFFSET FINDING METHOD 2

            # Find the center of the filter and the data by measuring the locations of the peak gradients on the blue and red
            # ends of the filter profile and spectra, and make sure that they're centered at the same place. 
            
            #First the filter
            wl_min = np.min(filt_tp[:,0])
            wl_max = np.max(filt_tp[:,0])
            wls = np.arange(wl_min, wl_max,0.001)
            filt_int = np.interp(wls,filt_tp[:,0],filt_tp[:,1])

            grad = np.gradient(filt_int)

            wl_min = np.min(filt_tp[:,0])
            wl_max = np.max(filt_tp[:,0])
            wls = np.arange(wl_min, wl_max,0.001)
            filt_int = np.interp(wls,filt_tp[:,0],filt_tp[:,1])

            grad = np.gradient(filt_int)
            #Can we fit two gaussians to this?
            gp = models.Gaussian1D(amplitude=np.nanmax(grad), mean=1.17, stddev=0.05)
            gm = models.Gaussian1D(amplitude=np.nanmin(grad), mean=1.32, stddev=0.05)

            low_inds = np.where( wls <= np.mean(wls))[0]
            fit_g = fitting.LevMarLSQFitter()
            gp_fit = fit_g(gp, wls[low_inds], grad[low_inds])
            # print(gp_fit.mean)

            high_inds = np.where(wls > np.mean(wls))[0]
            fit_g = fitting.LevMarLSQFitter()
            gm_fit = fit_g(gm, wls[high_inds], grad[high_inds])
            # print(gm_fit.mean)

            filt_grad_center = (gp_fit.mean + gm_fit.mean)/2.
            # print(grad_center)

            #Now the spectrum
            spec_wls = ld*spectra[i,0,:]+off0
            inds = np.where( ( spec_wls > 1.1) & (spec_wls < 1.35))
            spec_wls = spec_wls[inds]
            spec1_int = spectra[i,1,:][inds]
            spec_grad = np.gradient(spec1_int)

            #Can we fit two gaussians to this?
            gp = models.Gaussian1D(amplitude=np.nanmax(spec_grad), mean=1.17, stddev=0.05)
            gm = models.Gaussian1D(amplitude=np.nanmin(spec_grad), mean=1.32, stddev=0.05)

            low_inds = np.where( spec_wls <= np.mean(spec_wls))[0]
            fit_g = fitting.LevMarLSQFitter()
            gp_fit = fit_g(gp, spec_wls[low_inds], spec_grad[low_inds])
            # print(gp_fit.mean)

            high_inds = np.where(spec_wls > np.mean(spec_wls))[0]
            fit_g = fitting.LevMarLSQFitter()
            gm_fit = fit_g(gm, spec_wls[high_inds], spec_grad[high_inds])
            # print(gm_fit.mean)

            spec_grad_center = (gp_fit.mean + gm_fit.mean)/2.
            # print(grad_center)

            best_off = filt_grad_center - spec_grad_center
        else: 
            print("offset_method keyword not recognized, your wavelength solution is probably bogus")


        if verbose:
            print("Best offset for spectrum {} = {}".format(i,best_off))

        #Plot the filter profile with the shifted spectrum
        if plot_alignment:
            plt.figure(figsize=(10,5))
            plt.plot(filt_tp[:,0],filt_tp[:,1]*10000)
            plt.plot(spec_wl + best_off,spec_f)
            plt.xlim(1.15,1.35)
        
        #Apply the full wavelength solution with the best found offset
        spectra[i,0,:] = ld*spectra[i,0,:] + best_off + off0
        
        #Interpolate the filter values at the wavelengths of the spectrum
        filt = np.interp(spectra[i,0,:],filt_tp[:,0],filt_tp[:,1])
        #Divide through by the filter throughput
        spectra[i,1,:] = spectra[i,1,:]/filt
        # spectra[i,1,:] = gaussian_filter(spectra[i,1,:],seeing)/filt



    return spectra



def align_set_of_traces(traces_cube, ref_trace):
    """
    align_set_of_traces takes a cube of traces with dimension (number_of_traces, 3(wl, flux, flux_err), length_of_each_trace) 
    and align them with respect to the reference trace of the same length. 
    """
    new_cube = np.zeros(traces_cube.shape)
    ref = ref_trace
    #fig, (ax, ax2) = plt.subplots(2,4, figsize = (20,10))
    for i, j in enumerate(traces_cube):
        #print(np.max(ref), np.max(j))
        #corr = fftconvolve(ref/np.nanmax(ref), (j/np.nanmax(j))[::-1] )
        corr = fftconvolve(np.nan_to_num(ref/np.nanmax(ref)), np.nan_to_num((j/np.nanmax(j))[::-1]) )#j[1,:] is the flux vector

        shift_size = np.nanargmax(corr) - len(ref) +1
        #print(shift_size)
        new_cube[i] = shift(traces_cube[i], shift_size, order = 0) # this shifts wl, flux, and flux_error at the same time. order = 0 so no interpolation 
            
    return new_cube

def align_spectral_cube_helper(traces_cube, ref_trace):
    """
    align_set_of_traces takes a cube of traces with dimension (number_of_traces, 3(wl, flux, flux_err), length_of_each_trace) 
    and align them with respect to the reference trace of the same length. 
    """
    new_cube = np.zeros(traces_cube.shape)
    ref = ref_trace
    #fig, (ax, ax2) = plt.subplots(2,4, figsize = (20,10))
    for i, j in enumerate(traces_cube):
        #print(np.max(ref), np.max(j))
        #corr = fftconvolve(ref/np.nanmax(ref), (j/np.nanmax(j))[::-1] )
        corr = fftconvolve(np.nan_to_num(ref/np.nanmax(ref)), np.nan_to_num((j[1,:]/np.nanmax(j[1,:]))[::-1]) )#j[1,:] is the flux vector

        shift_size = np.nanargmax(corr) - len(ref) +1
        #print(shift_size)
        new_cube[i] = shift(traces_cube[i], (0,shift_size), order = 0) # this shifts wl, flux, and flux_error at the same time. order = 0 so no interpolation 
            
    return new_cube

def align_spectral_cube(spectral_cube, ref_trace = None):
    """
    A higher level function to run align_set_of_traces. This function takes a wirc+pol cube with dimensions
    (number of observations, 4 spectral traces, 3(wavelength, flux, flux_err), number of spectral pixels)
    align each flux array with the ref_trace (1d reference flux array), then return an aligned cube.

    if ref trace is not given, default to using a Qp trace from the first observation
    """
    #Define reference if not given,
    if ref_trace == None:
        ref_trace = spectral_cube[0,0,1,:] #This is first observation, first trace (Qp), flux, and the whole vector
    #create a destination array
    aligned_cube = np.zeros(spectral_cube.shape) 
    #loop through 4 spectral traces
    for i in range(spectral_cube.shape[1]): #this dimension is the 4 spectral traces for wirc+pol
        aligned_cube[:,i,:,:] = align_spectral_cube_helper(spectral_cube[:,i,:,:], ref_trace)

    return aligned_cube

def scale_and_combine_spectra(spec_cube, return_scaled_cube = False, xmin = 0, xmax = -1):
    """
    scale_and_combine_spectra takes a spectra cube from get_spec_cube_from_wirc_obj. For each trace 
    (Qp, Qm, Up, Um) in the series, scale each observation by the total observed flux, then
    median combine them. 
    xmin and xmax specify where the index range we will compute total flux from
    If return_scaled_cube == True, then return the scaled cube instead of the 4 median combined spectra
    """

    #spec_cube is 4 dimensions: spec_cube.shape = (num_images, 4_traces, 3[wavelength, flux, flux_error], number_of_spectral_pixel)

    #for each quadrant (Qp, Qm, Up, Um), scale the spectrum so that the total flux match that of the median spectrum
    med_specs = np.median(spec_cube, axis = 0)[:,1,:] #this is the 4 median spectra
    #print(med_specs.shape)
    total_flux = np.sum(med_specs[:,xmin:xmax], axis = 1) #4 total median fluxes, in the range specified by xmin, xmax
    #print(total_flux.shape)

    scaled_specs = np.copy(spec_cube)

    #for each observation, normalize
    for i in range(spec_cube.shape[0]):
        four_specs = spec_cube[i,:,1,:]
        #print(four_specs.shape)
        four_errors = spec_cube[i,:,2,:]
        #scaling factor
        scale_factor = total_flux/np.sum(median_filter(four_specs[:,xmin:xmax],size = (1,5) ), axis = 1) #median filter to remove contributions from noisy wings
        #all four traces should be scaled equally
        #take mean from the factors measured from the four traces, and apply that mean value to the 4 traces.
        scale_factor = np.array([np.mean(scale_factor)]*4)
        #print(scale_factor)
        
        #apply scale factor to the four_specs and four_errors, then put them into scaled_specs
        scaled_specs[i,:,1,:] = np.einsum('i,ij->ij',scale_factor,four_specs ) 
        scaled_specs[i,:,2,:] = np.einsum('i,ij->ij',scale_factor,four_errors)  

    # #now align the spectra in the wavelength direction by shifting: 
    #this should already be done by align set of traces
    # scaled_specs[:,0,1,]

    #return results
    if return_scaled_cube:
        return scaled_specs
    else:
        return np.median(scaled_specs, axis = 0)



def get_spec_cube_from_wirc_obj( wirc_objs, source_num = 0):
    """
    extract a cube of spectra from a list of wirc objects with the given source number.
    the output format is a 4 dimensions numpy array with (number_of_images, 4_traces[Qp,Qm,Up,Um], 
        3[wavelength, flux, flux error], number_of_spectral_pixel)
    """
    #get spectra out of the wirc_object into a cube
    spec_cube = []
    for j,i in enumerate(wirc_objs):
        try:
            spec_cube += i.source_list[source_num].trace_spectra
        except:
            print("wirc object number {} doesn't have source number {}".format(j, source_num))
    spec_cube = np.array(spec_cube)
    return spec_cube





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

#########################################################
################To be deprecated#########################
#########################################################
def align_spectra(spectra, lowcut=0, highcut=-1, big_filt_sz = 30, little_filt_sz = 3, x_start = [0.,1.,0.,0.,0.]):
    '''
    Aligns the spectra by minimizing a function that applies a 2nd order wavelenth shift and scales the flux. 

    WARNING: This assumes there is no polarization signal. IF there is a polarization it may mess with it, but I'm not sure
    Inputs: 
        spectra - An array with shape [4,3,n], where n is the number of elements in the spectra
    '''

    #First do the Qm trace:
    
    #Let's remove the low frequency parts of the spectrum:
    specq = copy.deepcopy(spectra[0:2,:,lowcut:highcut])
    # spec2 = copy.deepcopy(spectra[1,:,lowcut:highcut])

    specq[0,1,:] -= median_filter(specq[0,1,:],big_filt_sz)
    specq[1,1,:] -= median_filter(specq[1,1,:],big_filt_sz)
    
    #Then we'll do a 1" median filter to smooth out some of the noise:
    
    specq[0,1,:] = median_filter(specq[0,1,:],little_filt_sz)
    specq[1,1,:] = median_filter(specq[1,1,:],little_filt_sz)
    
    #Now run the alignment
    # x_start = [1.,0.,1.,0.,0.]
    new_qm_params = least_squares(fit_to_align_spectra,x_start,args=[specq])
    new_qm_flux = function_to_align_spectra(new_qm_params.x,spectra[0:2,:,:])

    spectra[1,0,:] = spectra[0,0,:]
    spectra[1,1,:] = new_qm_flux

    
    #Now do the Um trace:

        #Let's remove the low frequency parts of the spectrum:
    specu = copy.deepcopy(spectra[2:,:,lowcut:highcut])
    # spec2 = copy.deepcopy(spectra[1,:,lowcut:highcut])

    specu[0,1,:] -= median_filter(specu[0,1,:],big_filt_sz)
    specu[1,1,:] -= median_filter(specu[1,1,:],big_filt_sz)
    
    #Then we'll do a 1" median filter to smooth out some of the noise:
    
    specu[0,1,:] = median_filter(specu[0,1,:],little_filt_sz)
    specu[1,1,:] = median_filter(specu[1,1,:],little_filt_sz)
    

    #We'll start with the output params from the qm
    x_start = [1.,0.,1.,0.,0.]
    new_um_params = least_squares(fit_to_align_spectra,x_start,args=[specu])
    new_um_flux = function_to_align_spectra(new_um_params.x,spectra[2:,:,:])
    spectra[3,0,:] = spectra[2,0,:]
    spectra[3,1,:] = new_um_flux

    return spectra


def fit_to_align_spectra(x,two_spectra):
    '''
    The function that returns the difference between two spectra after applying a 2nd order wavelenth shift and scales the flux. 
    '''

    #Calculate the new spectrum
    new_spectra = function_to_align_spectra(x,two_spectra)

    #The scaling factor is used for the fitting, but not when we apply the new wavelength solution
    return np.std(x[0]*new_spectra - two_spectra[0,1,:])

def function_to_align_spectra(x,two_spectra):
    '''
    A function that applies a wavelength solution (in x) to the second spectrum in two_spectra
    '''
    
    old_lambda = two_spectra[1,0,:] 

    #Apply the wavelength solution
    new_lambda = x[1]+x[2]*old_lambda+x[3]*old_lambda**2+x[4]*old_lambda**3

    #Interpolate the flux values to the wavelenths of the first spectrum, 
    # using the newly calibrated wavelenths from the second spectrum
    out = np.interp(two_spectra[0,0,:],new_lambda,two_spectra[1,1,:])

    return out




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

#########################################################
################To be deprecated#########################
#########################################################

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


    # Qp_out = []
    # Qp_var_out = []
    # Qm_out = []
    # Qm_var_out = []
    # Up_out = []
    # Up_var_out = []
    # Um_out = []
    # Um_var_out = []
    # Q_out = []
    # Q_var_out = []
    # U_out = []
    # U_var_out = []
    # P_out = []
    # P_var_out = []
    # Theta_out = []
    # Theta_var_out = []

    # Qp = trace_spectra[0,1,:]
    Qp_var = trace_spectra[0,2,:]
    # Qm = trace_spectra[1,1,:]
    Qm_var = trace_spectra[1,2,:]
    # Up = trace_spectra[2,1,:]
    Up_var = trace_spectra[2,2,:]
    # Um = trace_spectra[3,1,:]
    Um_var = trace_spectra[3,2,:]
    
    # #Clip the heads and tails of spectra. Magic numbers for now! 
    # #TODO: This whole process needs to be refined. 
    # Qp = Qp[cutmin:cutmax]
    # Qp_var = Qp_var[cutmin:cutmax]
    # Qm = Qm[cutmin:cutmax]
    # Qm_var = Qm_var[cutmin:cutmax]
    
    # Up = Up[cutmin:cutmax]
    # Up_var = Up_var[cutmin:cutmax]
    # Um = Um[cutmin:cutmax]
    # Um_var = Um_var[cutmin:cutmax]
    
    wlQp = trace_spectra[0,0,:]
    wlQm = trace_spectra[1,0,:]
    wlUp = trace_spectra[2,0,:]
    wlUm = trace_spectra[3,0,:]

    Qp = trace_spectra[0,1,:]
    Qm = np.interp(wlQp,wlQm,trace_spectra[1,1,:])
    Up = trace_spectra[2,1,:]
    Um = np.interp(wlUp,wlUm,trace_spectra[3,1,:])

    q = (Qp-Qm)/(Qp+Qm)
    u = (Up-Um)/(Up+Um)

    return wlQp, q, q*0., wlUp, u, u*0.

    #TODO
        #The variances might be different now that we're shifted things around. Also from rough_lambda_and_filter_calibration
    
    # q, dq, dsq = compute_stokes_from_traces(Qp, np.sqrt(Qp_var), Qm, np.sqrt(Qm_var), plotted = False)
    # u, du, dsu = compute_stokes_from_traces(Up, np.sqrt(Up_var), Um, np.sqrt(Um_var), plotted = False)
    
    #shift traces so aligned traces can be returned
    # Qm = shift(Qm, dsq)
    # Qm_var = shift(Qm_var, dsq)

    # Um = shift(Um, dsu)
    # Um_var = shift(Um_var, dsu)        
    
    return wlQp, q, dq, wlUp, u, du


def make_scale_widget(Qp, Qm, x0):
    """
    make a fun widget to adjust scaling parameters and see the resulting q/u
    you can slide to adjust shift, scale, square root and quadrature terms.
    Say if the original Qm (or Um) spectrum is Qm(x), this resamples the spectrum
    as Qm(shift+scale*x+quad*x**2+square_root*np.sqrt(x))
    
    This does not work in the inline mode. 
    
    based on https://matplotlib.org/examples/widgets/slider_demo.html

    Input: plus and minus traces
    """
    #interpolate Qm 
    x = np.arange(len(Qm))
    Qm_int = interp1d(x, Qm, bounds_error = False)
    #slider to try setting scale
    from matplotlib.widgets import Slider, Button, RadioButtons

    fig, ax = plt.subplots(2,1,figsize = (6,10))

    #first draw initial plots
    t = np.arange(len(Qp))
    sh0 = -2
    sc0 = 1.05
    sq0 = 0.5
    qu0  = 0

    #s = Qm_int(sq0*np.sqrt(x)+sc0*x+sh0+(1-sc0)*x0-sq0*np.sqrt(x0) +qu0*x**2 - qu0*x0**2)
    s = Qm_int(sq0*np.sqrt(x)+sc0*x+sh0+(1-sc0)*x0-sq0*np.sqrt(x0) +qu0*(x- x0)**2 )
    ax[0].plot(t, Qp*0.95, color = 'blue')
    l, = ax[0].plot(t, s, color='red')
    ax[0].set_xlim([minlim,maxlim])
    ax[0].set_ylim([7000,14000])
    title = fig.suptitle("${:.3g}+{:.3g}x+{:.3g}sq(x)+{:.3g}(x-x0)^2 + (1-{:.3g})x_0 - {:.3g}sq(x_0)$".format(sh0,sc0,sq0,qu0,sc0,sq0))
    #title = fig.suptitle(r"$%f+%fx+%f\sqrt{x}+%f(x-x0)^2 + (1-%f)x_0 - %f\sqrt{x_0}$"%(sh0,sc0,sq0,qu0,sc0,sq0))
    #title = fig.suptitle("$\sqrt{x}$")

    plt.subplots_adjust(left=0.25, bottom=0.4)

    #qu plot

    m, = ax[1].plot( t,(Qp - s)/(Qp + s))
    ax[1].set_xlim([minlim,maxlim])
    ax[1].set_ylim([0,0.1])

    #s, = plt.plot(t, Qp, color='blue')
    #plt.axis([0, 1, -10, 10])

    #The sliders

    axcolor = 'lightgoldenrodyellow'
    axsh = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    axsc = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    axsq = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    axqu = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor=axcolor)
    axx0 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    # axsh = plt.axes([0.25, 0.4, 0.65, 0.03], facecolor=axcolor)
    # axsc = plt.axes([0.25, 0.5, 0.65, 0.03], facecolor=axcolor)
    # axsq = plt.axes([0.25, 0.6, 0.65, 0.03], facecolor=axcolor)

    ssh = Slider(axsh, 'Shift', -10, 10, valinit=sh0)
    ssc = Slider(axsc, 'Scale', 0.7, 1.5, valinit=sc0)
    ssq = Slider(axsq, 'Sqrt', -1.5, 1.5, valinit=sq0)
    squ = Slider(axqu, 'Square', 0.,0.001, valinit=qu0)
    sx0 = Slider(axx0, '$x_0$', 100,200, valinit=x0)


    def update(val):
        #amp = samp.val
        #freq = sfreq.val
        sh = ssh.val
        sc = ssc.val
        sq = ssq.val
        qu = squ.val
        new_x0 = sx0.val

        #s_up = Qm_int(sq*np.sqrt(x)+sc*x+sh+(1-sc)*x0-sq*np.sqrt(x0)+qu*x**2 - qu*x0**2) 
        l.set_ydata(Qm_int(sq*np.sqrt(x)+sc*x+sh+(1-sc)*new_x0-sq*np.sqrt(new_x0)+qu*x**2 - qu*new_x0**2) )
        m.set_ydata((Qp - Qm_int(sq*np.sqrt(x)+sc*x+sh+(1-sc)*new_x0-sq*np.sqrt(new_x0)+qu*x**2 - qu*new_x0**2))/
                        (Qp+Qm_int(sq*np.sqrt(x)+sc*x+sh+(1-sc)*new_x0-sq*np.sqrt(new_x0)+qu*x**2 - qu*new_x0**2)) )
        
        title.set_text("${:.3g}+{:.3g}x+{:.3g}sq(x)+{:.3g}(x-x0)^2 + (1-{:.3g})x_0 - {:.3g}sq(x_0)$".format(sh,sc,sq,qu,sc,sq))
        fig.canvas.draw_idle()

    ssh.on_changed(update)
    ssc.on_changed(update)
    ssq.on_changed(update)
    squ.on_changed(update)
    sx0.on_changed(update)

    #The reset button

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        ssh.reset()
        ssc.reset()
        ssq.reset()
        squ.reset()
        sx0.reset()
    button.on_clicked(reset)

    #rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    #radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


    #def colorfunc(label):
    #    l.set_color(label)
    #    fig.canvas.draw_idle()
    #radio.on_clicked(colorfunc)
    plt.show()
    
def broadband_aperture_photometry(thumbnails, width_scale = 5, source_offsets = (0,0), sky_offsets = (50,0), mode = 'pol', 
            filter_name = "J", plot = False, ron = 12, gain = 1.2, DQ_thumbnails = None, verbose = False,
            bkg_method = "aper"):
    """
    This function uses photutils package to define rectangular apertures over spectral traces 
    and compute aperture sums. 
    """
    if filter_name == "J":
        trace_length = 130
        template_width= 75
    elif filter_name == "H":
        trace_length = 260
        template_width = 120 #test this
    # Define lists to collect results.
    phot = [] #This vector collects extracted spectra from 4 traces
    phot_sky = []

    if mode=='pol':
        ntraces = 4 #how many traces? 4 for WIRC-POL
        if plot:
            fig = plt.figure(figsize=(11,4))
        
    if mode=='spec':
        ntraces = 1 #how many traces? 1 for WIRC-SPEC
        if plot:
            fig = plt.figure(figsize=(11,4))

    thumbnails_copy = copy.deepcopy(thumbnails)
    

    if DQ_thumbnails is not None:
        DQ_copy = copy.deepcopy(DQ_thumbnails)

    
    #Flip some of the traces around.
    if mode=='pol': 
        thumbnails_copy[1,:,:] = thumbnails_copy[1,-1::-1, -1::-1] #flip y, x. Bottom-right
        thumbnails_copy[2,:,:] = thumbnails_copy[2,:,-1::-1] #flip x #Top-right
        thumbnails_copy[3,:,:] = thumbnails_copy[3,-1::-1, :] #flip y #Bottom-left

        if DQ_thumbnails is not None:
            DQ_copy[1,:,:] = DQ_copy[1,-1::-1, -1::-1] #flip y, x. Bottom-right
            DQ_copy[2,:,:] = DQ_copy[2,:,-1::-1] #flip x #Top-right
            DQ_copy[3,:,:] = DQ_copy[3,-1::-1, :] #flip y #Bottom-left

        trace_titles=["Top-Left", "Bottom-Right", "Top-Right", "Bottom-left"]

    if mode=='spec':
        trace_titles=['Extracted Spectrum']

    #lists to collect widths and angles of the traces
    widths = []
    angles = []
    thumbnails_to_extract = [] #This is to collect the thumbnails that will actually be extracted (e.g. bkg subtracted/rotated)
    for j in range(ntraces):    
        trace_title = trace_titles[j]

        thumbnail = thumbnails_copy[j,:,:]
        if mode=='pol' and verbose:
            print("Extracting spectra from trace {} of 4".format(j))
        if mode=='spec' and verbose:
            print("Extracting spectrum".format(j))

        #subtract background for source finder

        bkg_sub_shift_size = 30 #doesn't matter...

        if False:        
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

            # shift_dir = 'diagonal'

            # if slit_num != 'slitless' or shift_dir == 'horizontal':
            #     bkg_stack = np.dstack((shift( thumbnail, [0,-bkg_sub_shift_size ]),shift( thumbnail, [0,bkg_sub_shift_size ] )))

            #     bkg = np.nanmean(bkg_stack, axis=2)

            # elif shift_dir == 'vertical':
            #     bkg_stack = np.dstack((shift( thumbnail, [-bkg_sub_shift_size,0 ]),shift( thumbnail, [bkg_sub_shift_size ,0] )))
            #     bkg = np.nanmean(bkg_stack, axis=2)
            # elif shift_dir =='diagonal': #for slitless data, shift in diagonal
            bkg_stack = np.dstack((shift( thumbnail, [-bkg_sub_shift_size,-bkg_sub_shift_size ]),shift( thumbnail, [bkg_sub_shift_size,bkg_sub_shift_size ] )))
            bkg = np.nanmean(bkg_stack, axis=2)
            #else:
            #    print('')


            bkg_sub = thumbnail - bkg

            #Mask out the area outside of the slit hole.
            # thumb_mask = makeDiagMask(len(bkg_sub[0]), slit_hole_diameter+3)
            # bkg_sub = bkg_sub * thumb_mask
            # bkg = bkg * thumb_mask

        else: #if not background subtraction, set bkg to 0.
            bkg_sub = np.copy(thumbnail)
            bkg = thumbnail*0.

        ###############################
        #locate the trace in the data##
        ###############################
        #trace is the vector of the y location of the trace for each x location in the frame
        #width is the width of the trace at its brightest point. 

        raw, trace, trace_width, measured_trace_angle = findTrace(bkg_sub, poly_order = 1, weighted=True, plot = 0, diag_mask=0, mode=mode) #linear fit to the trace

        #if background subtraction type is fit_background, then call the function

        diag_mask = 0
        if diag_mask:
            mask = makeDiagMask(np.shape(bkg_sub)[0], 25)
            bkg_sub[~mask] = 0.

        if verbose:
            print("Trace width {}".format(trace_width))

        weight_width = trace_width*width_scale

        widths += [trace_width]
        angles += [measured_trace_angle]
        
        #raw, trace, trace_width, measured_trace_angle = findTrace(bkg_sub, poly_order = 1, weighted=True, plot = 0, diag_mask=diag_mask,mode=mode) #linear fit to the trace
        #weight_width = trace_width*width_scale
        #if verbose:
        #    print("Trace width {}".format(trace_width))

        #find x location center of the trace
        x_loc = image_utils.trace_location_along_x(thumbnail, measured_trace_angle,plot = 0, template_width = template_width)

        ######################################
        ######Define Aperture#################
        ######################################
        phot_aper = RectangularAperture( (x_loc+source_offsets[0],trace[x_loc]+source_offsets[1]), \
           trace_length, width_scale*trace_width, np.radians(measured_trace_angle)) 

        #phot_aper = RectangularAperture( (int(len(trace)/2)+source_offsets[0],trace[int(len(trace)/2)]+source_offsets[1]), \
        #    trace_length, width_scale*trace_width, np.radians(measured_trace_angle)) 
            #trace center default to the center of the cutout in x, and the measured y position there, plus any given source offsets (x,y)
            #length is 130 for J, 260 for H
            #width is width_scale*measured trace_width and the angle is the measured angle

        sky_aper = RectangularAperture( (x_loc + sky_offsets[0],trace[x_loc]+sky_offsets[1]), \
            trace_length, width_scale*trace_width, np.radians(measured_trace_angle)) 
        # sky_aper = RectangularAperture( (int(len(trace)/2)+sky_offsets[0],trace[int(len(trace)/2)]+sky_offsets[1]), \
        #     trace_length, width_scale*trace_width, np.radians(measured_trace_angle)) 

        #show diagnosis plot if selected.
        if plot:
            fig,ax  = plt.subplots(1,1,figsize = (5,5))
            ax.imshow(thumbnail, origin = 'lower')
            phot_aper.plot(ax = ax, color = 'b')
            sky_aper.plot(ax = ax, color = 'r')
            plt.show()

        if bkg_method == 'median_mask':
            mask = make_source_mask(thumbnail,snr=3,npixels=5,dilate_size=11)
            mean,median,std = sigma_clipped_stats(thumbnail,sigma=3.0,mask=mask)
            thumbnail = thumbnail - median

        #Sum the aperture and output
        phot += [aperture_photometry(thumbnail, phot_aper)['aperture_sum'][0]]
        phot_sky  += [aperture_photometry(thumbnail, sky_aper)['aperture_sum'][0]]

    #for 4 traces
    phot = np.array(phot)
    phot_sky = np.array(phot_sky)

    #source_flux is phot-sky
    source_flux = phot - phot_sky

    #source std is sqrt(phot + phot_sky + read_noise**2) 
    source_std = np.sqrt(phot*gain + phot_sky*gain + ron**2)/gain #remember that poisson noise is sqrt(N in electrons)

    return source_flux, source_std


