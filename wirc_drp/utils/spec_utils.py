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
from scipy.ndimage import shift
from scipy import ndimage as ndi

from skimage.measure import profile_line

from astropy.modeling import models
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits as f

#From other packages
from wirc_drp.utils.image_utils import locationInIm, shift_and_subtract_background, fit_and_subtract_background, findTrace
from wirc_drp.masks.wircpol_masks import *

#debugging
import pdb

#path
import sys, os, time

#import pyklip.klip

wircpol_dir = os.environ['WIRC_DRP']

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
    
    # x = range(width-2*buffer)
    
    # #Gather values along trace
    # spec = []
    # loc = []
    # #Looping through columns    
    # for i in x:
    #     #if fit_result[i][1].amplitude.value == 0:
    #         #print(i,' is background')
    #     if fit_result[i][1].amplitude.value > 0: #If gaussian component is non-zero (Trace there)
    #         #Generate 2D gaussian centering at this location 
    #         gauss_filter = models.Gaussian2D(amplitude = 1, y_mean = width-buffer-i, \
    #                                             x_mean = i + fit_result[i][1].mean.value, \
    #                                             x_stddev = fit_result[i][1].stddev.value, \
    #                                             y_stddev = fit_result[i][1].stddev.value)
    #         xx,yy = np.mgrid[0:np.shape(cutout)[0], 0:np.shape(cutout)[1]]
    #         #plt.subplot(131)
    #         #plt.imshow(gauss_filter(xx,yy),origin = 'lower')
    #         #plt.subplot(132)
    #         #plt.imshow(cutout-bkg,origin = 'lower')
    #         #plt.subplot(133)
    #         #plt.imshow(gauss_filter(xx,yy)*(cutout-bkg),origin = 'lower')
    #         #plt.show()
    #         flux = np.sum((cutout-bkg)*gauss_filter(xx,yy))
    #         spec += [ flux]
    #         loc += [[width-buffer-i, i + fit_result[i][1].mean.value]]
    #         #plt.show()
    # return np.array(loc), np.array(spec)

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

            spec += [np.sum(weight * cutout)]
            var += [np.sum(weight * (cutout/gain + (ron/gain)**2))] #variance assuming readout noise and photon noise

    return np.array(spec[::-1]), np.array(var[::-1]) #flip so long wavelenght is to the right

def spec_extraction(thumbnails, slit_num, filter_name = 'J', plot = True, output_name = None, sub_background=True, method = ['skimage']):
    """
    This is the main function to perform spectral extraction on the spectral image
    given a set of thumbnails.
    """
    # Define lists to collect results.

    wavelengths = []
    spectra = [] #This vector collects extracted spectra from 4 traces
    spectra_std = []

    ntraces = 4 #how many traces? 4 for WIRC-POL

    if filter_name == 'J':
        cutout_size = 60 #Make cutout of each trace. This has to chage for J/H bands
    elif filter_name == 'H':
        cutout_size = 150

    if plot:
        fig = plt.figure(figsize=(11,4))

    
    #Flip some of the traces around. 
    thumbnails[1,:,:] = thumbnails[1,-1::-1, -1::-1] #flip y, x. Bottom-right
    thumbnails[2,:,:] = thumbnails[2,:,-1::-1] #flip x #Top-right
    thumbnails[3,:,:] = thumbnails[3,-1::-1, :] #flip y #Bottom-left

    trace_titles=["Top-Left", "Bottom-Right", "Top-Right", "Bottom-left"]
    
    for j in range(ntraces):    
        trace_title = trace_titles[j]

        thumbnail = thumbnails[j,:,:]
        print("Extracting spectra from trace {} of 4".format(j))

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

	    #For now, do shift and subtract
            bkg = (shift( thumbnail, [0,-21] ) + shift( thumbnail, [0,21] ))/2
            bkg_sub = thumbnail - bkg

                #Mask out the area outside of the slit hole.
            #thumb_mask = makeDiagMask(len(bkg_sub[0]), slit_hole_diameter+3)
            #bkg_sub = bkg_sub * thumb_mask
            #bkg = bkg * thumb_mask


        else: 
            bkg_sub = np.copy(thumbnail)
            bkg = thumbnail*0.

        ###############################
        #locate the trace in the data##
        ###############################
        #trace is the vector of the y location of the trace for each x location in the frame
        #width is the width of the trace at its brightest point. 
        start = time.time()            
        raw, trace, width = findTrace(bkg_sub, poly_order = 1, weighted=True, plot = False) #linear fit to the trace          

        ######################################
        ######Call spectral extraction routine
        ######################################
                
        ##Sketchy fit across trace

        ##skimage profile_line trying different interpolation orders
        if 'skimage' in method:
	    print("Extraction by skimage")
            linewidth = 20 #This should be adjusted based on fitted seeing.
            #for ii in range(6):
            #only do order = 4
	    for ii in [4]:
                #print(ii)
                spec_res = profile_line(bkg_sub, (0,trace[0]), (len(bkg_sub[1]),trace[-1]), linewidth = linewidth,order =  ii)                
                spectra.append(np.array(spec_res))
                spectra_std.append((gain*spec_res+linewidth * sigma_ron**2)/gain**2) #poisson + readout
        
        if 'weightedSum' in method:
            #define PSF (Gaussian for now)
            psf = np.zeros((21,21))
            xx,yy = np.mgrid[0:np.shape(psf)[0], 0:np.shape(psf)[1]]
            psf = models.Gaussian2D(amplitude = 1, y_mean = psf.shape[0]//2, x_mean = psf.shape[1]//2, \
                                   y_stddev = 2, x_stddev = 2)(yy,xx)
            #extract
            spec_res, spec_var = weighted_sum_extraction(bkg_sub, trace, psf)
            spectra.append(spec_res)
            spectra_std.append(np.sqrt(spec_var))
        
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

    min_len = min(len(spectra[0]),len(spectra[1]),len(spectra[2]),len(spectra[3]))
    for i in range(4):
        spectra[i] = spectra[i][0:min_len]
        spectra_std[i] = spectra_std[i][0:min_len]
    return np.array(spectra), np.array(spectra_std) #res_spec is a dict, res_stddev and thumbnails are list

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
def rough_wavelength_calibration_v2(trace, filter_name):
    """
    """
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
    return slope*(x - np.argmax(grad)) + wl_up

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
    
