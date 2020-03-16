# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 16:15:52 2016

@author: Kaew Tinyanont & Sizheng Ma
image_simulation
These are functions used to simulate an image based on a given spectrum.
"""
import sys, os, copy

cur = os.getcwd()
cur = cur.split('/')[:-1]
add_path = '/'.join(cur)+'/reduction'
sys.path.append(add_path)

from wirc_drp.constants import *
from scipy.ndimage import zoom, shift
from scipy.interpolate import interp1d
from scipy.integrate import simps
#from astropy.io import ascii as a, fits as f
from wirc_drp.utils.misc import fluxes, imGen, regularize, pixDownsample, binDownsample
from astropy.convolution import Gaussian2DKernel, convolve,convolve_fft, Box1DKernel, Gaussian1DKernel
from scipy.ndimage import shift, gaussian_filter, rotate
from astropy.io import fits, ascii as asci

from wirc_drp.utils.spec_utils import frame_rotate

import time
import numpy as np

def polarize(wl, spec, polarization, polarization_angle):
    """
    polarize takes in a template spectrum, degree of polarization, and the 
    angle of polarization, and return
    four polarized spectra
    Inputs: 
        wl           -- vector of wavelengths
        spec         -- vector of fluxes
        polarization -- a vector of degree of polarization telling us the 
                        relative flux each component gets. This has the same dimension
                        as wl and spec. 
        polarization_angle -- the angle of polarizaion in degrees. This is one number at all wl (?)
    output: 4 vectors of fluxes in 4 traces as well as the polarization angle
            in the same units as the input.
    """
    #calculate the Stokes parameter
    I = spec /2
    Q = polarization * I * np.cos(2*np.radians(polarization_angle))
    U = Q * np.tan(2*np.radians(polarization_angle))
    #calculate relative fluxes
    Q1 = (Q + I)/2
    Q2 = (I - Q)/2
    U1 = (U + I)/2
    U2 = (I - U)/2
    return [Q1,Q2,U1,U2] #generate 4 flux vectors for image generation
    
def locationInIm(wl, location_in_fov, offset_list = None):
    """compute for a source at the location_in_fov (y,x) in pixel scale, where
    4 traces would land in the actual detector.
    Inputs:
        wl -- the wavelength of the source
        location_in_fov -- (y,x) coordinates of the zeroth order in pixel
        offset_list -- a list of length 4, each giving a (dy, dx) offset from 
                        the calculated trace location. This is to match to real data.
    Outputs: A list of [x,y] location for [Q1, Q2, U1, U2]
    rounded"""
    #current assumption: 1.45 micron lands in the middle of the frame
    #update this with Seth's Zemax result
    dwl = wl-1.45 #This compute the deviation from (511,511)
    dpx = round(dwl/wlPerPix) #wlPerPix defined in constant.py
    #print(dpx)
    ####Old, Q+ = upper left, Q- lower right, U+ upper right, U- lower left
    # loc_in_im = [ [1024+location_in_fov[0]+dpx, location_in_fov[1]-dpx], \
    #                 [location_in_fov[0]-dpx, 1024+location_in_fov[1]+dpx], \
    #                [1024+ location_in_fov[0]+dpx, 1024+location_in_fov[1]+dpx], \
    #                 [location_in_fov[0]-dpx, location_in_fov[1]-dpx]]
    #New, from measurement: 
    #Location of UL, LR, UR, LL trace. Note that coordinates below are BEFORE 45 degree rotation. 
    if offset_list is None:
        offset_list = [[0,0]]*4
    elif len(offset_list) != 4:
        print("offset_list must be length of 4, offsets for the 4 traces.")
        offset_list = [[0,0]]*4
    loc_in_im = [[location_in_fov[0]-dpx      + offset_list[0][0]  , location_in_fov[1]-dpx     + offset_list[0][1]]   ,\
                [1024+ location_in_fov[0]+dpx + offset_list[1][0] , 1024+location_in_fov[1]+dpx + offset_list[1][1]]   ,\
                [location_in_fov[0]-dpx       + offset_list[2][0] , 1024+location_in_fov[1]+dpx + offset_list[2][1]]   ,\
                [1024+location_in_fov[0]+dpx  + offset_list[3][0] , location_in_fov[1]-dpx      + offset_list[3][1] ]]  
    return loc_in_im, dpx
    
def addTraces(traces, location, mid_wl, offset_list = None, angle_list = None):
    """addTraces take a list of 4 traces (each an image array) and put it in 4 
    quadrants of a 2048x2048 image according to the given location of the source
    in a 1024x1024 field of view.
    traces are in [Q1,Q2,U1,U2] format.
    Output is the 2048x2048 image"""
    #Deal with the xy and angle offsets
    if offset_list is None:
        offset_list = [[0,0]]*4
    elif len(offset_list) != 4:
        print("offset_list must be length of 4, offsets for the 4 traces.")
        offset_list = [[0,0]]*4
    if angle_list is None:
        angle_list = [0]*4
    elif len(angle_list) != 4:
        print("offset_list must be length of 4, offsets for the 4 traces.")
        angle_list = [0]*4
    loc, dpx = locationInIm(mid_wl, location, offset_list) #given the source location, get locations of 4 traces in the image
    #print('mid_wl', mid_wl)    
    image = np.zeros((2048,2048))
    # traces[0] = traces[0] #Q+   
    # traces[1] = traces[1][-1::-1, -1::-1] #Q-
    # traces[2] = traces[2][:, -1::-1] #U+
    # traces[3] = traces[3][-1::-1, :] #U-
    #flip traces into correct orientation: Q+ = LL, Q- = UR, U+ = LR, U- = UL
    traces[0] = traces[0][-1::-1, :] #Q+   
    traces[1] = traces[1][:, -1::-1] #Q-
    traces[2] = traces[2][-1::-1, -1::-1] #U+
    traces[3] = traces[3] #U-
    ###apply angles from angle_list
    for i in range(4):
        width = traces[i].shape[0]
        if angle_list[i] != 0:
            traces[i] = frame_rotate(traces[i], angle_list[i], cxy=[width/2,width/2])
    #put it in the full image
    for i in range(4):
        image[int(loc[i][0]-75):int(loc[i][0]+76), int(loc[i][1]-75):int(loc[i][1]+76)] = traces[i]
        #This is because we pick the box for each trace to be 151x151 pixel to accommodate J&H
    return image
    
def handleSkySpec(sky_spectrum, filter_name, obs_sky):
    """
    This takes in the sky_spectrum and filter transmission curve, then normalize
    the sky_spectrum so that the integrated flux matches that from obs_sky.
    
    Inputs: 
          sky_spectrum: Table with ['wl' (nm), 'flux' (photon/s/arcsec^2/nm/m^2)] format. This is from Gemini sky emission file
          filter_name: Specify which filter we want. 
          obs_sky: in mag/arcsec^2
    Output:
          sky_spectrum in [wl, count] 
    """
    #Get filter information    
    lb,dlb,f0,filter_trans_int, central_wl = getFilterInfo(filter_name)

    #interpolate the spectrum 
    sky_spectrum_int = interp1d(sky_spectrum['wl']/1000, sky_spectrum['flux']*1000) #convert to micron and photons/s/arcsec^2/micron/m^2
    
    #use array of wavelengths from filter_trans_int to get spectrum
    wl = np.linspace(1.001*np.min(filter_trans_int.x), 0.999*np.max(filter_trans_int.x), 1000)     #micron
    sky_spec = sky_spectrum_int(wl)*filter_trans_int(wl)  #photons/s/arcsec^2/micron/m^2

    #Integrate to get total flux. This is to be used to scale the spectrum with the given sky brightness
    total_flux = simps(sky_spectrum_int(wl)*filter_trans_int(wl), wl)
    
    #Expected flux from the given sky background in mag/arcsec^2
    total_flux_exp = f0 * 10**(-obs_sky/2.5) *dlb #j_ph i photons/s/micron/m^2
    ratio = total_flux/total_flux_exp
    #print(ratio)
    #apply the ratio to match the observed sky brightness
    sky_spec = np.array(sky_spec / ratio)
    
    #convert to count in each pixel. Note that fluxes takes wl in micron and spec in W/m^2/micron. Multiply by hc/wl
    sky_spec_pix = fluxes(wl, sky_spec*h*c/wl, central_wl)
    return(sky_spec_pix) #This is in photon/s/
    
def formSkyCube(filter_name, sky_mag, focal_plane_mask):
    """
    This function forms sky data cube from the given filter and sky mag
    The sky cube is 3 dimensional: first two are y,x position and the last is the count

    """
    lb,dlb,f0,filter_trans_int, central_wl = getFilterInfo(filter_name)
    #First start with sky cube
    sky_cube = np.zeros((1024,1024, len(central_wl)))
    #get sky flux vector 
    sky_flux = handleSkySpec(sky_spec_data, filter_name, sky_mag) 
    #spatial variation
    macro_variation = zoom( np.random.normal( 1, 0.1, (2,2)), zoom = 1024/2) ###This is hard coded. Real background variation from data is needed. 
    #put this in cube
    start = time.time()
    #for i in range(1024):
    #    for j in range(1024):
    #        sky_cube[i,j] = sky_flux*macro_variation[i,j]*focal_plane_mask[i,j]
    sky_cube = sky_flux*macro_variation[:,:,None]*focal_plane_mask[:,:,None] #Indexing to multiply 2D arrays (macro_variation,mask) to a 3D array
    print('loop ', time.time()-start)
    #sky_cube[i,j,:] = sky_cube[i,j,:]*macro_variation[i,j]
    return sky_cube

    
def background(mag_bg, sigma_bg, focal_plane_mask=None, filter_name = 'J'):
    """
    Takes sky brightness in mag/arcsec**2, filter, and focal_plane_mask, which 
    is an array of 0 or 1. Makes an on detector simulation of the effect of this 
    slit mask.
    """
    #fov size 
    #fov_size = 1024 #size of the opening in the focal plane mask
    #Get filter info
    lb,dlb,f0,filter_trans_int, central_wl = getFilterInfo(filter_name)
    #Focal plane mask
    if focal_plane_mask == None:
        focal_plane_mask = np.ones((fov_size,fov_size))
    data_cube = formSkyCube(filter_name, mag_bg, focal_plane_mask) * th * area * wlPerPix * sampling**2 #convert to counts
    num_spec_element = np.shape(data_cube)[2]
    #now sum in the diagonal direction
    qp = np.zeros((fov_size+ num_spec_element,fov_size+ num_spec_element))
    qm = np.zeros((fov_size+ num_spec_element,fov_size+ num_spec_element))
    up = np.zeros((fov_size+ num_spec_element,fov_size+ num_spec_element))
    um = np.zeros((fov_size+ num_spec_element,fov_size+ num_spec_element))
    for i in range(num_spec_element):
        qp[i:fov_size+i, num_spec_element-i: fov_size+num_spec_element-i ] += data_cube[:,:,i]/4 #Divide by 4 because it's dispersed into 4 quadrants
        qm[ num_spec_element-i: fov_size+num_spec_element-i , i:fov_size+i] += data_cube[:,:,i]/4
        up[i:fov_size+i, i:fov_size+i ] += data_cube[:,:,i]/4
        um[num_spec_element-i: fov_size+num_spec_element-i , num_spec_element-i: fov_size+num_spec_element-i ] += data_cube[:,:,i]/4
    #Now crop based on the dispersion of the band
    foo,dr = locationInIm(central_wl[0], (0,0)) #dr is the shift in pixel we want
    qp = shift(qp, (dr,-dr))[:fov_size,num_spec_element:]  
    qm = shift(qm, (-dr,dr))[num_spec_element:, :fov_size] 
    up = shift(up, (dr,dr))[:fov_size, :fov_size]
    um = shift(um, (-dr,-dr))[num_spec_element:,num_spec_element:]
    #print(np.shape(qp))
    
    up = np.hstack((qp,up))
    down = np.hstack((um,qm))
    bg = np.vstack((down,up))
    #For direct background
    direct_bg = np.zeros((detector_size,detector_size))
    direct_bg[1023-512:1023+512,1023-512:1023+512 ] = np.sum(data_cube, axis = 2)
    #Get rid of residual small numbers
    bg = np.abs(bg)
    direct_bg = np.abs(direct_bg)
    return bg, direct_bg
       
       
#some dummy masks
mask = np.ones((1024,1024))
mask[511-120:512+120, 511-120:511+120] = 0
mask[511-2:511+2, 511-2:511+2] = 1
#plt.imshow(mask, origin = 'lower')

#bg, direct_bg = background(15.4, 0.1, mask, 'J')
#plt.imshow(bg, origin = 'lower')

#bg, direct_bg = background(sky_h, 0.1, mask, 'H')
#plt.imshow(bg, origin = 'lower')        
#    

def shot_noise(scene):
    """
    calculate the shot noise which caused by photon's Poisson distribution. Because of the 
    random nature of photon noise and diffience of each pixel, go through each pixel
    and draw a random number from the Poisson distribution with mean = count at that
    pixel.
    """
    #noisy = np.zeros((2048,2048))
    #for i in range (2048):
    #    for j in range(2048):
            #print(scene[i][j])
    #        noisy[i][j] = np.random.poisson(lam = scene[i][j])
    noisy = np.random.poisson(lam = scene) 
    return noisy

def ron(read_noise, sigma_ron):
    """
    calculate the readout noise. It follows the normal distribution, the mean is read_noise,
    and the input sigma_ron is standard deviation
    """
    return np.random.normal(read_noise,sigma_ron,(2048,2048))


def FF(sigma_FF):
    """
    calculate the flat field, also the input is standard deviation.
    input: sigma_FF -- standard deviation in the flat field
    output: an image array of normalized flat field. 
    """
    return np.random.normal(1,sigma_FF,(2048,2048))

import matplotlib.pyplot as plt       
def makeObject(spec, exp_time, seeing_pix, pol_vec, location, filter_name = 'J', offset_list = None, angle_list = None):
    """makeObject create a noiseless image of a source with a given spectrum, 
    exp_time, seeing, and the location 
    input: spec = [spec_wl, spec_flux], is the spectrum of the source in energy flux
            exp_time, seeing_pix: self-explanatory
            pol_vec = [pol_wl, pol_fraction, pol_angle], is the polarization spectrum of the source
            pol_angle: the polarization angle
            location: (x,y) of the source.
            filter_name: the observing band, J or H
            offset_list: list of [dy, dx] offsets for the 4 traces
            angle_list: list of angle offsets (from 45 deg) for the 4 traces
    """
    #Get filter information
    lb,dlb,f0,filter_trans_int, central_wl_pix = getFilterInfo(filter_name)
    
    #First, make spec_wl and spec_flux at regular dwl spacing
    spec_wl = spec[0]
    spec_flux = spec[1]
    #wl, fx = regularize(spec_wl, spec_flux, (lb-dlb/2-0.2, lb+dlb/2+0.2))
    fx = interp1d(spec_wl, spec_flux, bounds_error=False)
    
    #apply filter transmission function
    wl = np.linspace(1.001*np.min(filter_trans_int.x), 0.999*np.max(filter_trans_int.x), 1000)
    fx = fx(wl)*filter_trans_int(wl)  #photons/s/arcsec^2/micron/m^2
    
    #Now we need to resample it at the pixel resolution. 
    fx_pixRes = pixDownsample(wl, fx, wlPerPix) #wlPerPix from constants.py. This is defined by the grating
    #print fx_pixRes    
    
    #Make the polarization spectrum
    interp_pol = interp1d(pol_vec[0], pol_vec[1], bounds_error=False)
    interp_pol_ang = interp1d(pol_vec[0], pol_vec[2], bounds_error=False)

    

    #polarize the trace
    Q1, Q2, U1, U2 = polarize(wl, fx_pixRes, interp_pol(wl), interp_pol_ang(wl))

    # plt.plot(wl, Q1)
    # plt.plot(wl, Q2)
    # plt.plot(wl, U1)
    # plt.plot(wl, U2)

    # plt.show()

    #now for each polarization, generate a trace and put it where it belongs on the final image.
    #image = np.zeros((2048,2048))
    traces = []
    # np.savetxt('wavelength.txt',central_wl_pix)
    j=0
    for i in [Q1,Q2,U1,U2]:
        j += 1
        #first draw flux in each bin
        flux_in_pix = fluxes(wl, i, central_wl_pix) * wlPerPix
        #convert this into flux per pixel
        flux_in_pix = flux_in_pix * area * th *(1-zero_fraction)  #multiply by collecting area and throughput   
        # plt.plot(flux_in_pix)    
        # np.savetxt(str(j)+'.txt',flux_in_pix)
        _,trace = imGen(flux_in_pix, 151, exp_time, seeing_pix) #151 is to accommodate both J and H traces
        #trace = rotate(trace,1,order = 5, reshape = False)
        traces += [trace]

    #add traces into image
    #addTraces_time= time.time()
    #image = addTraces(traces, location, (central_wl_pix[-1] + central_wl_pix[0] )/2)
    image = addTraces(traces, location, lb, offset_list = offset_list, angle_list = angle_list)
    
    #print('addTraces takes ', time.time()- addTraces_time)
    #zeroth order and direct image
    flux_in_zero = np.sum(fluxes(wl, fx_pixRes, central_wl_pix)*wlPerPix * area*th * exp_time/4) #use 1/4 of exposure time because flux would be 4 times that of dispersed image 
    #print(flux_in_zero)
    direct_image = np.zeros((2048,2048))
    direct_image[int((1024+2*location[0])/2),int((1024+2*location[1])/2)] = flux_in_zero #direct image, all flux go into 0th order
    
    #Define seeing kernel using astropy Gaussian2DKernel    
    seeingKernel = Gaussian2DKernel(seeing_pix/(2*np.sqrt(2*np.log(2)))) #seeing_pix is FWHM = 2 sqrt(2 ln 2) sigma
    # start = time.time()    
    direct_image = convolve(direct_image, seeingKernel) #seeing convolution 
    #direct_image = gaussian_filter(direct_image, seeing_pix/(2*np.sqrt(2*np.log(2))))
    #print('conv takes ', time.time()-start)

    image = image + zero_fraction*direct_image #zero_fraction of direct_image leaks into the spectral image
    # plt.imshow(image, origin = 'lower');plt.show()
    return direct_image, image

def imageFromSpec(obj_list, exp_time, seeing_pix, angle, bg_mag, bg_sigma, ron, ff_sigma, mask = None, filter_name = 'J' ):
    """
    This is the control function that calls all other functions to 
    produce an image given the spectrum.
    Inputs: obj_list -- a list of objects and relevant info in this format:
                        [spec_wl, spec_flux, (x,y), pol_vec ]
                        pol_vec is givien as [[wavelengths],[pol_fraction], [pol_angle]]
            seeing_pix -- FWHM of the seeing in pixel unit. 
            angle  --  angle from 45 degrees
            bg_mag -- the mean brightness of the background in mag/arcsec^2
            ron -- RMS read out noise in electron/pixel
            ff_sigma -- the SD of the normalized flat field histogram (median = 1)
            
    Output: An 2048x2048 image array. 
    """
    #Start with 2 blank frames for 0th order and the spectral image
    imFS_start = time.time()
    direct_im = np.zeros((2048,2048))    
    scene = np.zeros((2048,2048))
    
    #for each object in obj_list, call makeObject and add the result to the frames
    for i in obj_list:
        mO_time = time.time()
        direct_image, image = makeObject([i[0], i[1]], exp_time, seeing_pix, i[3], i[2], filter_name = filter_name) #good ordering!
        #print('makeObject takes ', time.time()-mO_time)        
        direct_im += direct_image
        scene += image
    #print('elapsed one ', time.time()-imFS_start)
    
    #Now generate sky background
    sky, sky_direct = background(bg_mag, bg_sigma,mask, filter_name)
#    x_shift = 100
#    y_shift = 100
#    scene = shift(scene,(x_shift,y_shift))
#    direct_im = shift(direct_im,(x_shift,y_shift))
    
    #Introduce some rotation
    # scene = rotate(scene, angle, reshape = False, order = 4)
    width = scene.shape[0]
    scene = frame_rotate(scene, angle, cxy=[width/2,width/2]) #replace scipy rotate with opencv bicubic
    #print(scene[scene<0])
    
    #Add sky background
    scene = scene+sky*exp_time #add sky background to the scene
    direct_im = direct_im+sky_direct*exp_time/4 #use 1/4 of exposure time because flux would be 4 times that of dispersed image   
    #print('adding shit takes ', time.time()-addshit_time)
    
    #Poisson noise
    scene[scene<0] = 0
    scene = shot_noise(scene)
    direct_im = shot_noise(direct_im)
    #print('elapsed three ', time.time()-imFS_start)   
    return scene, direct_im, sky
    
def injectSource(base_image, obj_list, HWP_angle, seeing_pix,  exp_time, filter_name, \
                     offset_list = 'default', angle_list = None):
    """
    injectSource takes an input real image from WIRC, and inject sources from the object list. 
    Inputs: 
            base_image -- an array of original image to be injected with a fake source
            obj_list -- a list of objects and relevant info in this format:
                        [spec_wl, spec_flux, (x,y), pol_vec]
                        pol_vec is given as [[wavelengths],[pol_fraction],[pol_angle]]
            seeing_pix -- FWHM of the seeing in pixel unit. 
            offset_list -- list of offsets (y,x), if 'default', use default for each filter
            angle_list -- list of angle offsets from 45 degree. 
    Output: a fits file of the input_image with all sources from obj_list injected.

    To do: Make sure to deal with image dimension without using magic numbers. 
    """
    #Default offsets
    if offset_list == 'default':
        if filter_name == 'J':
            offset_list =  [[-15,-13],[0,20],[-20,0],[10,-5]] #measured from data. This may change depending on source location
        elif filter_name == 'H':
            offset_list = [[-20,-15],[5,20],[-20,0],[10,0]]
    #for each object in obj_list, call makeObject and add the result to the frames
    scene = np.zeros(base_image.shape)
    for i in obj_list:
        #deal with the effect of the HWP
        pol_wl   = i[3][0]
        pol_frac = i[3][1]
        pol_ang  = i[3][2]
        pol_ang_actual = pol_ang + 2*HWP_angle 
        _, image = makeObject([i[0], i[1]], exp_time, seeing_pix, [pol_wl, pol_frac, pol_ang_actual], i[2], filter_name = filter_name, \
                                    offset_list = offset_list, angle_list = angle_list) #good ordering!
        # direct_im += direct_image
        scene += image
        
    #Introduce some rotation -- deprecated. Now this is done by defining angle offset for each trace in angle_list. 
    # scene = rotate(scene, angle = 0, reshape = False, order = 4)
    # width = scene.shape[0]
    # scene = frame_rotate(scene, angle = 0, cxy=[width/2,width/2])  #replace scipy rotate with opencv bicubic
    # scene[scene<0] = 0
    #print(scene[scene<0])
    
    #Poisson noise
    scene[scene<0] = 0

    scene = shot_noise(scene)

    return base_image + scene

def injectSourceToFiles(filelist, out_path, obj_list, seeing_pix = 4, \
                        fake_exp_time = None, offset_list = 'default', angle_list = None):
    """
    Inputs: 
            filelist -- a list of fits filenames of real images from WIRC+Pol to be injected
            out_path -- where to put output images. 
            obj_list -- a list of objects and relevant info in this format:
                        [spec_wl, spec_flux, (x,y), pol_vec]
                        pol_vec is given as [[wavelengths],[pol_fraction],[pol_angle]]
            seeing_pix -- FWHM of the seeing in pixel unit. 
            fake_exp_time -- if a deliberately wrong exposure time is desired. 
            offset_list -- list of offsets (y,x), if 'default', use default for each filter
            angle_list -- list of angle offsets from 45 degree. 
    """
    for input_image in filelist:
        # Open file
        input_im = fits.open(input_image)
        # Set some parameters
        # Exposure time, use real one unless a fake exposure time is given
        if fake_exp_time is None:
            exp_time = input_im[0].header['EXPTIME'] * input_im[0].header['COADDS']
            # print(exp_time)
        else: 
            exp_time = fake_exp_time
        #Set filter name
        filter_str = input_im[0].header['AFT']
        if 'J__' in filter_str:
            filter_name = 'J'
        elif 'H__' in filter_str:
            filter_name = 'H'
        else:
            print('Filter %s not supported. Only J or H'%filter_str)
            return None

        #This parameter is the image
        image = input_im[0].data

        # direct_im = np.zeros(scene.shape) #for backward compatibility

        #Deal with HWP angle
        HWP_angle = input_im[0].header['HWP_ANG']
        # print(HWP_angle)

        #debug
        print(exp_time, filter_name, HWP_angle)

        injected_image = injectSource(image, obj_list, HWP_angle, seeing_pix,  exp_time, filter_name, \
            offset_list = offset_list, angle_list = angle_list)

        #create a new HDU list to save to
        out_file = copy.deepcopy(input_im)
        out_file[0].data =  injected_image
        # out_file[0].header['HISTORY'] += ''
        new_name = 'injected_'+input_image.split('/')[-1] 
        print(new_name)
        out_file.writeto(out_path+new_name)



"""
TO DOs:
(0) Helper function to inject different source properties. 
(1) Add filter transmission profile shift between the top and bottom traces (postponed)
"""

def fakeSourceCreator(mag, spec_template_fn, pol_template_fn, pos):
    """
    Produce a fake source to be put in obj_list for source injection. 
    Inputs:
        mag: magnitude in J band
        spec_template_fn: filename of the spectral shape of the source. The spectrum will be normalized to the given mag ([wl,flux])
        pol_template_fn: polarization profile file (format [wl, pol, ang])
        pos: (x,y) position of the target
        **Note, wl in the 3 vectors should cover the same range. It MUST cover the J band for normalization.
    Output:
        a list describing an object. The format is [spec_wl, spec_flux, (x,y), pol_vec], 
        which is for the source injection functions
    """
    spec_template = asci.read(spec_template_fn)
    pol_template = asci.read(pol_template_fn)

    wl = spec_template[0]
    #interpolate if the wl grid are not the same
    if pol_template[0] != wl:
        pol_interp = interp1d(pol_template[0], pol_template[1], bounds_error=False)
        pol_vec = pol_interp(wl)
        ang_interp = interp1d(pol_template[0], pol_template[2], bounds_error=False)
        ang_vec = ang_interp(wl)

    #normalize the flux (assuming given F_lambda)
    #Get J band info
    lb,dlb,f0,filter_trans_int, central_wl = getFilterInfo('J')
    fil_trans = filter_trans_int(wl)
    #Total flux
    int_flux = simps(spec_template[1]*fil_trans, wl, dx = wl[1]-wl[0])
    #J band zero point
    J_zero_lam = 3.147e-9 #W/m^2/micron (sorry, it's the unit the old function used)
    desired_flux = J_zero_lam * 10**(-mag/2.5)
    #Now, this is the actual spectrum in the W/m^2/micron unit at the given mag in J band
    actual_spec = spec_template[1] * desired_flux/int_flux 

    return [wl, actual_spec, (pos[0],pos[1]), [wl, pol_vec, ang_vec]]
