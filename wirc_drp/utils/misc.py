###Miscellaneous functions to help with WIRC+Pol data reduction efforts
from astropy import coordinates
from astropy.time import Time
import astropy.units as u 
import numpy as np 

def compute_object_position(list_of_obs_time, obj_coords, observatory = 'Palomar'):
    """
    This function takes a list of observation times, or just an observation time, taken from FITS header, the object coordinates
    in astropy SkyCoord format, and compute the altitude, hour angle, and parallactic angle for each of the given time. 

    Input:
    	list_of_obs_time: a list of astropy Time object, one Time object is also acceptable 
    	obj_coords: coordinates of the object in astropy SkyCoord format
    	observatory: Default to Palomar, but can be changed to any string recognized by astropy: see astropy.coordinates.EarthLocation.get_site_names()
    """
#     alt = []
#     ha = []
#     pa = []
    #If single time object given, make it into a list
    if type(list_of_obs_time) != list:
        list_of_obs_time = [list_of_obs_time]

    #check if provided location is recognized
    if observatory not in coordinates.EarthLocation.get_site_names():
    	observatory = 'Palomar'
    	print('%s not recognized as a site, default to Palomar')

    #Set location so we can get sidereal time
    list_of_obs_time = [Time(x, location = coordinates.EarthLocation.of_site(observatory)) for x in list_of_obs_time]
    #First we compute the local sidereal time
    LST = [x.sidereal_time('apparent') for x in list_of_obs_time]
    #Then hour angle by subtracting RA from this
    HA = np.array([(x - obj_coords.ra).radian for x in LST]) #now in numpy array, with radian unit
    
    #Compute altitude
    lat = coordinates.EarthLocation.of_site(observatory).lat.radian #Get observatory latitude, default is Palomar
    dec = obj_coords.dec.radian
    
    alt = np.arcsin(np.sin(dec)*np.sin(lat)+np.cos(dec)*np.cos(lat)*np.cos(HA))
    
    #Now we can compute Parallactic Angle
    PA = np.arcsin(np.cos(lat)/np.cos(alt)*np.sin(HA))
    
    #Don't forget the Azimuth
    az = np.arcsin(-np.sin(HA)*np.cos(dec)/np.cos(alt))
    #Return everything IN RADIAN
    return alt, az, HA, PA


############Functions from old DRP used by image_simulation.py
import numpy as np
from wirc_drp.constants import *
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
from astropy.convolution import Gaussian2DKernel,convolve, convolve_fft, Box1DKernel, Gaussian1DKernel
from scipy.optimize import minimize, fmin_powell, leastsq, least_squares
from sklearn import linear_model
import matplotlib.pylab as plt
from scipy.ndimage import shift, rotate

def regularize(spec_wl, spec_flux, wl_range):
    """Return spectrum (wl and flux) vectors with regular spacing in wl between
    the range defined by wl_range (wl_min, wl_max).
    Inputs: spec_wl -- a vector of wavelength
            spec_flux -- a vector of fluxes
            wl_range -- a tuple of (wl_min, wl_max)
    Output: 2 vectors wl and fx representing a spectrum with regular dwl interval
            between wl_min and wl_max
    """
    #check if the wl_range is not exceeding spec_wl range
    if (min(spec_wl) > min(wl_range)) or (max(spec_wl) < max(wl_range)):
        print('wl_range wider than the given data.')        
        return None
    else:
        #first interpolate. use linear interpolation
        interpSpec = interp1d(spec_wl, spec_flux)
        #get wl points between wl_range
        wl_min = wl_range[0]
        wl_max = wl_range[1]
        idx = np.logical_and(spec_wl > wl_min, spec_wl < wl_max)
        num_point = len(spec_wl[idx])
        #now generate a wavelength vector from wl_min to wl_max with a regular spacing
        wl = np.linspace(wl_min, wl_max, num_point)
        #and the fluxes are just
        fx = interpSpec(wl)
        return wl, fx
        
def pixDownsample(spec_wl, spec_flux, wlPerPix):
    """Down sample the given spectrum to the pixel resolution by convolving with a box.
    Input: spec_wl -- a vector of wavelengths with regular dwl (equal increase in wl
                    from element to element.)
            spec_flux -- a vector of flux
            wlPerPix -- wl coverage per pixel. This is determined by the grating specification.
    Output: a vector of flux at pixel resolution
    """
    dwl = spec_wl[1] - spec_wl[0]
    pixKernel = Box1DKernel(wlPerPix/dwl)
    fx_pixRes = convolve(spec_flux, pixKernel)
    return fx_pixRes

def binDownsample(spec_wl, spec_flux, wlPerPix, seeing_pix):
    """Down sample the given spectrum to the spectral bin resolution by convolving with a seeing Gaussian.
    Input: spec_wl -- a vector of wavelengths with regular dwl (equal increase in wl
                    from element to element.)
            spec_flux -- a vector of flux
            wlPerPix -- wl coverage per pixel. This is determined by the grating specification.
            seeing_pix -- the FWHM of seeing in pixel scale. This is used to calculate the 
                            wl per bin
    Output: a vector of flux at spectral bin resolution
    """
    dwl = spec_wl[1] - spec_wl[0]
    #calculate spectral bin width. Note that seeing_pix is on the side. Divide by sqrt(2) to get diagonal    
    wlPerBin = wlPerPix * seeing_pix/np.sqrt(2) #THIS IS FWHM STILL
    binKernel = Gaussian1DKernel(wlPerBin/(2*np.sqrt(2*np.log(2)))/dwl) #division by  2*np.sqrt(2*np.log(2)) to convert FWHM to sigma
    fx_binRes = convolve(spec_flux, binKernel)
    return fx_binRes

def fluxes(spec_wl, spec_flux, central_wl):
    """Take vectors spec_wl and spec_flux of a spectrum, and the central_wl, assuming constant dlb in central_wl.
    spec_wl in angstrom
    spec_flux in W/m^2/micron  
    firstly convolve the spectrum!!
    returns a vector of flux per spectral bin in photons/s/m^2/micron.     
    """
    #define the center wavelength of central_wl and d_lambda
    idx = np.logical_and(spec_wl > np.min(central_wl)-0.01, spec_wl < np.max(central_wl)+0.01)
    num_point = len(spec_wl[idx])
    #print(num_point)
    interp_spec = interp1d(spec_wl,spec_flux)
    wl = np.linspace(np.min(central_wl)-0.01,  np.max(central_wl)+0.01,num_point)
    #wl = np.linspace(lb-passband/2-0.1, lb+passband/2+0.1, num_point)
    fx = interp_spec(wl)/(h*c/wl) #This is per trace.
    #wlPerPix = 2.83e-3 #2.83 nm/pixel DIAGONAL
    dlb = np.median(wl[1:]-wl[:-1]) #this is dlb. Note that in the resampled data, dlb is constant
    pixelKernel = Box1DKernel(wlPerPix/dlb)
    fx_pixRes = convolve(fx, pixelKernel)
    flux_in_pix = interp1d(wl, fx_pixRes)(central_wl)
    return np.array(flux_in_pix) #The unit returned is photons/s/m^2/micron

def imGen(flux_in_pix, len_pix, exp_time, seeing_pix):
    """imGen takes a vector of fluxes in each pixel, width of the output image,
    exposure time, and seeing in pixel, then creates a model trace array with
    values in counts in each pixel by convolving Gaussian to the skeleton trace.
    Outputs: image -- image without the seeing effect
            seeing_image -- image WITH the seeing effect (Gaussian). 
    """
    #zoomed = zoom(flux_in_pix, 58/len(flux_in_pix))
    zoomed = flux_in_pix
    #print(zoomed)
    flux_in_pix = zoomed * np.sum(flux_in_pix)/np.sum(zoomed)
    image = np.zeros((len_pix, len_pix)) #101x101 empty array
    #create the skeleton
    for i in range(int(np.floor((len_pix-len(flux_in_pix))/2)), int(np.floor((len_pix+len(flux_in_pix))/2))):
        image[i,len_pix-i] = flux_in_pix[i-int(np.floor((len_pix-len(flux_in_pix))/2))]*exp_time #now it's just #photons
        #print(image[i,101-i])
    #convolve to get the seeing effect
    seeingKernel = Gaussian2DKernel(seeing_pix/(2*np.sqrt(2*np.log(2)))) #seeing_pix is FWHM = 2 sqrt(2 ln 2) sigma
    seeing_image = convolve(image,seeingKernel)
    return image, seeing_image

