# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 22:03:27 2016

@author: kaew
Constants
This file defines all constants needed in the data reduction pipeline. All updates
should be only done to this file.
"""
import numpy as np
import astropy.io.ascii as a
from scipy.interpolate import interp1d
import os

wircpol_dir = os.environ['WIRC_DRP']

####TELESCOPE + DETECTOR SPEC####
#Photon flux (Allen)
j_ph=2e10#number of ph-e- per sec per micron per m**2

#Measurements from Palomar sky background
sky_j = 15.4 #mag per arcsec**2
sky_h = 13.9 #mag per arcsec**2
#sky spectrum from Mauna Kea. To be scaled
sky_spec_data = np.load(wircpol_dir+'/wirc_drp/specification/sky_emission.npy')

#Palomar Hale telescope
diam=5.1#meter
obs=0.36#unitless
sampling=0.2487#in arcsecond / pix
plate_scale = sampling
#seeing = 1
#seeing_pix = seeing/sampling
area = (diam/2)**2. * np.pi - (obs*diam/2)**2. * np.pi #meter^2
#area = 13 #m^2, source http://www.astro.caltech.edu/palomar/observer/200inchResources/lfc/lfcpick.html
angular_magnification = 97.7

#instrument--these are to be deprecated. Filter info below is to be used.
lb=1.25#in microns
passband = 0.162#in microns
th = 0.17#instrument throughput; source http://www.astro.caltech.edu/palomar/observer/200inchResources/lfc/lfcpick.html
#0.3#optical throughput: revised by Ming but corrected for new detector increase QE
sigma_bg = 1       #background sigma number
sigma_ron = 12      #readout sigma in electrons
sigma_FF = 0.15    #Flux field sigma number  This number is gotten from OSIRIS archival
gain = 1.2 #e-/ADU 
#Detector dimension
fov_size = 1024 #about half opening
detector_size = 2048 #full detector array

#Mask specification
slit_length = 50 #pixel on detector
#slit_width = 10 #TBC!!

slit_hole_diameter = 3/sampling #12 pixels, 3" diameter
slit_hole_gap = 21 #pixels. Again, this is spec. should be remeasured

#central slit position
slit_position_x = 1027-1
slit_position_y = 1060-1

# Nominal offsets (dx,dy) between traces and zeroth order in PG mode
dUL = (-442.76, 451.76)
dLL = (-452.90, -446.85)
dUR = (451.90, 442.55)
dLR = (442.25, -457.75)

#grating specification
wlPerPix = 2.83/1000 #wl coverage per diagonal pixel in microns
#zero order image flux/total flux. 0.5% for the current spec.
zero_fraction = 0.005
pg_period = 17 #microns

###PHYSICAL CONSTANTS
h = 6.626e-34 #J s
c = 2.998e14 #micron/s


##############################################
#########################Filter specifications
##############################################
#J band
J_lam = 1.2483 #micron
J_dlam = 0.162  
F0_J = 1.93e10 #photons/s/micron/m^2
J_trans = a.read(wircpol_dir+'/wirc_drp/specification/J_WIRC.csv',names = ('wl','transmission'))
#in micron, %
J_trans_int = interp1d(J_trans['wl'],J_trans['transmission']*0.01,bounds_error = False, fill_value = 0)



#H band
H_lam = 1.6313 #micron
H_dlam = 0.296 
F0_H = 0.93e10 #photons/s/micron/m^2
H_trans = a.read(wircpol_dir+'/wirc_drp/specification/H_WIRC.csv',names = ('wl','transmission'))
#in micron, %
H_trans_int = interp1d(H_trans['wl'],H_trans['transmission']*0.01, bounds_error = False, fill_value = 0)

def centralWlCalc(filter_transmission):
    """
    give a vector of central wavelengths of a row of pixels along a spectal trace
    """
    #start where filter_transmission becomes nonzero, stop when it's zero in 
    #wlPerPix increment
    central_wl = []
    start = np.min(filter_transmission.x) #start wl
    stop = np.max(filter_transmission.x)
    i = start
    while i < stop:
        if filter_transmission(i) > 0.8/100:
            central_wl+=[i]
        i += wlPerPix
    return np.array(central_wl)



def getFilterInfo(name):
    """This is a simple function that gives wavelength, bandpass, zeropoint flux, 
    transmission function, and the vector of central wavelengths given filter name."""
    if name == 'J':
        lb = J_lam
        dlb = J_dlam
        f0 = F0_J
        transmission = J_trans_int
        central_wl_pix = centralWlCalc(transmission)
    elif name == 'H':
        lb = H_lam
        dlb = H_dlam
        f0 = F0_H
        transmission = H_trans_int
        central_wl_pix = centralWlCalc(transmission)
    else:
        lb = None
        dlb = None
        f0 = None
        transmission  =None
        central_wl_pix = None
        print('Filter does not exist')
    return lb,dlb, f0, transmission, central_wl_pix

#test
#lb,dlb,f,tr,cwlp =getFilterInfo('J')


#scenario, J Band
#r = 155
#dlb  = lb/r
#nchannel = passband/dlb
#exp_time = 630      # in seconds
#num_exp = 10
#np.load(wircpol_dir+'/DRP/specification/sky_emission.npy')
trace_template = np.load(wircpol_dir+'wirc_drp/specification/trace_template.npy')




