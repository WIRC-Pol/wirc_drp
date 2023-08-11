# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:22:27 2016

@author: kaew
Various focal plane masks to be simulated
"""
import numpy as np
from imageio.v2 import imread #Scipy got rid of imread in version 1.2 and we now need to use iamgeio
import os 
from astropy.io import fits
#from constants import *

#wircpol_dir = os.environ['WIRC_DRP']
wircpol_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2]) 

fov_size = 1024
spider_thickness = 5
slit_width  = 5
####Square with slit
sq_mask = np.ones((1024,1024))
#mask
sq_mask[511-120:512+120, 511-120:511+120] = 0
#spider
sq_mask[511-spider_thickness:511+spider_thickness,:] = 0
sq_mask[:,511-spider_thickness:511+spider_thickness] = 0
#slit
sq_mask[511-slit_width:511+slit_width, 511-slit_width:511+slit_width] = 1

####Square with long slit
sq_mask_ls = np.ones((1024,1024))
#mask
sq_mask_ls[511-120:512+120, 511-120:511+120] = 0
#spider
sq_mask_ls[511-spider_thickness:511+spider_thickness,:] = 0
sq_mask_ls[:,511-spider_thickness:511+spider_thickness] = 0
#slit
sq_mask_ls[511-5*slit_width:511+5*slit_width, 511-slit_width:511+slit_width] = 1

####Circle with slit
#circ_mask = np.ones((1024,1024))
xx, yy = np.mgrid[:fov_size, :fov_size]
circ = (xx-511)**2+(yy-511)**2 #circle equation at the center of FOV
circ_mask = circ> 2*(120)**2 #radius
#spider
circ_mask[511-spider_thickness:511+spider_thickness,:] = 0
circ_mask[:,511-spider_thickness:511+spider_thickness] = 0
#slit
circ_mask[511-slit_width:511+slit_width, 511-slit_width:511+slit_width] = 1

####Cross mask with slit
#cross_mask = imread('../mask_design/cross_mask/cross_mask.001.png', flatten = True)
#cross_mask[cross_mask < 20] = 0
#cross_mask[cross_mask > 20] = 1


####Cross mask version 2
cross_mask_v2 = imread(wircpol_dir+'/masks/cross_mask/cross_mask_v2.002.png', mode="L")
cross_mask_v2[cross_mask_v2 < 20] = 0
cross_mask_v2[cross_mask_v2 > 20] = 1

####Cross mask, no slit. For reduction pipeline
cross_mask_ns = imread(wircpol_dir+'/masks/cross_mask/cross_mask_v2.003.png', mode="L")
cross_mask_ns[cross_mask_ns < 20] = 0
cross_mask_ns[cross_mask_ns > 20] = 1

####Cross mask, with circular holes
cross_mask_circ = imread(wircpol_dir+'/masks/cross_mask/cross_mask_v2.003.png', mode="L")
cross_mask_circ[cross_mask_circ < 20] = 0
cross_mask_circ[cross_mask_circ > 20] = 1

#put holes in 
xx, yy = np.mgrid[:len(cross_mask_circ[0]), :len(cross_mask_circ[0])]
circ0 = (xx-511.5+21)**2+(yy-511.5)**2
circ1 = (xx-511.5)**2+(yy-511.5)**2 #circle equation at the center of FOV
circ2 = (xx-511.5-21)**2+(yy-511.5)**2
circ_mask = np.logical_or(np.logical_or(circ0 < (6)**2 , circ1 < (6)**2), circ2 < 6**2) #radius

cross_mask_circ = cross_mask_circ + circ_mask
cross_mask_circ = np.transpose(cross_mask_circ)

def makeDiagMask(size, width):
    """
    Create a diagonal mask to remove the messy background part in background
    subtracted thumbnails of a source in the slit. 
    Inputs: size: the dimension of the square thumbnail
            width: the width of the area around diagonal to let through (and block the rest)
    Output: an array of dimension (size, size) with value 1 within length = "width"
            above and below diagonal, 0 elsewhere.
    """
    mask = np.zeros((size,size),dtype="int8")
    for i in range(size):
        mask[int(size-i-1),int( max(i-width,0)): int(min(i+width, size)) ] = 1
    return mask.astype("bool")

def make_mask_from_findTrace(fit, angle, width=None):
    """
    Create a diagonal mask based on the output from findTrace 
    """
    if width is None:
    #actual width
        width = width/np.cos(np.radians(angle))**2 #cos^2 because one factor from rotation, another from the smaller pixel
    
    size = len(fit)
    mask = np.zeros((size,size))

    for i in range(size):
        min_ind = int(np.max([fit[i]-width, 0]))
        max_ind = int(np.min([fit[i]+width, size]))

        mask[min_ind:max_ind, i] = 1

    return mask.astype("bool")
        

trace_masks = fits.open(wircpol_dir+'/masks/trace_masks.fits')[0].data

cross_correlation_template = fits.getdata(wircpol_dir+'/masks/cross_correlation_template.fits')
trace_template = fits.getdata(wircpol_dir+'/masks/trace_template.fits')


####Cross mask with 2 size slit
#cross_mask_vs = imread('../mask_design/cross_mask_v2/cross_mask_v2.003.png', flatten = True)
#cross_mask_vs[cross_mask_vs < 20] = 0
#cross_mask_vs[cross_mask_vs > 20] = 1
