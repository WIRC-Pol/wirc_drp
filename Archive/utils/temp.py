import numpy as np
from astropy.io import fits as f
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter, label, find_objects
import time
from DRP.reduction.constants import *
import astropy.io.ascii as a
from scipy.ndimage import gaussian_filter as gauss
import os
import copy


def find_sources_in_direct_image(direct_image, mask, threshold_sigma, guess_seeing, plot = False):
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
    if direct_image != None:
        
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
   
def locationInIm(wl, location_in_fov):
    """compute for a source at the location_in_fov (y,x) in pixel scale, where
    4 traces would land in the actual detector.
    Outputs: A list of [x,y] location for [Q1, Q2, U1, U2], rounded

    Used by cutout_trace_thumbnails
    """
    #Use measured values from test data
    dwl = wl-1.25 #This compute the deviation from J band where the offsets were mesured
    dpx = round(dwl/(wlPerPix))
    
    traceLocation = [ [ 453+location_in_fov[0]+dpx , -435 + location_in_fov[1]-dpx],\
                    [  -459+location_in_fov[0]-dpx, 450+ location_in_fov[1]+dpx], \
                   [    446+location_in_fov[0]+dpx, 449+location_in_fov[1]+dpx], \
                    [  -445+location_in_fov[0]-dpx, -455+location_in_fov[1]-dpx]]
    return np.array(traceLocation)

def cutout_trace_thumbnails(image, locations, flip = True, filter_name = 'J', sub_bar = True):
    '''
    This function Extracts the thumbnails of each trace for a given image give a locations list. 
    image - the image where you want to extract the traces
    locations - the locations in the image that you want to use as a basis for extraction
    flip - An optional switch that allows you to flip all the trace thumbnails to be orientated in the same direction 
            (i.e. wavelength increasing in the same direction)
    filter_name  - the filter. This determines the cutout size. 
    '''

    if filter_name == 'J':
        cutout_size = 80 #Make cutout of each trace. This has to chage for J/H bands
    elif filter_name == 'H':
        cutout_size =150
    else:
        print('Filter name %s not recognized, assuming J' %filter_name)
        cutout_size = 80

    cutouts = [] #A list of thumbnails lists

    for k,i in enumerate(locations[:,0]):

        thumbnails = [] #A thumbnail list of each traces
        
        traceLocation = locationInIm(lb, i).astype(int) #These are locations in Q+,Q-,U+,U- respectively. 
        #print(traceLocation)        


        ###This part is just to get show 4 traces of a source
        ntraces = len(traceLocation) #how many traces? 4 for WIRC-POL

        for j in range(ntraces):

            #The indices for the cutout
            cutout = np.s_[traceLocation[j][0]-cutout_size:traceLocation[j][0]+cutout_size+1,\
                            traceLocation[j][1]-cutout_size:traceLocation[j][1]+cutout_size+1 ]

            
            #cut the spectral image into a thumbnail containing the trace
            thumbnail = copy.deepcopy(image)[cutout]
                       
            #flip the thumbnail so that it's in the Q+ orientation (from top left to bottom right)
            if flip: 
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

            if filter_name == 'J':
                if sub_bar:
                    #Check to see if the traces hit the vertical bars of doom
                    if ((i[1] > slit_position_x+300) & (i[1] < slit_position_x+600)) | ((i[1] < slit_position_x-300) & (i[1] > slit_position_x-600)):
                        
                        if j == 0: 
                            print("Source {}'s traces will hit the vertical bar of doom, compensating by subtracting the median of the edges of each row".format(k+1))

                        for xind in range(np.shape(thumbnail)[1]):
                            sub_length = 10 #The number of pixels at the beginning and end to estimate the background
                            thumbnail[:,xind] -= np.median(np.concatenate([thumbnail[:sub_length-1,xind],thumbnail[-(sub_length):,xind]]))


                    
                    if ((i[0] > slit_position_y+300) & (i[0] < slit_position_y+600)) | ((i[0] < slit_position_y-300) & (i[0] > slit_position_y-600)):
                        if j == 0: 
                            print("Source {}'s traces will hit the horizontal bar of doom".format(k+1))

                        for yind in range(np.shape(thumbnail)[0]):
                            sub_length = 10 #The number of pixels at the beginning and end to estimate the background
                            thumbnail[yind,:] -= np.median(np.concatenate([thumbnail[yind,:sub_length-2],thumbnail[yind,-(sub_length):]]))


            thumbnails.append(thumbnail)

        cutouts.append(thumbnails)

    return cutouts

