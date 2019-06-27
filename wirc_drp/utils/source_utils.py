from wirc_drp.constants import *
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.ndimage import shift
from scipy.signal import fftconvolve 
from astropy.io import ascii as asci
from astropy.io import fits 
from wirc_drp.utils import spec_utils as su



def plot_source_traces(source_list, cmap = None, figsize=(8,8), plot_lims = None):
    '''
    Plot all the traces 

    '''

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    #Pick a colormap
    if cmap == None:
        cm1 = plt.get_cmap('bwr')


    nsources = np.size(source_list)

    for i,source in enumerate(source_list):

        c = cm1(1.*(i)/nsources)

        ax1.plot(source.trace_spectra[0,0,:],source.trace_spectra[0,1,:], color=c)
        ax2.plot(source.trace_spectra[1,0,:],source.trace_spectra[1,1,:], color=c)
        ax3.plot(source.trace_spectra[2,0,:],source.trace_spectra[2,1,:], color=c)
        ax4.plot(source.trace_spectra[3,0,:],source.trace_spectra[3,1,:], color=c)

        if plot_lims != None:
            ax1.set_xlim(plot_lims[0:2])
            ax2.set_xlim(plot_lims[0:2])
            ax3.set_xlim(plot_lims[0:2])
            ax4.set_xlim(plot_lims[0:2])

            ax1.set_ylim(plot_lims[2:])
            ax2.set_ylim(plot_lims[2:])
            ax3.set_ylim(plot_lims[2:])
            ax4.set_ylim(plot_lims[2:])


def align_spectra(source_list, ref_source = None, xlow=0, xhigh=-1):
    '''
    Align each trace to a reference trace with cross correlation. 
    '''
    for i in np.arange(0,len(source_list)):
        for j in range(4):
            new_trace = copy.deepcopy(source_list[i].trace_spectra[j,1,xlow:xhigh])

            if ref_source == None:
                ref = source_list[0].trace_spectra[j,1,xlow:xhigh]
                len0 = np.size(source_list[0].trace_spectra[j,1,:])
            else:
                ref = ref_source.trace_spectra[j,1,xlow:xhigh]
                len0 = np.size(source_list[0].trace_spectra[j,1,:])

            corr = fftconvolve(np.nan_to_num(ref/np.nanmax(ref)), np.nan_to_num((new_trace/np.nanmax(new_trace))))

            # shift_size = np.nanargmax(corr) - len(ref) +1
            shift_size = (np.nanargmax(corr) - len0)/2

            source_list[i].trace_spectra[j,1,:] = shift(source_list[i].trace_spectra[j,1,:], -shift_size)

def get_angles_widths_from_list(filelist, data_dir = '', source_number = 0):
    """
    Go through the list of calibrated and extracted files and read out angles for given source_number
    """
    angles = []
    widths = []

    filelist = asci.read(filelist, format = 'no_header')['col1'] 

    for j in filelist:
        hdulist = fits.open(data_dir+j)
        i = source_number
        try:
            widths += [np.fromstring(hdulist[(2*i)+2].header["WIDTHS"][1:-1], sep = ' ')]
            angles += [np.fromstring(hdulist[(2*i)+2].header["ANGLES"][1:-1], sep = ' ')]
        except:
            print('Widths or angles not available')
    return np.array(widths), np.array(angles)

#POLARIZATION CALCULATION HELPER
# q, u, q_err, u_err, q_position, u_position = compute_qu(spec1, spec2, HWP1, HWP2)
#helper function to compute q and u given two spectra cubes
def compute_qu(spec1, spec2, HWP1, HWP2, run_alignment = True):
    """
    compute_qu is a helper function that takes two spectral cubes, each with the dimensions of (4,3,spec_pix)
    with two orthogonal HWP angles (0 and 45 or 22.5 and 67.5), then compute q and u
    Inputs:
        spec1, spec2: spectra cubes, each with the dimensions of (4,3,spec_pix). 
                        First index is the 4 spectra in one WIRC image.
                        Second index is (wavelength, flux, flux_error)
                        Last index is the spectral pixel direction
        HWP1, HWP2: half wave plate angles for spec1 and spec2 respectively. We need HWP2-HWP1 to be 45 deg (orthogonal)
        run_alignment: booleen indicating whether to run align_spectral_cube and scale_and_combine_spectra. Default is True
    Output:
        q, u: normalized stokes vectors q and u where q, u corresponds to polarization along 0 and 45 degrees respectively
        q_err, u_err: associated uncertainties
        q_ind, u_ind: indices of frames used to compute q and u. This is provided so we can check the results. 
    """ 
    #stack spectra
    # if spec1.shape != spec2.shape:
    if ((round(HWP1,2) - round(HWP2,2))%45) >0.01: #add some tolerance
        print(np.abs((HWP1 - HWP2)%45))
        print("Error, halfwave plate angles (%f, %f) are not orthogonal."%(HWP1,HWP2))
        return None
    else:
        spec_cube = np.stack([spec1, spec2]) #This has to be the same shape
        #align and scale cubes
        if run_alignment:
            aligned_cube = su.align_spectral_cube(spec_cube)
            scaled_cube = su.scale_and_combine_spectra(aligned_cube, return_scaled_cube = True)
            scaled_cube = aligned_cube
        else:
            scaled_cube = spec_cube

        #polarization vector and uncertainty. This is (spec1-spec2)/(spec1+spec2)
        pol_vec = (scaled_cube[0,:,1,:] - scaled_cube[1,:,1,:])/(scaled_cube[0,:,1,:] + scaled_cube[1,:,1,:])
        pol_err = (2/(scaled_cube[0,:,1,:] + scaled_cube[1,:,1,:])**2) * np.sqrt(  (scaled_cube[0,:,1,:]*scaled_cube[1,:,2,:])**2 + (scaled_cube[0,:,2,:]* scaled_cube[1,:,1,:])**2)

        #now determine which is which
        sampling_angles_0 = np.array([135, 45, 0, 90]) #THIS IS FROM UL, LR, UR, LL = U-, U+, Q-, Q+ as determined from twilight. 
        sampling_angles_1 = (sampling_angles_0 + 2*(HWP1))%180 #angles are mod 180 deg.  
        sampling_angles_2 = (sampling_angles_0 + 2*(HWP2))%180 #angles are mod 180 deg. 
        signs = np.sign(sampling_angles_2 - sampling_angles_1) # 0 - 45 is +q, 22.5 - 67.5 is +u

        #q's are those with sampling_angles_1 = 0 or 90 and sampling_angles_2 = 90 or 0
        q_ind = np.where(np.logical_or(sampling_angles_1 == 0, sampling_angles_1 == 90))
        u_ind = np.where(np.logical_or(sampling_angles_1 == 45, sampling_angles_1 == 135))

        #print(HWP1, HWP2, sampling_angles_1, sampling_angles_2, q_ind, u_ind)
        # print(signs)
        # print(signs[list(q_ind[0])])
        # print('q shape is ',pol_vec[q_ind[0]].shape)
        q_sign = signs[q_ind[0]]
        u_sign = signs[u_ind[0]]
        q =  pol_vec[q_ind[0]]*q_sign[:,None] 
        u =  pol_vec[u_ind[0]]*u_sign[:,None] 
        q_err =  pol_err[list(q_ind[0])] 
        u_err =  pol_err[list(u_ind[0])] 
        # print(q.shape, q_err.shape)
        return q, u, q_err, u_err, q_ind[0], u_ind[0]

def group_HWP(HWP_set):
    """
    Helper function to compute_qu_for_obs_sequence. It groups given list of HWP angles into sets of two orthogonal observations. 

    Input: HWP_set, a vector of all half wave plate angles in the observing compute_qu_for_obs_sequence
    Output: Two arrays each for sets of 0/45 deg and another for 22.5/67.5. compute_qu_for_obs_sequence can then use this info to call 
            compute_qu to compute qu for each appropriate pair.  
    """
    # #HWP_index determine which pair is q and which is u. If HWP = 0 or 45, HWP_ind = 0; if HWP = 22.5, 67.5, HWP_ind = 1
    # #So HWP_ind = 0; LL, UR is q, LR, UL is u. Flipped for HWP_ind = 1
    # HWP_ind = (HWP_set//22.5)%2

    # group_0 = HWP_set[HWP_ind == 0]
    # group_1 = HWP_set[HWP_ind == 1]
    # #initialize the sets of observations
    # set_0 = []
    # set_1 = []
    # holding_0_ind = []
    # holding_1_ind = []
    # for i, HWP in enumerate(group_0):

    set_0 = np.where(HWP_set == 0)
    set_225 = np.where(HWP_set == 22.5)
    set_45 = np.where(HWP_set == 45)
    set_675 = np.where(HWP_set == 67.5)

    pairs_0 = np.stack([set_0[0], set_45[0]], axis = 1) #This is an array with shape (N/4, 2), each element is 2 indices of best 0, 45 pair. 
    pairs_225 = np.stack([set_225[0], set_675[0]], axis = 1)

    return pairs_0, pairs_225

def compute_qu_for_obs_sequence(spectra_cube, HWP_set, HWP_offset = 0, run_alignment = True):
    """
    This function takes a set of aligned spectra along with a set of HWP angles, both with the same length, 
    and call compute_qu to measure polarization q and u. 

    Input:
        obs_set: a spectral cube with shape (N, 4, 3, spec_pix) where N is the number of frames. Each element is the 4 spectra from single image. 
        HWP_set: a vector of length N, prescribing the half wave plate angle for each of the frame in obs_set. Values should be 0, 45, 22.5, 67.5 for double diff. 
                 if there is an offset from this orthogonal set, indicae so in HWP_offset
        HWP_offset: a float indicating the zeropoint of the HWP angle. We proceed with HWP_set - HWP_offset.

    Output:
        q, q_err, u, u_err **currently single differencing in time. Can do double difference manually afterward. This may change. 
    """
    #First, check length
    if spectra_cube.shape[0] != len(HWP_set):
        raise ValueError("Lengths of spectra_cube and HWP_set are not equal.")

    #Apply HWP_offset
    HWP_final = HWP_set - HWP_offset
    #check if the values are good. 
    all_ang = set([0,45,22.5,67.5])
    if set(HWP_final) != all_ang:
        raise ValueError("HWP set doesn't have all 4 angles or have wrong angles: %s"%str(set(HWP_final)))

    #Arrange the sequence into best pairs of 0/45 and 22.5/67.5 to compute qu
    pairs_0, pairs_225 = group_HWP(HWP_final)

    #First deal with observations with HWP angles 0/45. Go through the list and compute q and u for each pair

    all_q0 = []
    all_u0 = []
    all_qerr0 = []
    all_uerr0 = []
    all_qind0 = []
    all_uind0 = []

    for i in pairs_0:
        q, u, q_err, u_err, q_ind, u_ind = compute_qu(spectra_cube[i[0]], spectra_cube[i[1]], HWP_final[i[0]], HWP_final[i[1]], run_alignment)
        all_q0    += [q]
        all_u0    += [u]
        all_qerr0 += [q_err]
        all_uerr0 += [u_err]
        all_qind0 += [q_ind]
        all_uind0 += [u_ind]

    #Now deal with observations with HWP angles 22.5/67.5. 

    all_q225 = []
    all_u225 = []
    all_qerr225 = []
    all_uerr225 = []
    all_qind225 = []
    all_uind225 = []

    for i in pairs_225:
        q, u, q_err, u_err, q_ind, u_ind = compute_qu(spectra_cube[i[0]], spectra_cube[i[1]], HWP_final[i[0]], HWP_final[i[1]], run_alignment)
        all_q225    += [q]
        all_u225    += [u]
        all_qerr225 += [q_err]
        all_uerr225 += [u_err]
        all_qind225 += [q_ind]
        all_uind225 += [u_ind]


    all_q       = np.array(all_q0 + all_q225    )
    all_u       = np.array(all_u0 + all_u225     )
    all_qerr   = np.array(all_qerr0 + all_qerr225   )
    all_uerr   = np.array(all_uerr0 + all_uerr225   )
    all_qind   = np.array(all_qind0 + all_qind225   )
    all_uind   = np.array(all_uind0 + all_uind225   )

    return all_q, all_u, all_qerr, all_uerr, all_qind, all_uind


def find_best_background(list_of_headers, separation_threshold = 2):
    """
    find_best_background takes a list of headers from WIRC+Pol observations and find best background frame for each element. 
    Here are the conditions: 
        Same HWP angle
        With telescope offset greater than 'separation_threshold' (default at 2 arcsec)
        Closest in time
        Not already used by another frame (this condition is relaxed if every frame is used up. Say we have extra set of exposures at position A)

    Input:
        list_of_headers: a list of fits headers of the observations. 
        separation_threshold: how far away, in arcsec, the background frame is required to be from the current frame. 

    Output: 
        list of the same length of list_of_headers giving the index of the best background for each frame in list_of_headers
        example: if the best background for file wirc0001 is wirc0005, then best_bkg[1] = 5 
    """
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    import astropy.units as u 
    #closest in time, some distance away, same HWP
    sep_threshold = separation_threshold 
    all_hdr = list_of_headers
    #get some useful quantities
    coords = np.array([ SkyCoord(x['RA'], x['DEC'], unit = (u.hourangle, u.deg)) for x in all_hdr ])
    names = np.array([x['RAW_FN'] for x in all_hdr])
    hwps = np.array([x['HWP_ANG'] for x in all_hdr])
    times = np.array([Time(x['UTSHUT'], format = 'isot') for x in all_hdr])

    #array to keep track of best background frame
    best_bkgs = np.array( [None]*len(all_hdr) )
    #array to keep track of whether this index is already in a pair 
    already_in_pair = np.zeros(len(all_hdr))

    for i in range(len(all_hdr)):
        if best_bkgs[i] is None:
    #         print(i)
            all_dist = np.array([ (coords[i].separation(x)).arcsec for x in coords ])
            far_enough = all_dist > sep_threshold
            same_hwp = hwps == hwps[i]

            all_good = np.logical_and(far_enough, same_hwp)


        
            #are there good background that is not already in a pair?
            not_in_pair = already_in_pair == 0
            if np.any(not_in_pair[all_good]): #if there're some frames available not already in a pair, use those first
                all_good = np.logical_and(all_good, not_in_pair)
            #otherwise, just accept the repeat
     

            #time difference
            t0 = Time(all_hdr[i]['UTSHUT'], format = 'isot')
            time_diff = np.array([ np.abs((x - t0).value) for x in times[all_good]])

            #Use minimal time difference
            best_bkg = names[all_good][np.where(time_diff == np.min(time_diff))[0]]
            


            #Save this in best_bkg array
            best_bkgs[i] = np.where( names == best_bkg[0] )[0][0]
            
            #remember that this is already in an AB pair
            already_in_pair[i] = 1
            
            #But also reciprocate, so it's neatly in a pair
            if best_bkgs[np.where(names == best_bkg )[0]][0] is None:
                #print(names[np.where(names == best_bkg )[0]])
                best_bkgs[np.where(names == best_bkg )[0]] = np.where(names == names[i])[0][0]
                #also remember that this frame is already in a pair
                already_in_pair[np.where(names == best_bkg )[0]] = 1
            
    return best_bkgs.astype('int')     
    
    #     print(all_dist)







        