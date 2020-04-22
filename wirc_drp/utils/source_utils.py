from wirc_drp.constants import *
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.ndimage import shift
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve 
from astropy.io import ascii as asci
from astropy.io import fits 
from wirc_drp.utils import spec_utils as su
from wirc_drp.utils.image_utils import findTrace
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl


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
def compute_qu(spec1, spec2, HWP1, HWP2, run_alignment = True, method = 'flux_ratio', sign = '-'):
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
        method: Either 'flux_ratio' or 'double_difference'. 
    Output:
        q, u: normalized stokes vectors q and u where q, u corresponds to polarization along 0 and 45 degrees respectively
        q_err, u_err: associated uncertainties
        q_ind, u_ind: indices of frames used to compute q and u. This is provided so we can check the results. 
    """ 
    #If method is neither 'flux_ratio' nor 'double_difference', revert to 'flux_ratio'
    if method not in ['flux_ratio','double_difference']:
        print("method has to be either flux_ratio or double_difference. not %s. revert to flux_ratio"%method)

    #stack spectra
    # if spec1.shape != spec2.shape:
    if ((round(HWP1,2) - round(HWP2,2))%45) >0.01: #add some tolerance
        # print(np.abs((HWP1 - HWP2)%45))
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

        if method == 'double_difference':
            #polarization vector and uncertainty. This is (spec1-spec2)/(spec1+spec2)
            pol_vec = (scaled_cube[0,:,1,:] - scaled_cube[1,:,1,:])/(scaled_cube[0,:,1,:] + scaled_cube[1,:,1,:])
            pol_err = (2/(scaled_cube[0,:,1,:] + scaled_cube[1,:,1,:])**2) * np.sqrt(  (scaled_cube[0,:,1,:]*scaled_cube[1,:,2,:])**2 + (scaled_cube[0,:,2,:]* scaled_cube[1,:,1,:])**2)

            #now determine which is which
            sampling_angles_0 = np.array([135, 45, 90,0]) #THIS IS FROM UL, LR, UR, LL = U-, U+, Q-, Q+ as determined from twilight. 
            if sign == '-':
                sampling_angles_1 = (sampling_angles_0 - 2*(HWP1))%180 #angles are mod 180 deg.  
                sampling_angles_2 = (sampling_angles_0 - 2*(HWP2))%180 #angles are mod 180 deg. 
            elif sign == '+':
                sampling_angles_1 = (sampling_angles_0 + 2*(HWP1))%180 #angles are mod 180 deg.  
                sampling_angles_2 = (sampling_angles_0 + 2*(HWP2))%180 #angles are mod 180 deg. 
            signs = np.sign(sampling_angles_2 - sampling_angles_1) # 0 - 45 is +q, 22.5 - 67.5 is +u
            # print(sampling_angles_1)
            # print(sampling_angles_2)
            # print(signs)
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

        elif method == 'flux_ratio':
            #First, figure out the sampling angles
            sampling_angles_0 = np.array([135, 45, 90, 0]) #THIS IS FROM UL, LR, UR, LL = U-, U+, Q-, Q+ as determined from twilight. 
            # sampling_angles_0 = np.array([0, 90, 45, 135])
            if sign == '-':
                sampling_angles_1 = (sampling_angles_0 - 2*(HWP1))%180 #angles are mod 180 deg.  
                sampling_angles_2 = (sampling_angles_0 - 2*(HWP2))%180 #angles are mod 180 deg. 
            elif sign=='+':
                sampling_angles_1 = (sampling_angles_0 + 2*(HWP1))%180 #angles are mod 180 deg.  
                sampling_angles_2 = (sampling_angles_0 + 2*(HWP2))%180 #angles are mod 180 deg.    
            # print(HWP1, HWP2)             
            # print(sampling_angles_1)
            # print(sampling_angles_2)
            #indices (non elegant solution...)
            ind0_0 =   np.where(sampling_angles_1 == 0)[0]
            ind0_90 =  np.where(sampling_angles_1 == 90)[0]
            ind0_45 =  np.where(sampling_angles_1 == 45)[0]
            ind0_135 = np.where(sampling_angles_1 == 135)[0]
            ind1_0 =   np.where(sampling_angles_2 == 0)[0]
            ind1_90 =  np.where(sampling_angles_2 == 90)[0]
            ind1_45 =  np.where(sampling_angles_2 == 45)[0]
            ind1_135 = np.where(sampling_angles_2 == 135)[0]
            # print(ind0_0, ind0_90, ind0_45, ind0_135)
            # print(ind1_0, ind1_90, ind1_45, ind1_135)

            #q computation, 
            Rq_sq = (scaled_cube[0,ind0_0,1,:]/scaled_cube[0,ind0_90,1,:]) / (scaled_cube[1,ind1_90,1,:]/scaled_cube[1,ind1_0,1,:])
            Rq_sq_err = Rq_sq * np.sqrt( (scaled_cube[0,ind0_0,2,:]/scaled_cube[0,ind0_0,1,:])**2 +
                                         (scaled_cube[0,ind0_90,2,:]/scaled_cube[0,ind0_90,1,:])**2 +
                                         (scaled_cube[0,ind1_0,2,:]/scaled_cube[0,ind1_0,1,:])**2 +
                                         (scaled_cube[0,ind1_90,2,:]/scaled_cube[0,ind1_90,1,:])**2 )

            q = (np.sqrt(Rq_sq) - 1)/(np.sqrt(Rq_sq) + 1)
            q_err = Rq_sq_err /np.sqrt(Rq_sq)/(np.sqrt(Rq_sq)+1)**2 

            #u computation, 
            Ru_sq = (scaled_cube[0,ind0_45,1,:]/scaled_cube[0,ind0_135,1,:]) / (scaled_cube[1,ind1_135,1,:]/scaled_cube[1,ind1_45,1,:])
            Ru_sq_err = Ru_sq * np.sqrt( (scaled_cube[0,ind0_45,2,:]/scaled_cube[0,ind0_45,1,:])**2 +
                                         (scaled_cube[0,ind0_135,2,:]/scaled_cube[0,ind0_135,1,:])**2 +
                                         (scaled_cube[0,ind1_45,2,:]/scaled_cube[0,ind1_45,1,:])**2 +
                                         (scaled_cube[0,ind1_135,2,:]/scaled_cube[0,ind1_135,1,:])**2 )

            u = (np.sqrt(Ru_sq) - 1)/(np.sqrt(Ru_sq) + 1)
            u_err = Ru_sq_err /np.sqrt(Ru_sq)/(np.sqrt(Ru_sq)+1)**2 
            return q, u, q_err, u_err, None, None

        

def group_HWP(HWP_set):
    """
    Helper function to compute_qu_for_obs_sequence. It groups given list of HWP angles into sets of two orthogonal observations. 

    Input: HWP_set, a vector of all half wave plate angles in the observing compute_qu_for_obs_sequence
    Output: Two arrays each for sets of 0/45 deg and another for 22.5/67.5. compute_qu_for_obs_sequence can then use this info to call 
            compute_qu to compute qu for each appropriate pair.  
            Another array to record pairless indices. 
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

    small_0_45 = np.min( (len(set_0[0]), len(set_45[0])) )
    small_225_675 = np.min( (len(set_225[0]), len(set_675[0])))

    pairs_0 = np.stack([set_0[0][0:small_0_45], set_45[0][0:small_0_45]], axis = 1) #This is an array with shape (N/4, 2), each element is 2 indices of best 0, 45 pair. 
    pairs_225 = np.stack([set_225[0][0:small_225_675], set_675[0][0:small_225_675]], axis = 1)
    
    # pairs_0 = np.stack([set_0[0], set_45[0]], axis = 1) #This is an array with shape (N/4, 2), each element is 2 indices of best 0, 45 pair.
    # pairs_225 = np.stack([set_225[0], set_675[0]], axis = 1)

    #Report indices that didn't get paired up. 
    # args = np.arange(len(HWP_set))
    # paired = np.concatenate([pairs_0.flatten(), pairs_225.flatten()])
    # pair_less = np.setdiff1d(args, paired)
    return pairs_0, pairs_225#, pair_less

# def null_qu(HWP_set):
#     """
#     Helper function similar to group_HWP, but intentionally pair up observations with the same HWP 
#     so that the final computed qu is supposed to be zero. 
#     """
#     set_0 = np.where(HWP_set == 0)
#     set_225 = np.where(HWP_set == 22.5)
#     set_45 = np.where(HWP_set == 45)
#     set_675 = np.where(HWP_set == 67.5)

#     small_0_45 = np.min( (len(set_0[0]), len(set_45[0])) )
#     small_225_675 = np.min( (len(set_225[0]), len(set_675[0])))

#     pairs_0 = np.stack([set_0[0][0:small_0_45], set_45[0][0:small_0_45]], axis = 1) #This is an array with shape (N/4, 2), each element is 2 indices of best 0, 45 pair. 
#     pairs_225 = np.stack([set_225[0][0:small_225_675], set_675[0][0:small_225_675]], axis = 1)
    
#     # pairs_0 = np.stack([set_0[0], set_45[0]], axis = 1) #This is an array with shape (N/4, 2), each element is 2 indices of best 0, 45 pair.
#     # pairs_225 = np.stack([set_225[0], set_675[0]], axis = 1)
#     return pairs_0, pairs_225   

def compute_qu_for_obs_sequence(spectra_cube, HWP_set, HWP_offset = 0, run_alignment = True, method = 'flux_ratio', sign = '-'):
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
    #And check the computation method
    #If method is neither 'flux_ratio' nor 'double_difference', revert to 'flux_ratio'
    if method not in ['flux_ratio','double_difference']:
        print("method has to be either flux_ratio or double_difference. not %s. revert to flux_ratio"%method)

    #Apply HWP_offset
    HWP_final = HWP_set - HWP_offset
    #check if the values are good. 
    all_ang = set([0,45,22.5,67.5])
    if set(HWP_final) != all_ang:
        raise ValueError("HWP set doesn't have all 4 angles or have wrong angles: %s"%str(set(HWP_final)))

    #Arrange the sequence into best pairs of 0/45 and 22.5/67.5 to compute qu
    pairs_0, pairs_225 = group_HWP(HWP_final)
    # print( pairs_0, pairs_225)
    #First deal with observations with HWP angles 0/45. Go through the list and compute q and u for each pair

    #Get rid of observations that have no pair, print report. 
    # spectra_cube_with_pair = 

    all_q0 = []
    all_u0 = []
    all_qerr0 = []
    all_uerr0 = []
    all_qind0 = []
    all_uind0 = []

    ind_of_obs = [] #This record the index of observations

    for i in pairs_0:
        q, u, q_err, u_err, q_ind, u_ind = compute_qu(spectra_cube[i[0]], spectra_cube[i[1]], \
                                                        HWP_final[i[0]], HWP_final[i[1]], run_alignment, method = method, sign = sign)
        if method == 'flux_ratio':
            all_q0    += [q[0]]
            all_u0    += [u[0]]
            all_qerr0 += [q_err[0]]
            all_uerr0 += [u_err[0]]
            all_qind0 += [0]
            all_uind0 += [1]
        elif method == 'double_difference':
            all_q0    += [(q[0]+q[1])/2]
            all_u0    += [(u[0]+u[1])/2]
            all_qerr0 += [(q_err[0]+q_err[1])/(2*np.sqrt(2))]
            all_uerr0 += [(u_err[0]+u_err[1])/(2*np.sqrt(2))]
            all_qind0 += [0]
            all_uind0 += [1]

        ind_of_obs += [np.mean(i)] #average between the two indices of the observations is recorded

    #Now deal with observations with HWP angles 22.5/67.5. 

    all_q225 = []
    all_u225 = []
    all_qerr225 = []
    all_uerr225 = []
    all_qind225 = []
    all_uind225 = []

    for i in pairs_225:
        q, u, q_err, u_err, q_ind, u_ind = compute_qu(spectra_cube[i[0]], spectra_cube[i[1]], \
                                                        HWP_final[i[0]], HWP_final[i[1]], run_alignment, method = method, sign = sign)
        # print(q[0].shape)
        if method == 'flux_ratio':
            all_q225    += [q[0]]
            all_u225    += [u[0]]
            all_qerr225 += [q_err[0]]
            all_uerr225 += [u_err[0]]
            all_qind225 += [1]
            all_uind225 += [0]
        elif method == 'double_difference':
            all_q225    += [(q[0]+q[1])/2]
            all_u225    += [(u[0]+u[1])/2]
            all_qerr225 += [(q_err[0]+q_err[1])/(2*np.sqrt(2))]
            all_uerr225 += [(u_err[0]+u_err[1])/(2*np.sqrt(2))]
            all_qind225 += [1]
            all_uind225 += [0]

        ind_of_obs += [np.mean(i)] #average between the two indices of the observations is recorded


    #Original:
    # all_q       = np.array(all_q0 + all_q225    )
    # all_u       = np.array(all_u0 + all_u225     )
    # all_qerr   = np.array(all_qerr0 + all_qerr225   )
    # all_uerr   = np.array(all_uerr0 + all_uerr225   )
    # all_qind   = np.array(all_qind0 + all_qind225   )
    # all_uind   = np.array(all_uind0 + all_uind225   )
    # #import pdb; pdb.set_trace()
    #Now we want to return to the original measurement order, so we interleave the two sets: 
    #For now we do this by finding the indexes of the 0 and 22.5 waveplate positions. 
    #If there's a screwy HWP Sequence this will get messed up. 
#     first_inds = np.where(HWP_set ==0 )[0]//2
#     second_inds = np.where(HWP_set == 22.5 )[0]//2
    
#    # import pdb;pdb.set_trace()
#     all_q = np.empty((np.shape(all_q0)[0]+np.shape(all_q225)[0],np.shape(all_q0)[1]))
#     all_q[first_inds] = all_q0
#     all_q[second_inds] = all_q225

#     all_u = np.empty((np.shape(all_u0)[0]+np.shape(all_u225)[0],np.shape(all_u0)[1]))
#     all_u[first_inds] = all_u0
#     all_u[second_inds] = all_u225

#     all_qerr = np.empty((np.shape(all_qerr0)[0]+np.shape(all_qerr225)[0],np.shape(all_qerr0)[1]))
#     all_qerr[first_inds] = all_qerr0
#     all_qerr[second_inds] = all_qerr225

#     all_uerr = np.empty((np.shape(all_uerr0)[0]+np.shape(all_uerr225)[0],np.shape(all_uerr0)[1]))
#     all_uerr[first_inds] = all_uerr0
#     all_uerr[second_inds] = all_uerr225

#     #These next few lines are probably overkill, but copying and pasting is easier than thinking. 
#     all_qind = np.empty((np.shape(all_qind0)[0]+np.shape(all_qind225)[0]))
#     all_qind[first_inds] = all_qind0
#     all_qind[second_inds] = all_qind225   

#     all_uind = np.empty((np.shape(all_uind0)[0]+np.shape(all_uind225)[0]))
#     all_uind[first_inds] = all_uind0
#     all_uind[second_inds] = all_uind225  

    #Instead of assuming regular HWP sequence, we will sort using ind_of_obs
    sort_by_obs = (np.array(ind_of_obs)).argsort()

    # print(ind_of_obs)
    # print(sort_by_obs)

    #add results from all_*0 and all_*225 together, then sort by the observation indices. 
    all_q      = np.array(all_q0    + all_q225      )[sort_by_obs] 
    all_u      = np.array(all_u0    + all_u225      )[sort_by_obs]
    all_qerr   = np.array(all_qerr0 + all_qerr225   )[sort_by_obs]
    all_uerr   = np.array(all_uerr0 + all_uerr225   )[sort_by_obs]
    all_qind   = np.array(all_qind0 + all_qind225   )[sort_by_obs]
    all_uind   = np.array(all_uind0 + all_uind225   )[sort_by_obs]


    return all_q, all_u, all_qerr, all_uerr, all_qind, all_uind






def find_best_background(list_of_files, separation_threshold = 2, verbose = False):
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
    from astropy.io import fits
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    import astropy.units as u 
    list_of_headers = []
    for i in list_of_files:
        list_of_headers += [fits.getheader(i)]
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
            if verbose:
                print(list_of_files[i])
            all_dist = np.array([ (coords[i].separation(x)).arcsec for x in coords ])
            far_enough = all_dist > sep_threshold
            same_hwp = hwps == hwps[i]

            if verbose:
                print(all_dist)
                print(same_hwp)

            all_good = np.logical_and(far_enough, same_hwp)


        
            #are there good background that is not already in a pair?
            not_in_pair = already_in_pair == 0
            if np.any(not_in_pair[all_good]): #if there're some frames available not already in a pair, use those first
                all_good = np.logical_and(all_good, not_in_pair)
            #otherwise, just accept the repeat
     

            #time difference
            t0 = Time(all_hdr[i]['UTSHUT'], format = 'isot')
            time_diff = np.array([ np.abs((x - t0).value) for x in times[all_good]])

            if verbose:
                print(time_diff)

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

def compute_p_and_pa( q, u, q_err, u_err):
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
    p_corr = np.sqrt(p**2 - dp**2) #debiased deg of polarization
    
    #Compute angle of polarization
    theta = 1/2.*np.arctan2(u,q)
    dtheta = 1/(2*p**2) * np.sqrt( (q * u_err)**2 + (u * q_err)**2)
    
    return p, p_corr, dp, theta, dtheta

def serkowski_polarization(wl, wl_max, p_max, K, theta = None):
    """Compute the polarization spectrum expected from ISP Serkowski law
    p_serk = p_max * exp(-K ln^2(wl_max/wl))
    """
    p_spec = p_max * np.exp( -K * (np.log(wl_max/wl))**2)
    if theta is None:
        return p_spec 
    else:
        return p_spec, p_spec*np.cos(2*np.radians(theta)), p_spec*np.sin(2*np.radians(theta)) 

def plot_pol_summary(master_wvs,spec,q,u,qerr,uerr,qinds=None,uinds=None,mode='mean',xlow=1.15,xhigh=1.325,
    ylow=-0.02,yhigh=0.02,target_name="",date="19850625",t_ext = 0,binsize=1,theta_wrap=180,
    ldwarf=False,show=True,save_path=None,legend_loc ="bottom left",all_theta=False,
    fig = None, axes = None,filename=None,figsize=(16,20),title=None,tdwarf=False):
    '''
    Make a summary plot of polarization. The formatting assumes that the inputs (q,u,qerr,uerr)
    are the output of compute_qu_for_obs_sequence. 

    Inputs:
    mode      - Either "mean" or "median"
    fig, axes - to plot on existing figure/axes. None if not. 
    '''

    #First calculate the double_difference values
    # import pdb; pdb.set_trace()
    q_dd = copy.deepcopy(q) #Doing it this way because the way q and u calculated changed. 
    u_dd = copy.deepcopy(u)
    # q_dd = np.nanmean(q,axis=1)

    good_wvs = (master_wvs > 1.175) & (master_wvs < 1.325)
    q_dd = q_dd[:,good_wvs]
    u_dd = u_dd[:,good_wvs]
    qerr = qerr[:,good_wvs]
    uerr = uerr[:,good_wvs]
    wvs = master_wvs[good_wvs]
    spec = spec[good_wvs]
    # u_dd = np.nanmean(u,axis=1)
   # q_dd = q
   # u_dd = u
    p_dd = np.sqrt(q_dd**2+u_dd**2)
    theta_dd = 0.5*np.degrees(np.arctan2(u_dd,q_dd))
    theta_dd[theta_dd < 0] +=180
   # q_dd = q
   # u_dd = u

    #Doing this because of how things changed in compute polarization

    q_dd_err = copy.deepcopy(qerr)
    u_dd_err = copy.deepcopy(uerr)

    #This was the old way. 
    # q_dd_err = np.sqrt(np.sum((qerr**2),axis=1))/qerr.shape[1]
    # u_dd_err = np.sqrt(np.sum((uerr**2),axis=1))/uerr.shape[1]

    #Now calculate the mean or median
    from astropy import stats
    
    q_mean = np.zeros([q_dd.shape[1]]) #We name this mean, though it could either be Mean or Median
    q_std = np.zeros([q_dd.shape[1]])
    u_mean = np.zeros([u_dd.shape[1]])
    u_std = np.zeros([u_dd.shape[1]])

    if qinds is not None:
        q_mean_pair1 = np.zeros([q_dd.shape[1]])
        q_mean_pair2 = np.zeros([q_dd.shape[1]])
        q_std_pair1 = np.zeros([q_dd.shape[1]])
        q_std_pair2 = np.zeros([q_dd.shape[1]])

    if uinds is not None:
        u_mean_pair1 = np.zeros([u_dd.shape[1]])
        u_mean_pair2 = np.zeros([u_dd.shape[1]])
        u_std_pair1 = np.zeros([u_dd.shape[1]])
        u_std_pair2 = np.zeros([u_dd.shape[1]])


    for i in range(q_dd.shape[1]):

        mn,md,std = stats.sigma_clipped_stats(q_dd[:,i], sigma=3, maxiters=5)
        if mode == 'median':
            q_mean[i] = md
        else:
            q_mean[i] = mn

        q_std[i] = std/np.sqrt(q_dd.shape[0])
        
        if qinds is not None:
            mn,md,std = stats.sigma_clipped_stats(q_dd[qinds==0,i][q_dd[qinds==0,i] == q_dd[qinds==0,i]], sigma=3, maxiters=5)
            if mode == 'median':
                q_mean_pair1[i] = md
            else:
                q_mean_pair1[i] = mn
            q_std_pair1[i] = std/np.sqrt(q_dd[qinds==1].shape[0])

            # import pdb;pdb.set_trace()

            mn,md,std = stats.sigma_clipped_stats(q_dd[qinds==1,i][q_dd[qinds==1,i]==q_dd[qinds==1,i]], sigma=3, maxiters=5)
            if mode == 'median':
                q_mean_pair2[i] = md
            else:
                q_mean_pair2[i] = mn
            q_std_pair2[i] = std/np.sqrt(q_dd[qinds==1].shape[0])

        mn,md,std = stats.sigma_clipped_stats(u_dd[:,i], sigma=3, maxiters=5)
        u_std[i] = std/np.sqrt(q_dd.shape[0])
        if mode == 'median':
            u_mean[i] = md
        else:
            u_mean[i] = mn

        if uinds is not None:
            mn,md,std = stats.sigma_clipped_stats(u_dd[uinds==0,i][u_dd[uinds==0,i] == u_dd[uinds==0,i]], sigma=3, maxiters=5)
            if mode == 'median':
                u_mean_pair1[i] = md
            else:
                u_mean_pair1[i] = mn
            u_std_pair1[i] = std/np.sqrt(u_dd[uinds==0].shape[0])

            mn,md,std = stats.sigma_clipped_stats(u_dd[uinds==1,i][u_dd[uinds==1,i] == u_dd[uinds==1,i]], sigma=3, maxiters=5)
            if mode == 'median':
                u_mean_pair2[i] = md
            else:
                u_mean_pair2[i] = mn
            u_std_pair2[i] = std/np.sqrt(u_dd[uinds==1].shape[0])



    p_mean = np.sqrt(q_mean**2+u_mean**2)
    theta_mean = 0.5*np.degrees(np.arctan2(u_mean,q_mean))
    theta_mean[theta_mean < theta_wrap] +=180

    q_mean_err = np.sqrt(np.sum(q_dd_err**2,axis=0))/q_dd_err.shape[0]
    u_mean_err = np.sqrt(np.sum(u_dd_err**2,axis=0))/q_dd_err.shape[0]
    p_mean_err = np.sqrt(q_mean**2*q_mean_err**2+u_mean**2*u_mean_err**2)/p_mean
    p_std = np.sqrt(q_mean**2*q_std**2+u_mean**2*u_std**2)/p_mean
    theta_mean_err = 0.5*np.degrees( np.sqrt( (u_mean**2*q_mean_err**2+q_mean**2*u_mean_err**2)/(q_mean**2+u_mean**2)**2))
    theta_std = 0.5*np.degrees( np.sqrt( (u_mean**2*q_std**2+q_mean**2*u_std**2)/(q_mean**2+u_mean**2)**2))

    ### Implement Binning
    if binsize != 1:
        snip = q_mean.shape[0] % binsize
        if snip != 0:
            q_mean = np.mean(q_mean[:-snip].reshape(-1,binsize),axis=1)
            u_mean = np.mean(u_mean[:-snip].reshape(-1,binsize),axis=1)
            wvs = np.mean(wvs[:-snip].reshape(-1,binsize),axis=1)
            p_mean = np.sqrt(q_mean**2+u_mean**2)
            theta_mean = 0.5*np.degrees(np.arctan2(u_mean,q_mean))
            if spec is not None:
                spec = np.mean(spec[:-snip].reshape(-1,binsize),axis=1)
            
            if qinds is not None:
                q_mean_pair1 = np.mean(q_mean_pair1[:-snip].reshape(-1,binsize),axis=1)
                q_mean_pair2 = np.mean(q_mean_pair2[:-snip].reshape(-1,binsize),axis=1)
                q_std_pair1 = np.sqrt(np.sum(q_std_pair1[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
                q_std_pair2 = np.sqrt(np.sum(q_std_pair2[:-snip].reshape(-1,binsize)**2,axis=1))/binsize

            if uinds is not None:
                u_mean_pair1 = np.mean(u_mean_pair1[:-snip].reshape(-1,binsize),axis=1)
                u_mean_pair2 = np.mean(u_mean_pair2[:-snip].reshape(-1,binsize),axis=1)
                u_std_pair1 = np.sqrt(np.sum(u_std_pair1[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
                u_std_pair2 = np.sqrt(np.sum(u_std_pair2[:-snip].reshape(-1,binsize)**2,axis=1))/binsize

            q_mean_err = np.sqrt(np.sum(q_mean_err[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
            q_std = np.sqrt(np.sum(q_std[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
            u_mean_err = np.sqrt(np.sum(u_mean_err[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
            u_std = np.sqrt(np.sum(u_std[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
            p_mean_err = np.sqrt(q_mean**2*q_mean_err**2+u_mean**2*u_mean_err**2)/p_mean
            theta_mean_err = 0.5*np.degrees( np.sqrt( (u_mean**2*q_mean_err**2+q_mean**2*u_mean_err**2)/(q_mean**2+u_mean**2)**2))
            
        
        else: 
            q_mean = np.mean(q_mean.reshape(-1,binsize),axis=1)
            u_mean = np.mean(u_mean.reshape(-1,binsize),axis=1)
            wvs = np.mean(wvs.reshape(-1,binsize),axis=1)
            p_mean = np.sqrt(q_mean**2+u_mean**2)
            theta_mean = 0.5*np.degrees(np.arctan2(u_mean,q_mean))
            # theta_bin[theta_bin < 0] +=180
            if spec is not None:
                spec = np.mean(spec.reshape(-1,binsize),axis=1)

            q_mean_err = np.sqrt(np.sum(q_mean_err.reshape(-1,binsize)**2,axis=1))/binsize
            q_std = np.sqrt(np.sum(q_std.reshape(-1,binsize)**2,axis=1))/binsize
            u_mean_err = np.sqrt(np.sum(u_mean_err.reshape(-1,binsize)**2,axis=1))/binsize
            u_std = np.sqrt(np.sum(u_std.reshape(-1,binsize)**2,axis=1))/binsize
            p_mean_err = np.sqrt(q_mean**2*q_mean_err**2+u_mean**2*u_mean_err**2)/p_mean
            theta_mean_err = 0.5*np.degrees( np.sqrt( (u_mean**2*q_mean_err**2+q_mean**2*u_mean_err**2)/(q_mean**2+u_mean**2)**2))
    p_std = np.sqrt(q_mean**2*q_std**2+u_mean**2*u_std**2)/p_mean
    theta_std = 0.5*np.degrees( np.sqrt( (u_mean**2*q_std**2+q_mean**2*u_std**2)/(q_mean**2+u_mean**2)**2))
    
    #Wrap theta about 180
    theta_mean[theta_mean>theta_wrap] -= 180

    ##Calculate the mean values
    low = 65
    high = 135

    inds = (wvs > 1.165) & (wvs<1.31)
    mn_q = np.mean(q_mean[inds])
    mn_u = np.mean(u_mean[inds])
    mn_p = np.mean(p_mean[inds])

    mn_q_err = np.mean(q_mean_err[inds])
    mn_u_err = np.mean(u_mean_err[inds])
    mn_p_err = np.mean(p_mean_err[inds])

    std_q = np.std(q_mean[inds])
    std_u = np.std(u_mean[inds])
    std_p = np.std(p_mean[inds])

    ### Make the plot!!! ###
    if fig is None and axes is None:
        fig,axes = plt.subplots(3,2,figsize=figsize)

    axes[2,1] = plt.subplot(3,2,6, projection='polar')
    ##### Plot Q, U, P and theta ######
    #The mean values
    if mode == "median":
        axes[0,0].plot(wvs,q_mean,'k',label="Median",zorder=10)
    else:
        axes[0,0].plot(wvs,q_mean,'k',label="Mean",zorder=10)    
    axes[0,1].plot(wvs,u_mean,'k',zorder=10)       
    axes[1,0].plot(wvs,p_mean,'k',zorder=10)    

    if qinds is not None:
        axes[0,0].plot(wvs,q_mean)
    
    #### TEMPORARY! ####
    # np.save("p",p_mean)
    # np.save("wvs",wvs)
    # np.save("perr",p_std)

    #Make a line at zero
    axes[0,0].axhline(0.,color='r',linestyle='--')
    axes[0,1].axhline(0.,color='r',linestyle='--')

    #Fill in photon/ron error ranges
    axes[0,0].fill_between(wvs,q_mean+q_mean_err,q_mean-q_mean_err,color='k',alpha=0.1,label="Propagated Noise",zorder=10)    
    axes[0,1].fill_between(wvs,u_mean+u_mean_err,u_mean-u_mean_err,color='k',alpha=0.1,zorder=10)    
    axes[1,0].fill_between(wvs,p_mean+p_mean_err,p_mean-p_mean_err,color='k',alpha=0.1,zorder=10)    
    # axes[1,1].fill_between(wvs,theta_mean+theta_mean_err,theta_mean-theta_mean_err,color='k',alpha=0.1)
    if qinds is not None:
        # import pdb; pdb.set_trace()
        axes[0,0].fill_between(wvs,q_mean_pair1+q_std_pair1,q_mean_pair1-q_std_pair1,color='tab:blue',alpha=0.4,label="Pair 1",zorder=0)
        axes[0,0].fill_between(wvs,q_mean_pair2+q_std_pair2,q_mean_pair2-q_std_pair2,color='tab:orange',alpha=0.4,label="Pair 2",zorder=0)
    if uinds is not None:
        axes[0,1].fill_between(wvs,u_mean_pair1+u_std_pair1,u_mean_pair1-u_std_pair1,color='tab:blue',alpha=0.4,label="Pair 1",zorder=0)
        axes[0,1].fill_between(wvs,u_mean_pair2+u_std_pair2,u_mean_pair2-u_std_pair2,color='tab:orange',alpha=0.4,label="Pair 1",zorder=0)

    #Only plot theta where > 3sigma
    if all_theta:
        where_theta = p_mean > 0
    else:
        where_theta = p_mean > 3*p_std
    axes[1,1].errorbar(wvs[where_theta],theta_mean[where_theta],yerr=theta_mean_err[where_theta],linestyle="None",marker='o',color='k')
    axes[2,1].errorbar(np.radians(theta_mean[where_theta]),wvs[where_theta],xerr=np.radians(theta_mean_err[where_theta]),linestyle="None",marker='o',color='k')
    #Fill in photon/ron error ranges from stds
    axes[0,0].plot(wvs,q_mean+q_std,'k--',alpha=0.5,label="Standard Error on the Mean")
    axes[0,0].plot(wvs,q_mean-q_std,'k--',alpha=0.5)
    axes[0,1].plot(wvs,u_mean+u_std,'k--',alpha=0.5)
    axes[0,1].plot(wvs,u_mean-u_std,'k--',alpha=0.5)
    axes[1,0].plot(wvs,p_mean+p_std,'k--',alpha=0.5)
    axes[1,0].plot(wvs,p_mean-p_std,'k--',alpha=0.5)
    # axes[1,1].plot(wvs,theta_mean+theta_std,'k--',alpha=0.5)
    # axes[1,1].plot(wvs,theta_mean-theta_std,'k--',alpha=0.5)

    #3-sigma for p
    # axes[1,0].plot(wvs,3*p_std,'r')
    # axes[1,0].plot(wvs,3*p_mean_err,'r')

    #Axis plot ranges
    axes[0,0].set_xlim(xlow,xhigh)
    axes[0,1].set_xlim(xlow,xhigh)
    axes[1,0].set_xlim(xlow,xhigh)
    axes[1,1].set_xlim(xlow,xhigh)

    axes[0,0].set_ylim(ylow,yhigh)
    axes[0,1].set_ylim(ylow,yhigh)
    axes[1,0].set_ylim(0,yhigh)
    axes[1,1].set_ylim(theta_wrap-180,theta_wrap)

    axes[0,0].locator_params(nbins=6)
    axes[0,1].locator_params(nbins=6)
    axes[1,0].locator_params(nbins=6)
    axes[1,1].locator_params(nbins=6)
    axes[2,1].locator_params(nbins=3)

    axes[0,0].legend(loc=legend_loc,fontsize=14)
    #Figure Title
    if title is None:
        fig.suptitle("{}, {}, t_exp = {}, Bin size = {}".format(target_name,date,t_ext,binsize),fontsize=24)
    else:
        fig.suptitle(title,fontsize=24)

    #Some annotations: 
    diff=(yhigh-ylow)
    spacing = diff/16
    # #### Add some Text
    axes[0,0].text(1.025*xlow,yhigh-spacing,r"Mean $q$ = {:.2f}%".format(mn_q*100),fontsize=20)
    axes[0,0].text(1.025*xlow,yhigh-2*spacing,r"Mean $q$ Error = {:.2f}%".format(mn_q_err*100),fontsize=20)
    axes[0,0].text(1.025*xlow,yhigh-3*spacing,r"Std $q$ = {:.2f}%".format(std_q*100),fontsize=20)

    axes[0,1].text(1.025*xlow,yhigh-spacing,"Mean $u$ = {:.2f}%".format(mn_u*100),fontsize=20)
    axes[0,1].text(1.025*xlow,yhigh-2*spacing,"Mean $u$ Error = {:.2f}%".format(mn_u_err*100),fontsize=20)
    axes[0,1].text(1.025*xlow,yhigh-3*spacing,"Std $u$ = {:.2f}%".format(std_u*100),fontsize=20)

    axes[1,0].text(1.025*xlow,yhigh-spacing,"Mean $p$ = {:.2f}%".format(mn_p*100),fontsize=20)
    axes[1,0].text(1.025*xlow,yhigh-1.5*spacing,"Mean $p$ Error = {:.2f}%".format(mn_p_err*100),fontsize=20)
    axes[1,0].text(1.025*xlow,yhigh-2*spacing,"Std $p$ = {:.2f}%".format(std_p*100),fontsize=20)

    ## Put in the plot overlaid with the spectrum
    axes[2,0].plot(wvs,p_mean,'k')
    axes[2,0].fill_between(wvs,p_mean+p_mean_err,p_mean-p_mean_err,color='k',alpha=0.1)
    # axes[2,0].plot(wvs,3*p_mean_err,'r',label=r"3$\sigma$ from zero")
    # axes[1,0].plot(wvs,3*p_mean_err,'r',label=r"3$\sigma$ from zero")
    axes[2,0].plot(wvs,3*p_std,'r',label=r"3$\sigma$ from zero")
    axes[1,0].plot(wvs,3*p_std,'r',label=r"3$\sigma$ from zero")
    axes[2,0].legend(fontsize=14)
    if spec is not None:
        #Twin axis to show the mean spectrum
        twin = axes[2,0].twinx()
        p_right, = twin.plot(wvs,spec)
        twin.set_ylim(0,1.3*np.max(spec))

    #### Lots of plot setup ####

    #Labels
    axes[0,0].set_ylabel("q",fontsize=24)
    axes[0,1].set_ylabel("u",fontsize=24)
    axes[1,0].set_ylabel("p",fontsize=24)
    axes[1,1].set_ylabel(r"$\theta$",fontsize=24)
    axes[1,1].set_xlabel(r"$\theta$",fontsize=24)

    axes[1,0].set_xlabel(r"Wavelength [$\mu m$]",fontsize=24)
    axes[1,1].set_xlabel(r"Wavelength [$\mu m$]",fontsize=24)
    axes[2,0].set_xlabel(r"Wavelength [$\mu m$]",fontsize=24)
    axes[2,1].set_xlabel(r"Wavelength [$\mu m$]",fontsize=24)
    if spec is not None:
        twin.set_ylabel("Uncalibrated Spectrum",color=p_right.get_color())

    #Shrink the space for the title
    plt.subplots_adjust(top=0.95)

    ### Axis plot ranges
    axes[0,0].set_xlim(xlow,xhigh)
    axes[0,1].set_xlim(xlow,xhigh)
    axes[1,0].set_xlim(xlow,xhigh)
    axes[1,1].set_xlim(xlow,xhigh)
    axes[2,0].set_xlim(xlow,xhigh)

    axes[0,0].set_ylim(ylow,yhigh)
    axes[0,1].set_ylim(ylow,yhigh)
    axes[1,0].set_ylim(0,yhigh)
    axes[1,1].set_ylim(theta_wrap-180,theta_wrap)
    axes[2,1].set_ylim(xlow,xhigh)
    axes[2,1].set_xlim(np.radians(theta_wrap)-np.pi,np.radians(theta_wrap))
    axes[2,0].set_ylim(0,yhigh)
    axes[2,1].set_rticks([1.15, 1.2, 1.25, 1.30])
    ### Number of ticks
    axes[0,0].locator_params(nbins=6)
    axes[0,1].locator_params(nbins=6)
    axes[1,0].locator_params(nbins=6)
    axes[1,1].locator_params(nbins=6)
    axes[2,0].locator_params(nbins=6)

    if ldwarf:
        ############################
        ######## Absorption Lines, Ranges and Bandheads #########
        ############################

        #Vo Bandhead
        axes[2,0].plot((1.17,1.22),[0.93*yhigh,0.93*yhigh],'k',label="VO")
        axes[2,0].text((1.195),0.95*yhigh,"V0",fontsize=14)#H20 Range

        #Na Line
        axes[2,0].plot((1.138,1.138),(0.85*yhigh,0.95*yhigh),'k',label="Na")
        axes[2,0].plot((1.141,1.141),(0.85*yhigh,0.95*yhigh),'k',label="Na")
        axes[2,0].text(1.141,0.9*yhigh, "Na",fontsize=14)

        #K lines
        axes[2,0].plot((1.169,1.169),(0.75*yhigh,0.85*yhigh),'C1',label="K")
        axes[2,0].plot((1.177,1.177),(0.75*yhigh,0.85*yhigh),'C1',label="K")
        axes[2,0].text(1.171,0.8*yhigh,"K",fontsize=14,color='C1')

        axes[2,0].plot((1.252,1.252),(0.75*yhigh,0.85*yhigh),'C1',label="K")
        axes[2,0].plot((1.243,1.243),(0.75*yhigh,0.85*yhigh),'C1',label="K")
        axes[2,0].text(1.245,0.8*yhigh,"K",fontsize=14,color='C1')

        #FeH doublet
        axes[2,0].plot((1.1939,1.1939),(0.75*yhigh,0.85*yhigh),"C2",label="FeH")
        axes[2,0].plot((1.210,1.210),(0.75*yhigh,0.85*yhigh),"C2",label="FeH")
        axes[2,0].text(1.1955,0.8*yhigh,"FeH",fontsize=14,color='C2')

        axes[2,0].plot((1.2389,1.2389),(0.75*yhigh,0.85*yhigh),"C2",label="FeH")
        axes[2,0].text(1.225,0.8*yhigh,"FeH",fontsize=14,color='C2')

        #H20
        axes[2,0].plot((1.3,1.51),(0.85*yhigh,0.85*yhigh),'c',label=r"H$_2$0") #Changed from Mike's list below to be the range from Cushing
        axes[2,0].text(1.31,0.86*yhigh,r"H$_2$0",fontsize=14,color='c')

        axes[2,0].plot((1.1,1.2),(0.86*yhigh,0.86*yhigh),'c',label=r"H$_2$0")
        axes[2,0].text(1.170,0.88*yhigh,r"H$_2$0",fontsize=14,color='c')


    if tdwarf:

        #CH4
        axes[2,0].plot([1.1,1.24],[0.80*yhigh,0.80*yhigh],color='k')
        axes[2,0].text(1.20,0.825*yhigh, r'CH$_4$',fontsize=14,color='k')

        axes[2,0].plot([1.28,1.44],[0.8*yhigh,0.8*yhigh],color='k')
        axes[2,0].text(1.30,0.825*yhigh, r'CH$_4$',fontsize=14,color='k')

        #H20
        axes[2,0].plot((1.3,1.51),(0.85*yhigh,0.85*yhigh),'k',label=r"H$_2$0") #Changed from Mike's list below to be the range from Cushing
        axes[2,0].text(1.31,0.86*yhigh,r"H$_2$0",fontsize=14)

        #H20
        axes[2,0].plot((1.1,1.2),(0.86*yhigh,0.86*yhigh),'k',label=r"H$_2$0")
        axes[2,0].text(1.170,0.88*yhigh,r"H$_2$0",fontsize=14)


    # "\n",
    # "plt.plot([1.28,1.44],[0.8,0.80],linestyle='-',lw=1.5,color='grey')\n",
    # "plt.text(1.33,0.825, r'CH$_4$',fontsize=20,color='grey')\n",
    if save_path is not None:
        if filename is None:
            if binsize > 1:
                filename = "{}_{}_Binned.png".format(target_name,date,binsize)
            else: 
                filename = "{}_{}_Binned.png".format(target_name,date)
        plt.savefig(save_path+filename)
    if show:
        plt.draw()

def plot_pol_summary_time_bins(master_wvs,master_spec,spec_cube,hwp_ang,n_time_bins=1,mode='mean',xlow=1.15,xhigh=1.325,ylow=-0.02,yhigh=0.02,
    target_name="",date="19850625",t_ext = 0,binsize=1,theta_wrap=180,ldwarf=False,show=True,
    save_path=None,legend_loc ="bottom left",all_theta=False,cmap=None,dt=None,period=None):
    '''
    Make a summary plot of polarization. The formatting assumes that the inputs (q,u,qerr,uerr)
    are the output of compute_qu_for_obs_sequence. 

    Inputs:
    wvs     - The wavelengths for the x-axis. Should correspond to the rest of the inputs
    spec    - The total intensity spectrum
    spec_cube - A cube that holds All of the spectra (aligned) from your data. Should have shape (nfiles, 4, 3, spec_length)
    hwp_ang - The HWP angles. Should have length nfiles
    mode    - Either "mean" or "median"
    dt      - the change in time from the first frame. 
    '''

    ### Make the plot!!! ###
    fig,axes = plt.subplots(3,2,figsize=(16,20))

    #Cycle through the time bins

    time_snip = spec_cube.shape[0] % (n_time_bins*4)
    time_bin_size = (spec_cube.shape[0]-time_snip)//n_time_bins

    good_wvs = (master_wvs > 1.175) & (master_wvs < 1.325)
    spec_cube = copy.deepcopy(spec_cube[:,:,:,good_wvs])
    master_wvs = copy.deepcopy(master_wvs[good_wvs])
    master_spec = copy.deepcopy(master_spec[good_wvs])

    if dt is not None:
        if time_snip != 0:
            dt_bins = np.mean(np.reshape(dt[:-time_snip],(-1,time_bin_size)),axis=1)
        else:
            dt_bins = np.mean(np.reshape(dt,(-1,time_bin_size)),axis=1)

        if period is None:
            period = np.max(dt)

        phase = (dt_bins % period)/period
    else:
        phase = np.linspace(0,1.0,n_time_bins,endpoint=False)

    time_inds = np.arange(spec_cube.shape[0])
    if cmap is None:
        colormapp = mpl.colors.LinearSegmentedColormap.from_list("", ["k","firebrick","crimson","darkorchid","blueviolet","mediumblue","navy","k"])
    else:
        colormapp = plt.get_cmap(cmap) 
    # print(phase)
    colors = colormapp(phase)
    
    pmeans = []
    qmeans = []
    umeans = []
    theta_means = []
    for k in range(n_time_bins):
        # print("time_bin_size = {}".format(time_bin_size)) 
        good_inds = time_inds[np.where((time_inds >= k*time_bin_size) & (time_inds < (k+1)*time_bin_size))]

        # import pdb; pdb.set_trace()
        # print("Using inds {}".format(good_inds))
        q,u,qerr,uerr,qind,uind = compute_qu_for_obs_sequence(spec_cube[good_inds],hwp_ang[good_inds],run_alignment=False)

        #First calculate the double_difference values
        q_dd = np.nanmean(q,axis=1)
        u_dd = np.nanmean(u,axis=1)
        p_dd = np.sqrt(q_dd**2+u_dd**2)
        theta_dd = 0.5*np.degrees(np.arctan2(u_dd,q_dd))
        theta_dd[theta_dd < 0] +=180

        # q_dd_err = np.sqrt(np.sum((qerr**2),axis=1))/qerr.shape[1]
        q_dd_err = qerr
        # u_dd_err = np.sqrt(np.sum((uerr**2),axis=1))/uerr.shape[1]
        u_dd_err = uerr

        #Now calculate the mean or median
        from astropy import stats
        q_mean = np.zeros([q.shape[1]]) #We name this mean, though it could either be Mean or Median
        q_std = np.zeros([q.shape[1]])
        u_mean = np.zeros([u.shape[1]])
        u_std = np.zeros([u.shape[1]])

        for i in range(q.shape[1]):
            # import pdb; pdb.set_trace()
            mn,md,std = stats.sigma_clipped_stats(q[:,i], sigma=3, maxiters=5)
            if mode == 'median':
                q_mean[i] = md
            else:
                q_mean[i] = mn
            q_std[i] = std/np.sqrt(q_dd.shape[0])
            mn,md,std = stats.sigma_clipped_stats(u[:,i], sigma=3, maxiters=5)
            u_std[i] = std/np.sqrt(q_dd.shape[0])
            if mode == 'median':
                u_mean[i] = md
            else:
                u_mean[i] = mn

        p_mean = np.sqrt(q_mean**2+u_mean**2)
        theta_mean = 0.5*np.degrees(np.arctan2(u_mean,q_mean))
        theta_mean[theta_mean < theta_wrap] +=180

        q_mean_err = np.sqrt(np.sum(q_dd_err**2,axis=0))/q_dd_err.shape[0]
        u_mean_err = np.sqrt(np.sum(u_dd_err**2,axis=0))/q_dd_err.shape[0]
        p_mean_err = np.sqrt(q_mean**2*q_mean_err**2+u_mean**2*u_mean_err**2)/p_mean
        p_std = np.sqrt(q_mean**2*q_std**2+u_mean**2*u_std**2)/p_mean
        theta_mean_err = 0.5*np.degrees( np.sqrt( (u_mean**2*q_mean_err**2+q_mean**2*u_mean_err**2)/(q_mean**2+u_mean**2)**2))
        theta_std = 0.5*np.degrees( np.sqrt( (u_mean**2*q_std**2+q_mean**2*u_std**2)/(q_mean**2+u_mean**2)**2))


        wvs = copy.deepcopy(master_wvs)
        spec = copy.deepcopy(master_spec)
        ### Implement Binning
        if binsize != 1:
            snip = q_mean.shape[0] % binsize
            if snip != 0:
                q_mean = np.mean(q_mean[:-snip].reshape(-1,binsize),axis=1)
                u_mean = np.mean(u_mean[:-snip].reshape(-1,binsize),axis=1)
                wvs = np.mean(wvs[:-snip].reshape(-1,binsize),axis=1)
                p_mean = np.sqrt(q_mean**2+u_mean**2)
                theta_mean = 0.5*np.degrees(np.arctan2(u_mean,q_mean))

                spec = np.mean(spec[:-snip].reshape(-1,binsize),axis=1)

                q_mean_err = np.sqrt(np.sum(q_mean_err[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
                q_std = np.sqrt(np.sum(q_std[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
                u_mean_err = np.sqrt(np.sum(u_mean_err[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
                u_std = np.sqrt(np.sum(u_std[:-snip].reshape(-1,binsize)**2,axis=1))/binsize
                p_mean_err = np.sqrt(q_mean**2*q_mean_err**2+u_mean**2*u_mean_err**2)/p_mean
                theta_mean_err = 0.5*np.degrees( np.sqrt( (u_mean**2*q_mean_err**2+q_mean**2*u_mean_err**2)/(q_mean**2+u_mean**2)**2))
                
            
            else: 
                q_mean = np.mean(q_mean.reshape(-1,binsize),axis=1)
                u_mean = np.mean(u_mean.reshape(-1,binsize),axis=1)
                wvs = np.mean(wvs.reshape(-1,binsize),axis=1)
                p_mean = np.sqrt(q_mean**2+u_mean**2)
                theta_mean = 0.5*np.degrees(np.arctan2(u_mean,q_mean))
                # theta_bin[theta_bin < 0] +=180

                spec = np.mean(spec.reshape(-1,binsize),axis=1)

                q_mean_err = np.sqrt(np.sum(q_mean_err.reshape(-1,binsize)**2,axis=1))/binsize
                q_std = np.sqrt(np.sum(q_std.reshape(-1,binsize)**2,axis=1))/binsize
                u_mean_err = np.sqrt(np.sum(u_mean_err.reshape(-1,binsize)**2,axis=1))/binsize
                u_std = np.sqrt(np.sum(u_std.reshape(-1,binsize)**2,axis=1))/binsize
                p_mean_err = np.sqrt(q_mean**2*q_mean_err**2+u_mean**2*u_mean_err**2)/p_mean
                theta_mean_err = 0.5*np.degrees( np.sqrt( (u_mean**2*q_mean_err**2+q_mean**2*u_mean_err**2)/(q_mean**2+u_mean**2)**2))
        p_std = np.sqrt(q_mean**2*q_std**2+u_mean**2*u_std**2)/p_mean
        theta_std = 0.5*np.degrees( np.sqrt( (u_mean**2*q_std**2+q_mean**2*u_std**2)/(q_mean**2+u_mean**2)**2))
        
        pmeans.append(p_mean)
        qmeans.append(q_mean)
        umeans.append(u_mean) 
        #Wrap theta about 180
        theta_mean[theta_mean>theta_wrap] -= 180
        theta_means.append(theta_mean)        

        ##Calculate the mean values
        low = 65
        high = 135

        inds = (wvs > 1.165) & (wvs<1.31)
        mn_q = np.mean(q_mean[inds])
        mn_u = np.mean(u_mean[inds])
        mn_p = np.mean(p_mean[inds])

        mn_q_err = np.mean(q_mean_err[inds])
        mn_u_err = np.mean(u_mean_err[inds])
        mn_p_err = np.mean(p_mean_err[inds])

        std_q = np.std(q_mean[inds])
        std_u = np.std(u_mean[inds])
        std_p = np.std(p_mean[inds])

        ### Make the plot!!! ###

        axes[2,1] = plt.subplot(3,2,6, projection='polar')
        ##### Plot Q, U, P and theta ######
        #The mean values
        if mode == "median":
            axes[0,0].plot(wvs,q_mean,color=colors[k],label="Median")
        else:
            axes[0,0].plot(wvs,q_mean,color=colors[k],label="Mean")    
        axes[0,1].plot(wvs,u_mean,color=colors[k])   
        axes[1,0].plot(wvs,p_mean,color=colors[k])
        

        #Make a line at zero
        axes[0,0].axhline(0.,color='k',linestyle='--')
        axes[0,1].axhline(0.,color='k',linestyle='--')

        #Fill in photon/ron error ranges
        axes[0,0].fill_between(wvs,q_mean+q_mean_err,q_mean-q_mean_err,color=colors[k],alpha=0.1,label="Propagated Noise")
        axes[0,1].fill_between(wvs,u_mean+u_mean_err,u_mean-u_mean_err,color=colors[k],alpha=0.1)
        axes[1,0].fill_between(wvs,p_mean+p_mean_err,p_mean-p_mean_err,color=colors[k],alpha=0.1)
        # axes[1,1].fill_between(wvs,theta_mean+theta_mean_err,theta_mean-theta_mean_err,color='k',alpha=0.1)

        #Only plot theta where > 3sigma
        if all_theta:
            where_theta = p_mean > 0
        else:
            where_theta = p_mean > 3*p_std
        axes[1,1].errorbar(wvs[where_theta],theta_mean[where_theta],yerr=theta_mean_err[where_theta],
            linestyle="None",marker='o',color=colors[k])
        axes[2,1].errorbar(np.radians(theta_mean[where_theta]),wvs[where_theta],xerr=np.radians(theta_mean_err[where_theta]),
            linestyle="None",marker='o',color=colors[k])
        #Fill in photon/ron error ranges from stds
        axes[0,0].plot(wvs,q_mean+q_std,'--',color=colors[k],alpha=0.5,label="Standard Error on the Mean")
        axes[0,0].plot(wvs,q_mean-q_std,'--',color=colors[k],alpha=0.5)
        axes[0,1].plot(wvs,u_mean+u_std,'--',color=colors[k],alpha=0.5)
        axes[0,1].plot(wvs,u_mean-u_std,'--',color=colors[k],alpha=0.5)
        axes[1,0].plot(wvs,p_mean+p_std,'--',color=colors[k],alpha=0.5)
        axes[1,0].plot(wvs,p_mean-p_std,'--',color=colors[k],alpha=0.5)
        # axes[1,1].plot(wvs,theta_mean+theta_std,'k--',alpha=0.5)
        # axes[1,1].plot(wvs,theta_mean-theta_std,'k--',alpha=0.5)

        #3-sigma for p
        # axes[1,0].plot(wvs,3*p_std,color='cyan')
        # axes[1,0].plot(wvs,3*p_mean_err,'k--')
        
        ####Add the colormaps to show the phase colors
        #Top left
        cbaxes = inset_axes(axes[0,0], width="60%", height="7%", loc=9) 
        cbar = mpl.colorbar.ColorbarBase(cbaxes, cmap = colormapp,norm = mpl.colors.Normalize(vmin=0,vmax=1.),ticks=[0.,0.5,1], orientation='horizontal')
        if period == np.max(dt):
            cbar.set_label("Phase (Period = ??)")
        else:
            cbar.set_label("Phase (Period = {}h)".format(period))
        cbaxes.scatter(phase,phase*0.,marker="^",color='white',s=900)

        for i,p in enumerate(phase):
            cbaxes.text(p-0.015,0.075,"{:d}".format(i),weight="bold")

        #Top Right
        cbaxes = inset_axes(axes[0,1], width="60%", height="7%", loc=9) 
        cbar = mpl.colorbar.ColorbarBase(cbaxes, cmap = colormapp,norm = mpl.colors.Normalize(vmin=0,vmax=1.),ticks=[0.,0.5,1], orientation='horizontal')
        if period == np.max(dt):
            cbar.set_label("Phase (Period = ??)")
        else:
            cbar.set_label("Phase (Period = {}h)".format(period))
        cbaxes.scatter(phase,phase*0.,marker="^",color='white',s=900)
        for i,p in enumerate(phase):
            cbaxes.text(p-0.015,0.075,"{:d}".format(i),weight="bold")

        #Center Left
        cbaxes = inset_axes(axes[1,0], width="60%", height="7%", loc=9) 
        cbar = mpl.colorbar.ColorbarBase(cbaxes, cmap = colormapp,norm = mpl.colors.Normalize(vmin=0,vmax=1.),ticks=[0.,0.5,1], orientation='horizontal')
        if period == np.max(dt):
            cbar.set_label("Phase (Period = ??)")
        else:
            cbar.set_label("Phase (Period = {}h)".format(period))
        cbaxes.scatter(phase,phase*0.,marker="^",color='white',s=700)
        for i,p in enumerate(phase):
            cbaxes.text(p-0.015,0.075,"{:d}".format(i),weight="bold")

        #Axis plot ranges
        axes[0,0].set_xlim(xlow,xhigh)
        axes[0,1].set_xlim(xlow,xhigh)
        axes[1,0].set_xlim(xlow,xhigh)
        axes[1,1].set_xlim(xlow,xhigh)

        axes[0,0].set_ylim(ylow,yhigh)
        axes[0,1].set_ylim(ylow,yhigh)
        axes[1,0].set_ylim(0,yhigh)
        axes[1,1].set_ylim(theta_wrap-180,theta_wrap)

        axes[0,0].locator_params(nbins=6)
        axes[0,1].locator_params(nbins=6)
        axes[1,0].locator_params(nbins=6)
        axes[1,1].locator_params(nbins=6)
        axes[2,1].locator_params(nbins=3)

        if k == 0:
            axes[0,0].legend(loc=legend_loc,fontsize=14)
        #Figure Title
        fig.suptitle("{}, {}, t_exp = {}, Bin size = {}".format(target_name,date,t_ext,binsize),fontsize=24)

        #Some annotations: 
        diff=(yhigh-ylow)
        spacing = diff/16
        # #### Add some Text
        # axes[0,0].text(1.025*xlow,yhigh-spacing,r"Mean $q$ = {:.2f}%".format(mn_q*100),fontsize=20)
        # axes[0,0].text(1.025*xlow,yhigh-2*spacing,r"Mean $q$ Error = {:.2f}%".format(mn_q_err*100),fontsize=20)
        # axes[0,0].text(1.025*xlow,yhigh-3*spacing,r"Std $q$ = {:.2f}%".format(std_q*100),fontsize=20)

        # axes[0,1].text(1.025*xlow,yhigh-spacing,"Mean $u$ = {:.2f}%".format(mn_u*100),fontsize=20)
        # axes[0,1].text(1.025*xlow,yhigh-2*spacing,"Mean $u$ Error = {:.2f}%".format(mn_u_err*100),fontsize=20)
        # axes[0,1].text(1.025*xlow,yhigh-3*spacing,"Std $u$ = {:.2f}%".format(std_u*100),fontsize=20)

        # axes[1,0].text(1.025*xlow,yhigh-spacing,"Mean $p$ = {:.2f}%".format(mn_p*100),fontsize=20)
        # axes[1,0].text(1.025*xlow,yhigh-1.5*spacing,"Mean $p$ Error = {:.2f}%".format(mn_p_err*100),fontsize=20)
        # axes[1,0].text(1.025*xlow,yhigh-2*spacing,"Std $p$ = {:.2f}%".format(std_p*100),fontsize=20)

        ## Put in the plot overlaid with the spectrum
        axes[2,0].plot(wvs,p_mean,color=colors[k])
        axes[2,0].fill_between(wvs,p_mean+p_mean_err,p_mean-p_mean_err,color=colors[k],alpha=0.1)
        axes[2,0].plot(wvs,3*p_std,'r',label=r"3$\sigma$ from zero")
        axes[1,0].plot(wvs,3*p_std,'r',label=r"3$\sigma$ from zero")
        if k == 0:
            axes[2,0].legend(fontsize=14)
        #Twin axis to show the mean spectrum
        twin = axes[2,0].twinx()
        p_right, = twin.plot(wvs,spec)
        twin.set_ylim(0,1.3*np.max(spec))

        #### Lots of plot setup ####

        #Labels
        axes[0,0].set_ylabel("q",fontsize=24)
        axes[0,1].set_ylabel("u",fontsize=24)
        axes[1,0].set_ylabel("p",fontsize=24)
        axes[1,1].set_ylabel(r"$\theta$",fontsize=24)
        axes[1,1].set_xlabel(r"$\theta$",fontsize=24)

        axes[0,0].set_xlabel(r"Wavelength [$\mu m$]",fontsize=24)
        axes[0,1].set_xlabel(r"Wavelength [$\mu m$]",fontsize=24)
        axes[1,0].set_xlabel(r"Wavelength [$\mu m$]",fontsize=24)
        axes[1,1].set_xlabel(r"Wavelength [$\mu m$]",fontsize=24)
        axes[2,0].set_xlabel(r"Wavelength [$\mu m$]",fontsize=24)
        twin.set_ylabel("Uncalibrated Spectrum",color=p_right.get_color())

        #Shrink the space for the title
        plt.subplots_adjust(top=0.95)

        ### Axis plot ranges
        axes[0,0].set_xlim(xlow,xhigh)
        axes[0,1].set_xlim(xlow,xhigh)
        axes[1,0].set_xlim(xlow,xhigh)
        axes[1,1].set_xlim(xlow,xhigh)
        axes[2,0].set_xlim(xlow,xhigh)

        axes[0,0].set_ylim(ylow,yhigh)
        axes[0,1].set_ylim(ylow,yhigh)
        axes[1,0].set_ylim(0,yhigh)
        axes[1,1].set_ylim(theta_wrap-180,theta_wrap)
        axes[2,1].set_ylim(xlow,xhigh)
        axes[2,1].set_xlim(np.radians(theta_wrap)-np.pi,np.radians(theta_wrap))
        axes[2,0].set_ylim(0,yhigh)
        axes[2,1].set_rticks([1.15, 1.2, 1.25, 1.30])
        ### Number of ticks
        axes[0,0].locator_params(nbins=6)
        axes[0,1].locator_params(nbins=6)
        axes[1,0].locator_params(nbins=6)
        axes[1,1].locator_params(nbins=6)
        axes[2,0].locator_params(nbins=6)

        if ldwarf and k==0:
            ############################
            ######## Absorption Lines, Ranges and Bandheads #########
            ############################

            #Vo Bandhead
            axes[2,0].plot((1.17,1.22),[0.93*yhigh,0.93*yhigh],'k',label="VO")
            axes[2,0].text((1.195),0.95*yhigh,"V0",fontsize=14)#H20 Range

            axes[2,0].plot((1.1,1.2),(0.86*yhigh,0.86*yhigh),'k',label=r"H$_2$0")
            axes[2,0].text(1.170,0.88*yhigh,r"H$_2$0",fontsize=14)

            #Na Line
            axes[2,0].plot((1.14,1.14),(0.85*yhigh,0.95*yhigh),'k',label="Na")
            axes[2,0].text(1.141,0.9*yhigh, "Na",fontsize=14)

            #K lines
            axes[2,0].plot((1.177,1.177),(0.75*yhigh,0.85*yhigh),'k',label="K")
            axes[2,0].text(1.179,0.8*yhigh,"K",fontsize=14)


            axes[2,0].plot((1.2485,1.2485),(0.75*yhigh,0.85*yhigh),'k',label="K")
            axes[2,0].text(1.2505,0.8*yhigh,"K",fontsize=14)

            #FeH doublet
            axes[2,0].plot((1.1939,1.1939),(0.75*yhigh,0.85*yhigh),"k",label="FeH")
            axes[2,0].text(1.1955,0.8*yhigh,"FeH",fontsize=14)

            axes[2,0].plot((1.2389,1.2389),(0.75*yhigh,0.85*yhigh),"k",label="FeH")
            axes[2,0].text(1.2282,0.8*yhigh,"FeH",fontsize=14)

            #H20
            axes[2,0].plot((1.3,1.51),(0.85*yhigh,0.85*yhigh),'k',label=r"H$_2$0") #Changed from Mike's list below to be the range from Cushing
            axes[2,0].text(1.31,0.86*yhigh,r"H$_2$0",fontsize=14)

    if save_path is not None:
        if binsize > 1:
            fn = "{}_{}_Binned.png".format(target_name,date,binsize)
        else: 
            fn = "{}_{}_Binned.png".format(target_name,date)
        plt.savefig(save_path+fn)

    if show:
        plt.show()
    if dt is not None:
        return dt_bins,phase,qmeans,umeans,pmeans,theta_means, phase, wvs
        return qmeans,umeans,pmeans,theta_means, wvs


def thresholding_algo(y, lag, threshold, influence):
    """
    Helper function for quantize_peaks
    """
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def quantize_peaks(y, lag=30, threshold=5, influence=0, show_plot=True):
    """
    Finds peaks in a data vector and quantizes them
    """
    
    result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)

    if show_plot:
        # Plot result
        plt.subplot(211)
        plt.plot(np.arange(1, len(y)+1), y)

        plt.plot(np.arange(1, len(y)+1),
                result["avgFilter"], color="cyan", lw=2)

        plt.plot(np.arange(1, len(y)+1),
                result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

        plt.plot(np.arange(1, len(y)+1),
                result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

        plt.subplot(212)
        plt.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
        plt.ylim(-1.5, 1.5)
        plt.show()
        
    return result

def missing_elements(L):
    start, end = L[0], L[-1]
    return sorted(set(range(start, end + 1)).difference(L))

def perp_fit(fit, ctr):
    """
    calculates fit to a line perpendicular to trace 
    """
    x1=0
    x2=len(fit)
    
    y1 = fit[0]
    y2 = fit[-1]
    
    m = (y2-y1)/(x2-x1)
    
    #y_fit = (-1/m)*x_fit + b ==>
    #ctr[1] = (-1/m)*ctr[0] + b ==>
    b = ctr[1] + (1/m)*ctr[0]
    
    x_fit = np.arange(len(fit))
    
    pfit = (-1/m)*x_fit + b
    
    pfit[np.where(pfit<0)] = 0
    pfit[np.where(pfit>len(fit))] = len(fit)-1
    
    return pfit


def truncate(value, limits=[0, 160]):
    if value <= limits[0]:
        value = limits[0]
    elif value >= limits[1]:
        value = limits[1]
    return value


def fit_aperture(source, exp_time, stretch=1, plot=True, savefig=None, verbose=True,
                trace_len=90, trace_wid=8, fit_trace=False, interp_factor=10):

    gain = 1.2 #[e-/ADU]. 
    read_noise = 12 #[e-]
    dark_current = 1 #[e-/s]
    dark_current_noise = dark_current*exp_time
    
    apertures = []
    total_flux = []
    SNRs = []
    trace_labels = ['UL', 'LR', 'UR', 'LL']
    
    for k in range(4):

        im = source.trace_images[k]
        bkg = source.trace_bkg[k]
        orig_im = source.trace_images[k]

        im = im - bkg
        
        yy, xx = np.indices(im.shape)

        peak, fit, width, angle = findTrace(im, weighted=True, plot=False)

        angle=angle*np.pi/180

        #if true, we automatically fit the trace dimensions using length and width of FWHM 
        if fit_trace:
            trace = np.array([im[truncate(fit.astype(int)[i])][i] for i in range(len(fit))])

            yy, xx = np.indices(im.shape)

            results = quantize_peaks(trace, lag=10, threshold=4, influence=0, show_plot=plot)

            mean_peak = np.mean(trace[np.where(results["signals"]>0)][10:-10])

            index = np.where(results["signals"]>0)[0]

            if verbose:
                print("Mean peak flux = {}".format(mean_peak))

            peak_indices = missing_elements(np.where(trace<.5*mean_peak)[0])

            FWHM = peak_indices[-1]-peak_indices[0]

            if verbose:
                print("FWHM = {} px".format(FWHM))

            if plot:
                plt.figure()
                plt.plot(trace[np.where(results["signals"]>0)])
                plt.axvline(peak_indices[0] - index[0], ymin=0, ymax=1, color='r', ls='--', label='FWHM = {}'.format(FWHM))
                plt.axvline(peak_indices[-1] - index[0], ymin=0, ymax=1, color='r', ls='--')
                plt.axhline(mean_peak, 10/len(trace[np.where(results["signals"]>0)]),
                            (len(trace[np.where(results["signals"]>0)])-11)/len(trace[np.where(results["signals"]>0)]),
                            color='k', ls='--', label='Mean peak value = {}'.format(np.round(mean_peak, 2)))
                plt.legend()
                plt.show()
                plt.close()

            peak_index = (peak_indices[0]+peak_indices[-1])//2

            p_fit = perp_fit(fit, (peak_index, fit[peak_index]))

            p_trace = np.array([im[p_fit.astype(int)[i]][i] for i in range(len(fit))])

            #interpolate perp trace to better calculate FWHM
            p_trace_interp = interp1d(np.arange(len(p_trace)), p_trace)(np.linspace(0, len(fit)-1, len(fit)*interp_factor))

            if plot:
                plt.figure()
                plt.title('Perpendicular trace')
                plt.plot(p_trace)
                plt.show()
                plt.close()

            p_results = quantize_peaks(p_trace_interp, lag=20*interp_factor, threshold=5, influence=0, show_plot=plot)

            p_peak = np.max(p_trace_interp)

            p_index = np.where(p_results["signals"]>0)[0]

            if verbose:
                print("Peak flux = {}".format(p_peak))

            p_peak_indices = missing_elements(np.where(p_trace_interp<.5*p_peak)[0])

            p_peak_index = int(np.where(p_trace_interp==np.max(p_trace_interp))[0][0]/interp_factor)

            p_FWHM = (p_peak_indices[-1]-p_peak_indices[0])/interp_factor

            if verbose:
                print("FWHM = {} px".format(p_FWHM))

            if plot:
                plt.figure()
                plt.title('Perpendicular trace peak')
                plt.plot(p_trace_interp[np.where(p_results["signals"]>0)])
                plt.axvline(p_peak_indices[0] - p_index[0], ymin=0, ymax=1, color='r', ls='--', label='FWHM = {}'.format(p_FWHM))
                plt.axvline(p_peak_indices[-1] - p_index[0], ymin=0, ymax=1, color='r', ls='--')
                plt.axhline(p_peak, (p_peak_indices[0] - p_index[0])/len(p_trace_interp[np.where(p_results["signals"]>0)]),
                            (p_peak_indices[-1] - p_index[0])/len(p_trace_interp[np.where(p_results["signals"]>0)]),
                            color='k', ls='--', label='Peak value = {}'.format(np.round(p_peak, 2)))
                plt.legend()
                plt.show()
                plt.close()

            if verbose:
                print("trace ctr = {}".format((peak_index, int(fit[peak_index]))))
            x_ctr, y_ctr = peak_index, int(fit[peak_index])

            circ1 = (xx-x_ctr-x_stretch*FWHM/2)**2+(yy-y_ctr)**2
            circ2 = (xx-x_ctr+x_stretch*FWHM/2)**2+(yy-y_ctr)**2
            ends = np.logical_or((circ1<p_FWHM*y_stretch**2), (circ2<p_FWHM*y_stretch**2)).astype(float)
            
            box_x = np.logical_and((xx>x_ctr-x_stretch*FWHM/2), (xx<x_ctr+x_stretch*FWHM/2))
            box_y = np.logical_and((yy>y_ctr-y_stretch*p_FWHM/2),(yy<y_ctr+y_stretch*p_FWHM/2))
            box = np.logical_and(box_x, box_y).astype(float)
            
            racetrack = np.logical_or(box, ends).astype(float)

            xp = (xx-x_ctr)*np.cos(angle) + (yy-y_ctr)*np.sin(angle) + x_ctr
            yp = -(xx-x_ctr)*np.sin(angle) + (yy-y_ctr)*np.cos(angle) + y_ctr

            racetrack = np.nan_to_num(np.round(map_coordinates(racetrack, (yp, xp), cval=np.nan))).astype(bool)     

            
        #otherwise we use the default values given in the kwarg
        else:
            diff = peak-fit.astype(int)

            s = np.where(abs(diff)<5)[0]
            maxrun = -1
            rl = {}
            for x in s:
                run = rl[x] = rl.get(x-1, 0) + 1
                if run > maxrun:
                    maxend, maxrun = x, run

            trace_x = np.arange(maxend-maxrun+1, maxend+1)

            x_ctr, y_ctr = trace_x[len(trace_x)//2], fit[trace_x].astype(int)[len(trace_x)//2]

            circ1 = (xx-x_ctr-stretch*trace_len/2)**2+(yy-y_ctr)**2
            circ2 = (xx-x_ctr+stretch*trace_len/2)**2+(yy-y_ctr)**2
            ends = np.logical_or((circ1<(trace_wid*stretch)**2), (circ2<(trace_wid*stretch)**2)).astype(float)

            box_x = np.logical_and((xx>x_ctr-stretch*trace_len/2), (xx<x_ctr+stretch*trace_len/2))
            box_y = np.logical_and((yy>y_ctr-stretch*trace_wid/2),(yy<y_ctr+stretch*trace_wid/2))
            box = np.logical_and(box_x, box_y).astype(float)

            racetrack = np.logical_or(box, ends).astype(float)

            xp = (xx-x_ctr)*np.cos(angle) + (yy-y_ctr)*np.sin(angle) + x_ctr
            yp = -(xx-x_ctr)*np.sin(angle) + (yy-y_ctr)*np.cos(angle) + y_ctr

            racetrack = np.nan_to_num(np.round(map_coordinates(racetrack, (yp, xp), cval=np.nan))).astype(bool)

        aperture = np.copy(im.astype(float))
        aperture[~racetrack] = np.nan

        bkg_mask = np.copy(bkg.astype(float))
        bkg_mask[~racetrack] = np.nan

        f, ax = plt.subplots(1, 5, figsize=(19, 3), sharey=True)
        f.suptitle(trace_labels[k], fontsize=25)
        im0 = ax[0].imshow(im, origin='lower', vmin=np.nanmin(aperture), vmax=np.percentile(aperture, 98))
        ax[0].plot(fit)
        ax[0].plot(peak)
        ax[0].set_xlim(0, im.shape[1])
        ax[0].set_ylim(0, im.shape[0])
        im1 = ax[1].imshow(aperture, origin='lower', vmin=np.nanmin(aperture), vmax=np.percentile(aperture, 98))
        ax[1].set_xlabel('Total flux = {}'.format(np.round(np.nansum(aperture),2)))
        im2 = ax[2].imshow(orig_im, origin='lower', vmin=np.nanmin(orig_im), vmax=np.percentile(orig_im, 98))
        ax[2].set_xlabel('Pre-Subt')
        im3 = ax[3].imshow(bkg, origin='lower', vmin=np.nanmin(orig_im), vmax=np.percentile(orig_im, 98))
        ax[3].set_xlabel('Bkg')
        im4 = ax[4].imshow(im, origin='lower', vmin=np.nanmin(im), vmax=np.percentile(im, 98))
        ax[4].set_xlabel('Post-Subt')
        for i in range(5):
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            f.colorbar([im0, im1, im2, im3, im4][i], cax=cax, orientation='vertical')
        if savefig is not None:
            plt.savefig(savefig+'_'+trace_labels[k]+'.png', dpi=300, bbox_to_inches='tight', overwrite=True)
        if plot:
            plt.show()
        plt.close()

        N_flux = np.round(np.nansum(aperture),2)
        N_noise = np.round(np.nansum(bkg_mask),2)
        n_pix = np.count_nonzero(~np.isnan(aperture))
        
        SNR = N_flux/np.sqrt(N_flux+N_noise+n_pix*read_noise**2+n_pix*dark_current_noise)
        
        apertures.append(aperture)
        total_flux.append(N_flux)
        SNRs.append(np.round(SNR, 2))

    return apertures, total_flux, SNRs