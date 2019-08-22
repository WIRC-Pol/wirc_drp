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
def compute_qu(spec1, spec2, HWP1, HWP2, run_alignment = True, method = 'flux_ratio'):
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

        if method == 'double_difference':
            #polarization vector and uncertainty. This is (spec1-spec2)/(spec1+spec2)
            pol_vec = (scaled_cube[0,:,1,:] - scaled_cube[1,:,1,:])/(scaled_cube[0,:,1,:] + scaled_cube[1,:,1,:])
            pol_err = (2/(scaled_cube[0,:,1,:] + scaled_cube[1,:,1,:])**2) * np.sqrt(  (scaled_cube[0,:,1,:]*scaled_cube[1,:,2,:])**2 + (scaled_cube[0,:,2,:]* scaled_cube[1,:,1,:])**2)

            #now determine which is which
            sampling_angles_0 = np.array([135, 45, 90, 0]) #THIS IS FROM UL, LR, UR, LL = U-, U+, Q-, Q+ as determined from twilight. 
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

        elif method == 'flux_ratio':
            #First, figure out the sampling angles
            sampling_angles_0 = np.array([135, 45, 90, 0]) #THIS IS FROM UL, LR, UR, LL = U-, U+, Q-, Q+ as determined from twilight. 
            sampling_angles_1 = (sampling_angles_0 + 2*(HWP1))%180 #angles are mod 180 deg.  
            sampling_angles_2 = (sampling_angles_0 + 2*(HWP2))%180 #angles are mod 180 deg. 

            #indices (non elegant solution...)
            ind0_0 =   np.where(sampling_angles_1 == 0)[0]
            ind0_90 =  np.where(sampling_angles_1 == 90)[0]
            ind0_45 =  np.where(sampling_angles_1 == 45)[0]
            ind0_135 = np.where(sampling_angles_1 == 135)[0]
            ind1_0 =   np.where(sampling_angles_2 == 0)[0]
            ind1_90 =  np.where(sampling_angles_2 == 90)[0]
            ind1_45 =  np.where(sampling_angles_2 == 45)[0]
            ind1_135 = np.where(sampling_angles_2 == 135)[0]

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

def plot_pol_summary(wvs,spec,q,u,qerr,uerr,mode='mean',xlow=1.15,xhigh=1.325,ylow=-0.02,yhigh=0.02,
    target_name="",date="19850625",t_ext = 0,binsize=1,theta_wrap=180,ldwarf=False,show=True,
    save_path=None,legend_loc ="bottom left",all_theta=False,
    fig = None, axes = None):
    '''
    Make a summary plot of polarization. The formatting assumes that the inputs (q,u,qerr,uerr)
    are the output of compute_qu_for_obs_sequence. 

    Inputs:
    mode      - Either "mean" or "median"
    fig, axes - to plot on existing figure/axes. None if not. 
    '''

    #First calculate the double_difference values
    q_dd = np.nanmean(q,axis=1)
    u_dd = np.nanmean(u,axis=1)
    p_dd = np.sqrt(q_dd**2+u_dd**2)
    theta_dd = 0.5*np.degrees(np.arctan2(u_dd,q_dd))
    theta_dd[theta_dd < 0] +=180

    q_dd_err = np.sqrt(np.sum((qerr**2),axis=1))/qerr.shape[1]
    u_dd_err = np.sqrt(np.sum((uerr**2),axis=1))/uerr.shape[1]

    #Now calculate the mean or median
    from astropy import stats
    q_mean = np.zeros([q_dd.shape[1]]) #We name this mean, though it could either be Mean or Median
    q_std = np.zeros([q_dd.shape[1]])
    u_mean = np.zeros([u_dd.shape[1]])
    u_std = np.zeros([u_dd.shape[1]])

    for i in range(q_dd.shape[1]):

        mn,md,std = stats.sigma_clipped_stats(q_dd[:,i], sigma=3, maxiters=5)
        if mode == 'median':
            q_mean[i] = md
        else:
            q_mean[i] = mn
        q_std[i] = std/np.sqrt(q_dd.shape[0])
        mn,md,std = stats.sigma_clipped_stats(u_dd[:,i], sigma=3, maxiters=5)
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
        fig,axes = plt.subplots(3,2,figsize=(16,20))

    axes[2,1] = plt.subplot(3,2,6, projection='polar')
    ##### Plot Q, U, P and theta ######
    #The mean values
    if mode == "median":
        axes[0,0].plot(wvs,q_mean,'k',label="Median")
    else:
        axes[0,0].plot(wvs,q_mean,'k',label="Mean")    
    axes[0,1].plot(wvs,u_mean,'k')   
    axes[1,0].plot(wvs,p_mean,'k')
    

    #Make a line at zero
    axes[0,0].axhline(0.,color='r',linestyle='--')
    axes[0,1].axhline(0.,color='r',linestyle='--')

    #Fill in photon/ron error ranges
    axes[0,0].fill_between(wvs,q_mean+q_mean_err,q_mean-q_mean_err,color='k',alpha=0.1,label="Propagated Noise")
    axes[0,1].fill_between(wvs,u_mean+u_mean_err,u_mean-u_mean_err,color='k',alpha=0.1)
    axes[1,0].fill_between(wvs,p_mean+p_mean_err,p_mean-p_mean_err,color='k',alpha=0.1)
    # axes[1,1].fill_between(wvs,theta_mean+theta_mean_err,theta_mean-theta_mean_err,color='k',alpha=0.1)

    #Only plot theta where > 3sigma
    if all_theta:
        where_theta = p_mean > 0
    else:
        where_theta = p_mean > 3*p_mean_err
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
    axes[1,0].plot(inds,3*p_mean_err,'r')

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
    fig.suptitle("{}, {}, t_exp = {}, Bin size = {}".format(target_name,date,t_ext,binsize),fontsize=24)

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
    axes[2,0].plot(wvs,3*p_mean_err,'r',label=r"3$\sigma$ from zero")
    axes[1,0].plot(wvs,3*p_mean_err,'r',label=r"3$\sigma$ from zero")
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

    if show:
        plt.show()
    if save_path is not None:
        if binsize > 1:
            fn = "{}_{}_Binned.png".format(target_name,date,binsize)
        else: 
            fn = "{}_{}_Binned.png".format(target_name,date)
        plt.savefig(save_path+fn)

def plot_pol_summary_time_bins(master_wvs,master_spec,spec_cube,hwp_ang,n_time_bins=1,mode='mean',xlow=1.15,xhigh=1.325,ylow=-0.02,yhigh=0.02,
    target_name="",date="19850625",t_ext = 0,binsize=1,theta_wrap=180,ldwarf=False,show=True,
    save_path=None,legend_loc ="bottom left",all_theta=False,cmap=None,dt=None,period=1.):
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

    time_snip = spec_cube.shape[0] % n_time_bins
    time_bin_size = spec_cube.shape[0]//n_time_bins

    ### TODO: Add in something so that you can put in the rotational period here and have the colors be cyclic. 
    if dt is not None:
        if time_snip != 0:
            dt_bins = np.mean(np.reshape(dt[:-time_snip],(-1,time_bin_size)),axis=1)
        else:
            dt_bins = np.mean(np.reshape(dt,(-1,time_bin_size)),axis=1)
        phase = (dt_bins % period)/period
    else:
        phase = np.linspace(0,1.0,n_time_bins,endpoint=False)

    time_inds = np.arange(spec_cube.shape[0])
    if cmap is None:
        colormapp = plt.get_cmap('hsv')
    else:
        colormapp = plt.get_cmap(cmap) 
    print(phase)
    colors = colormapp(phase)
    
    pmeans = []
    qmeans = []
    umeans = []
    theta_means = []
    for k in range(n_time_bins):
        # print("time_bin_size = {}".format(time_bin_size)) 
        good_inds = time_inds[np.where((time_inds >= k*time_bin_size) & (time_inds < (k+1)*time_bin_size))]
        # print("Using inds {}".format(good_inds))
        q,u,qerr,uerr,qind,uind = compute_qu_for_obs_sequence(spec_cube[good_inds],hwp_ang[good_inds],run_alignment=False)

        #First calculate the double_difference values
        q_dd = np.nanmean(q,axis=1)
        u_dd = np.nanmean(u,axis=1)
        p_dd = np.sqrt(q_dd**2+u_dd**2)
        theta_dd = 0.5*np.degrees(np.arctan2(u_dd,q_dd))
        theta_dd[theta_dd < 0] +=180

        q_dd_err = np.sqrt(np.sum((qerr**2),axis=1))/qerr.shape[1]
        u_dd_err = np.sqrt(np.sum((uerr**2),axis=1))/uerr.shape[1]

        #Now calculate the mean or median
        from astropy import stats
        q_mean = np.zeros([q_dd.shape[1]]) #We name this mean, though it could either be Mean or Median
        q_std = np.zeros([q_dd.shape[1]])
        u_mean = np.zeros([u_dd.shape[1]])
        u_std = np.zeros([u_dd.shape[1]])

        for i in range(q_dd.shape[1]):

            mn,md,std = stats.sigma_clipped_stats(q_dd[:,i], sigma=3, maxiters=5)
            if mode == 'median':
                q_mean[i] = md
            else:
                q_mean[i] = mn
            q_std[i] = std/np.sqrt(q_dd.shape[0])
            mn,md,std = stats.sigma_clipped_stats(u_dd[:,i], sigma=3, maxiters=5)
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
            where_theta = p_mean > 3*p_mean_err
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
        axes[1,0].plot(inds,3*p_mean_err,'k')

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
        axes[2,0].plot(wvs,3*p_mean_err,'r',label=r"3$\sigma$ from zero")
        axes[1,0].plot(wvs,3*p_mean_err,'r',label=r"3$\sigma$ from zero")
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

        axes[1,0].set_xlabel("Wavelength [arb units.]",fontsize=24)
        axes[1,1].set_xlabel("Wavelength [arb units.]",fontsize=24)
        axes[2,0].set_xlabel("Wavelength [arb units.]",fontsize=24)
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

    if show:
        plt.show()
    if save_path is not None:
        if binsize > 1:
            fn = "{}_{}_Binned.png".format(target_name,date,binsize)
        else: 
            fn = "{}_{}_Binned.png".format(target_name,date)
        plt.savefig(save_path+fn)
    if dt is not None:
        return dt_bins,phase,qmeans,umeans,pmeans,theta_means
    else:
        return qmeans,umeans,pmeans,theta_means
