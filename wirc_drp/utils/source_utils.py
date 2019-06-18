from wirc_drp.constants import *
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.ndimage import shift
from scipy.signal import fftconvolve 
from astropy.io import ascii as asci
from astropy.io import fits 


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

        print(HWP1, HWP2, sampling_angles_1, sampling_angles_2, q_ind, u_ind)
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