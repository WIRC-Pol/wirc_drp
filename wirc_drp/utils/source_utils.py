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

def get_angles_widths_from_list(filelist, source_number = 0):
    """
    Go through the list of calibrated and extracted files and read out angles for given source_number
    """
    angles = []
    widths = []

    filelist = asci.read(filelist, format = 'no_header')['col1'] 

    for j in filelist:
        hdulist = fits.open(j)
        i = source_number
        try:
            widths += [np.fromstring(hdulist[(2*i)+2].header["WIDTHS"][1:-1], sep = ' ')]
            angles += [np.fromstring(hdulist[(2*i)+2].header["ANGLES"][1:-1], sep = ' ')]
        except:
            print('Widths or angles not available')
    return np.array(widths), np.array(angles)


