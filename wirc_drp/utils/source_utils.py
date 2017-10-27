from wirc_drp.constants import *
import matplotlib.pyplot as plt


def plot_source_traces(source_list, cmap = None, figsize=(8,8)):
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
		ax1.plot(source.trace_spectra[0,0,:],source.trace_spectra[0,1,:], color=c)
		ax1.plot(source.trace_spectra[0,0,:],source.trace_spectra[0,1,:], color=c)
		ax1.plot(source.trace_spectra[0,0,:],source.trace_spectra[0,1,:], color=c)

