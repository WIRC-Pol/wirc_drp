# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18, 19

@author: Kaew Tinyanont

This script looks for data reduced by auto_reduction.py and compute 
(1) Flux SNR
(2) q and u, assuming that a modulator exist
(3) q, u uncertainties,
(4) p and theta

"""


import astropy.io.fits as fits
import wirc_drp.wirc_object as wo 
from wirc_drp.utils import spec_utils as su
import matplotlib.pyplot as plt 
import numpy as np 

import matplotlib
matplotlib.use('Qt4Agg')

import sys, os, glob, gc, time


# q, u, q_err, u_err, q_position, u_position = compute_qu(spec1, spec2, HWP1, HWP2)
#helper function to compute q and u given two spectra cubes
def compute_qu(spec1, spec2, HWP1, HWP2):
	#stack spectra
	# if spec1.shape != spec2.shape:
	if ((round(HWP1,2) - round(HWP2,2))%45) >0.01: #add some tolerance
		print(np.abs((HWP1 - HWP2)%45))
		print("Error, halfwave plate angles (%f, %f) are not orthogonal."%(HWP1,HWP2))
		return None
	else:
		spec_cube = np.stack([spec1, spec2]) #This has to be the same shape
		#align and scale cubes
		aligned_cube = su.align_spectral_cube(spec_cube)
		#scaled_cube = su.scale_and_combine_spectra(aligned_cube, return_scaled_cube = True)
		scaled_cube = aligned_cube

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

if __name__ == "__main__":
	#First, define a base directory. This is specific to hcig1 for the moment. 
	base_dir = "/scr/data/auto_reduction/"
	base_cal = '/scr/data/calibrations/20190317/'
	#base_dir = '.'
	os.chdir(base_dir)


	if len(sys.argv) == 7: #argument given
		date = sys.argv[1]
		object_name = sys.argv[2]
		sky_ref = sys.argv[3] #frame number of the sky reference image for sky subtraction for automatic source finder
		first_file= sys.argv[4]
		last_file = sys.argv[5]
		exp_time = sys.argv[6]
		# y_coord = sys.argv[6]
		#if len(sys.argv) > 2: 
		#	obj_name = sys.argv[2] #set object name

	else: 
		date = input('Type in date in yyyymmdd format: ')
		object_name = input('Object name: ')
		sky_ref = input('image name of sky reference (e.g. 0012): ')
		first_file = input('image name of first science frame (e.g. 0013): ')
		last_file = input('image nam of last science frame (e.g. 0015): ')
		exp_time = input('Exposure time: ')
		# y_coord = input('Y coordinate: ')

	redux_dir = base_dir+date+'/'+object_name+'_%.1fs_auto/'%float(exp_time) #+file_name.split('.')[0] #+'_auto_extracted.fits'
	print("Computing polarization from directory "+redux_dir)

	#check if this date exists
	if os.path.isdir(redux_dir):
		print("Plotting automatically reduced data for date ", date)
	else: 
		print("Auto reduced data for date "+ str(date)+ " not available. Please run auto_reduction.py on that date")

	#this is for the on-the-fly automatic reduction
	#get into the date directory 
	os.chdir(redux_dir)
	
	#ask for the first file number to process: use default
	#first_file = str(input('First file number to process: '))

	if len(first_file) <= 5:
		first_file = first_file.zfill(4)
	else:
		first_file = input("File name %s not valid, type in new first file name e.g. 0015: "%first_file)

	#loop to get new files
	all_files = sorted(glob.glob(redux_dir+'*_auto_extracted.fits')) #get all files in the directory
	# print(redux_dir+'*_auto_extracted.fits', all_files)

	plt.ion()
	#plt.tight_layout()

	"""This is the live output plot, it will show 
	(1) 4 extracted spectra for the whole sequence
	(2) q, u
	(3) p, theta

	To compute this, we follow these steps:
	(1) For the first frame, load the spectra into the 'holding' array corresponding to the HWP angle. 
	(2) Load the second frame, if HWP angle is the same, add that to the holding array. 
	(3) If the HWP angle is different, take the new spectrum and the earliest one in the holding array and compute q and u (then p and theta)
		Note that only one holding array can be non-empty at the same time. 
	"""

	#####Holding list######
	# In order to compute q and u, we need two images of orthogonal HWP angles (e.g. 0 and 45 or 22.5 and 67.5).
	# So we have two holding lists, one for 0/45 ("Q") and another for 22.5/67.5 ("U"). 
	holding = [[],[]] #holding[0] is "Q" and [1] is "U"
	HWP_in_holding = [-99, -99]

	#arrays to keep polarization measurements
	# q_UL = np.array([])
	# q_LR = np.array([])
	# q_UR = np.array([])
	# q_LL = np.array([])
	# u_UL = np.array([])
	# u_LR = np.array([])
	# u_UR = np.array([])
	# u_LL = np.array([])

	all_q 		= []
	all_u 		= []
	all_q_err   = []
	all_u_err   = []
	dd_all_q 		= []
	dd_all_u 		= []
	dd_all_q_err   = []
	dd_all_u_err   = []
	all_q_pos	= [] #UL, LR, UR, or LL
	all_u_pos	= [] #UL, LR, UR, or LL
	all_p 		= []
	all_theta	= []
	all_p_err   = []
	all_theta_err   = []
	dd_all_p 		= []
	dd_all_theta	= []
	dd_all_p_err   = []
	dd_all_theta_err   = []

	number_in_holding = 0
	HWP_angles = np.array([])

	trace_labels = ['Top - Left', 'Bottom - Right', 'Top - Right', 'Bottom - Left']
	#The plots
	fig, ax = plt.subplots(3,2, figsize = (8, 10))

	ax[0,0].set_ylabel('Flux (ADU)', fontsize = 12)
	ax[0,1].set_ylabel('Flux (ADU)', fontsize = 12)


	ax[1,0].set_ylabel('q (%)', fontsize = 12)
	ax[1,1].set_ylabel('u (%)', fontsize = 12)
	ax[1,0].set_xlabel('Spectral pixel', fontsize = 12)
	ax[1,1].set_xlabel('Spectral pixel', fontsize = 12)

	ax[2,0].set_ylabel('p (%)', fontsize = 12)
	ax[2,1].set_ylabel(r'$\theta$ (deg)', fontsize = 12)
	ax[2,0].set_xlabel('Spectral pixel', fontsize = 12)
	ax[2,1].set_xlabel('Spectral pixel', fontsize = 12)

	plt.tight_layout()

	#blank cube for alignment
	align_cube = []
	first_time = True
	#now the actual loop
	while True:
		all_files = sorted(glob.glob('*_auto_extracted.fits')) #get all files in the directory
		# print(all_files)

		time.sleep(0.1)
		for file_name in all_files:
			#print("First file is", first_file)
			fn_string = file_name.split('_auto_extracted.fits')[0]
			# print(fn_string)
			if int(fn_string[-4:]) < int(first_file) or int(fn_string[-4:]) > int(last_file): #before first file
				pass
				#print(file_name, first_file)
			else: #after first file, do something
				#get exposure time and filter
				header = fits.getheader(file_name)
				exp_time = header['EXPTIME']
				filter_name = header['AFT'][0] #just the name, J or H
				if filter_name not in ['J', 'H']:
					filter_name = input('Invalid filter name (%s) from header, type J or H'%s)

				data = wo.wirc_data(wirc_object_filename=file_name, verbose=False, load_full_image = False )

				#Read spectra and half wave plate angle
				spectra = data.source_list[0].trace_spectra

				if len(align_cube) == 0:
					align_cube = spectra
				else: 
					# print(align_cube.shape, spectra.shape)
					spec_cube = np.stack([align_cube,spectra])
					aligned_cube = su.align_spectral_cube(spec_cube, ref_trace = None)
					scaled_cube = su.scale_and_combine_spectra(aligned_cube, return_scaled_cube = True)

					spectra = scaled_cube[1]

				HWP = float(data.header['HWP_ANG'])

				#Fix HWP angle degeneracy
				if HWP > 90 and HWP < 180:
					HWP = HWP - 90
				elif HWP < 0:
					HWP = HWP + 90

				#First of all, plot the spectra
				ax[0, 0].plot( spectra[0,1,:], 'b', label = '%s'%trace_labels[0])
				ax[0, 0].plot( spectra[1,1,:], 'r', label = '%s'%trace_labels[1])
				ax[0, 1].plot( spectra[2,1,:], 'b', label = '%s'%trace_labels[2])
				ax[0, 1].plot( spectra[3,1,:], 'r', label = '%s'%trace_labels[3])
				if first_time:
					ax[0, 0].legend(loc = 'lower center', frameon = False, fontsize = 12)
					ax[0, 1].legend(loc = 'lower center', frameon = False, fontsize = 12)
					first_time = False
				#Determine which pair this belongs to:

				if (HWP//22.5)%2 == 0: #This is for HWP = 0, 45, etc ("Q")
					HWP_ind = 0
					# print()
				elif (HWP//22.5)%2 == 1: #This is 22.5, 67.5, etc ("U") 
					HWP_ind = 1
				else:
					print("Don't support angles apart from 0, 22.5, 45, 67.5, and their multiples yet.")
					break

				#The rest of this will be govenred by the HWP angle. 
				#If this is the first frame, load it into the holding list
				if len(holding[HWP_ind]) == 0:
					holding[HWP_ind] += [spectra]
					HWP_in_holding[HWP_ind] = HWP #again, there can only be one HWP in holding

					ax[0,HWP_ind].set_title('HWP angle %.2f in holding, %d images'%(HWP, len(holding[HWP_ind])))
				else: #something in the holding list
					if HWP_in_holding[HWP_ind] == HWP: # in case of same angle, just add the new spectrum to the list
						holding[HWP_ind] += [spectra]
						ax[0,HWP_ind].set_title('HWP angle %.2f in holding, %d images'%(HWP, len(holding[HWP_ind])))
					else: #Different angle, call compute q, u to get q and u
						#Note that for each pair of images at different HWP angle, we get 
						ax[0,HWP_ind].set_title('Computing q, u from HWP angles %.1f and %.1f'%(HWP,HWP_in_holding[HWP_ind]))
						spec1 = spectra
						spec2 = holding[HWP_ind][0] #Take the 0th, i.e. first element in the holding list. 
						HWP2 = HWP_in_holding[HWP_ind] #same with the HWP angle. 
						if (HWP2-HWP)%45 != 0:
							print("Bug: two halfwave plates are not orthogonal.")
						del holding[HWP_ind][0] #Then delete it
						q, u, q_err, u_err, q_position, u_position = compute_qu(spec1, spec2, HWP, HWP2)
						all_q     += [q[0], q[1]]
						all_u     += [u[0], u[1]]
						all_q_err += [q_err[0], q_err[1]]
						all_u_err += [u_err[0], u_err[1]]
						all_q_pos += [q_position[0], q_position[1]]
						all_u_pos += [u_position[0], u_position[1]]

						#Double differencing
						q_dd = (q[0]+q[1])/2 
						u_dd = (u[0]+u[1])/2 
						q_err_dd = (q_err[0] + q_err[1])/2/np.sqrt(2)
						u_err_dd = (u_err[0] + u_err[1])/2/np.sqrt(2)
						dd_all_q     += [ q_dd ]
						dd_all_u     += [ u_dd ]
						dd_all_q_err += [ q_err_dd ]
						dd_all_u_err += [ u_err_dd ]


						print(np.array(all_q).shape)
						#Save measured q and u into a file. 
						#np.save(redux_dir+'%s_%s_qu.npy'%(object_name,  last_file), np.array([all_q, all_u, all_q_err, all_u_err, all_q_pos, all_u_pos]))
						#np.save(redux_dir+'%s_%s_qu_double_diff.npy'%(object_name, last_file), np.array([dd_all_q, dd_all_u, dd_all_q_err, dd_all_u_err]))

						colors = ['r','y','b','c']
						# Plot q and u into the collective plot! This one is single differencing version
						#for ind in range(2):
						# 	ax[1,0].errorbar(range(len(q[ind])), 100*q[ind], yerr = 100*q_err[ind], color = colors[q_position[ind]], alpha = 0.2 )
						# 	ax[1,1].errorbar(range(len(u[ind])), 100*u[ind], yerr = 100*u_err[ind], color = colors[q_position[ind]], alpha = 0.2 )
						# 	ax[1,0].axvline(0)
						# 	ax[1,1].axvline(0)

						#Plot double differencing values
						ax[1,0].errorbar(range(len(q_dd)), 100*q_dd, yerr = 100*q_err_dd, alpha = 0.2 )
						ax[1,1].errorbar(range(len(u_dd)), 100*u_dd, yerr = 100*u_err_dd, alpha = 0.2 )
						ax[1,0].axvline(0)
						ax[1,1].axvline(0)

							# ax[1,0].plot(range(len(q[ind])),  100*q_err[ind], alpha = 0.5 )
							# ax[1,1].plot(range(len(u[ind])),  100*u_err[ind], alpha = 0.5 )



						#compute current median q and u
						q_med = np.median(np.array(all_q), axis = 0)
						u_med = np.median(np.array(all_u), axis = 0)
						q_std = np.std(np.array(all_q), axis = 0)/np.sqrt(len(all_q))
						u_std = np.std(np.array(all_u), axis = 0)/np.sqrt(len(all_u))
						#double difference
						q_med_dd = np.median(np.array(dd_all_q), axis = 0)
						u_med_dd = np.median(np.array(dd_all_u), axis = 0)
						q_std_dd = np.std(np.array(dd_all_q), axis = 0)/np.sqrt(len(dd_all_q))
						u_std_dd = np.std(np.array(dd_all_u), axis = 0)/np.sqrt(len(dd_all_u))
						#LIVE SNR
						#HARD CODED AREA: FIX THIS
						q_med_med = np.median(q_med_dd[120:180])
						u_med_med = np.median(u_med_dd[120:180])
						q_std_med = np.median(q_std_dd[120:180])
						u_std_med = np.median(u_std_dd[120:180])					


						# #remove old line and plot a new one
						try: 
							med_q_line.remove()
							med_u_line.remove()
						except:
							pass
						# med_q_line =  ax[1,0].errorbar(range(len(q_med)), q_med*100, yerr = 100*q_std, alpha = 1, color = 'k')
						# med_u_line =  ax[1,1].errorbar(range(len(u_med)), u_med*100, yerr = 100*u_std, alpha = 1, color = 'k')
						med_q_line =  ax[1,0].errorbar(range(len(q_med_dd)), q_med_dd*100, yerr = 100*q_std_dd, alpha = 1, color = 'k')
						med_u_line =  ax[1,1].errorbar(range(len(u_med_dd)), u_med_dd*100, yerr = 100*u_std_dd, alpha = 1, color = 'k')

						ax[1,0].set_title('Median q = %.2f $\pm$ %.2f %%  SNR = %.2f'%(100*q_med_med, 100*q_std_med, q_med_med/q_std_med))
						ax[1,1].set_title('Median u = %.2f $\pm$ %.2f %%  SNR = %.2f'%(100*u_med_med, 100*u_std_med, u_med_med/u_std_med))



						#plot limits
						ax[1,0].set_ylim([-5,5])
						ax[1,1].set_ylim([-5,5])
						ax[1,0].set_xlim([50,150])
						ax[1,1].set_xlim([50,150])

						#Now compute degree and angle of polarization
						p = np.sqrt(q**2 + u**2)
						theta = 0.5*np.arctan2(u,q)
						#uncertainties
						p_err = 1/p*np.sqrt( (q*q_err)**2 + (u*u_err)**2)
						theta_err = (2/p**2) * np.sqrt((q*u_err)**2 + (u*q_err)**2)

						#Double differencing
						p_dd = np.sqrt(q_dd**2 + u_dd**2)
						theta_dd = 0.5*np.arctan2(u_dd,q_dd)
						#uncertainties
						p_err_dd = 1/p_dd*np.sqrt( (q_dd*q_err_dd)**2 + (u_dd*u_err_dd)**2)
						theta_err_dd = (2/p_dd**2) * np.sqrt((q_dd*u_err_dd)**2 + (u_dd*q_err_dd)**2)

						#add to the list
						all_p += [p[0], p[1]]
						all_theta += [theta[0], theta[1]]
						all_p_err += [[p_err[0], p_err[1]]]
						all_theta_err += [theta_err[0], theta_err[1]]

						#add to the list
						dd_all_p += [p_dd]
						dd_all_theta += [theta_dd]
						dd_all_p_err += [p_err_dd]
						dd_all_theta_err += [theta_err_dd]
						#Plot p and theta
						
						# for ind in range(2):
						# 	ax[2,0].errorbar(range(len(p[ind])),     100*p[ind], yerr = 100*p_err[ind], color = colors[q_position[ind]], alpha = 0.2 )
						# 	ax[2,1].errorbar(range(len(theta[ind])), 100*theta[ind], yerr = 100*theta_err[ind], color = colors[q_position[ind]], alpha = 0.2 )

						#Plot double differenced
						HWPcol = ['cyan', 'green']
						ax[2,0].errorbar(range(len(p_dd)),     100*p_dd, yerr = 100*p_err_dd, color = HWPcol[HWP_ind],alpha = 0.1 )
						ax[2,1].errorbar(range(len(theta_dd)), 100*theta_dd, yerr = 100*theta_err_dd, color = HWPcol[HWP_ind],alpha = 0.1 )
							


						#compute current median p and theta
						p_med = np.median(np.array(all_p), axis = 0)
						theta_med = np.median(np.array(all_theta), axis = 0)
						p_std = np.std(np.array(all_p), axis = 0)/np.sqrt(len(all_p))
						theta_std = np.std(np.array(all_theta), axis = 0)/np.sqrt(len(all_theta))


						#Vectorial calculation of p and theta
                        #print(current_pol_vec.shape)
						current_pol_vec = np.stack([np.array(dd_all_q), np.array(dd_all_u)])
						print('pol vector shape is ',current_pol_vec.shape)
						mean_pol = np.mean(current_pol_vec, axis = 1)
						print('mean_pol shape is ',mean_pol.shape)
						std_pol = np.std(current_pol_vec, axis = 1)/np.sqrt(current_pol_vec.shape[1])

						mean_p = np.sqrt(mean_pol[0]**2 + mean_pol[1]**2 )
						std_p = 1/mean_p *np.sqrt( np.sum(mean_pol*std_pol, axis = 0)**2)

						mean_theta = 0.5*np.arctan2(mean_pol[1],mean_pol[0])
						std_theta = (2/mean_p**2) * np.sqrt((mean_pol[0]*std_pol[1])**2 + (mean_pol[1]*std_pol[0])**2)

						#compute current median p and theta, double diff
						p_med_dd = np.median(np.array(dd_all_p), axis = 0)
						theta_med_dd = np.median(np.array(dd_all_theta), axis = 0)
						p_std_dd = np.std(np.array(dd_all_p), axis = 0)/np.sqrt(len(dd_all_p))
						theta_std_dd = np.std(np.array(dd_all_theta), axis = 0)/np.sqrt(len(dd_all_theta))

						#Mark the area where p < 3dp
						poor_snr = np.where( p_med < 3*p_std)

						# #remove old line and plot a new one
						try: 
							med_p_line.remove()
							#med_p_bad.remove()
							med_theta_line.remove()
							med_vec_p_line.remove()
							#(med_vec_p_line.pop(0)).remove()
							(med_vec_theta_line.pop(0)).remove()	
						except:
							print('poop1')
							pass

						#try:
					
						#except:
						#	print('poop2')
						#	pass
						# med_p_line     =  ax[2,0].errorbar(range(len(p_med)), p_med*100, yerr = 100*p_std, alpha = 1, color = 'k')
						med_p_line     =  ax[2,0].errorbar(range(len(p_med_dd)), (p_med_dd)*100, yerr = 100*p_std_dd, ls = '--', alpha = 0.8, color = 'k')

						med_theta_line =  ax[2,1].errorbar(range(len(theta_med_dd)), np.degrees(theta_med_dd)+90, yerr = np.degrees(theta_std_dd),ls= '--', alpha = 0.8, color = 'k')
						med_med_theta = np.degrees(np.median(np.nan_to_num(theta_med_dd)))
						med_std_theta = np.degrees(np.median(np.nan_to_num(theta_std_dd)))

						#Nem's p and theta
						med_vec_p_line     =  ax[2,0].errorbar(range(len(mean_p)), mean_p*100, yerr = 100*std_p, alpha = 1, color = 'k')
						#med_vec_p_line = ax[2,0].plot(range(len(mean_p)), mean_p*100, alpha = 1, color = 'k')

						#med_vec_theta_line =  ax[2,1].errorbar(range(len(mean_theta)), np.degrees(mean_theta), yerr = np.degrees(std_theta), alpha = 1, color = 'k')
						med_vec_theta_line = ax[2,1].plot(range(len(mean_theta)), np.degrees(mean_theta),  alpha = 1, color = 'k')

						med_vec_med_theta = np.degrees(np.median(np.nan_to_num(mean_theta)))
						med_vec_std_theta = np.degrees(np.median(np.nan_to_num(std_theta)))
						#This is part of p where p < 3 sigma_p, so probably zero polarization
						#med_p_bad =  ax[2,0].plot(np.arange(len(p_med))[poor_snr], (p_med[poor_snr])*100 , alpha = 1, color = 'r', marker = '.', ls = 'None')
						#plot limits
						ax[2,0].set_ylim([0,7])
						ax[2,1].set_ylim([20,50])
						ax[2,0].set_xlim([50,150])
						ax[2,1].set_xlim([50,150])
						ax[2,0].axhline(6.46*0.85)
						#ax[2,0].axhline(2.88*0.85)
						ax[2,1].axhline(24)

						


				# all_spec_cube.append(data.source_list[0].trace_spectra)
				# HWP_angles.append(data.header['HWP_ANG'])

				# print(HWP_angles)


				# trace_labels = ['Top - Left', 'Bottom - Right', 'Top - Right', 'Bottom - Left']

				# fig.suptitle('%s %s %s'%(date, object_name, file_name))
				# for i,j in enumerate(data.source_list[0].trace_images):
				# 	ax[0,i].clear()
				# 	ax[0,i].imshow(j, origin = 'lower')
				# 	ax[0,i].text(0.1*j.shape[1],0.8*j.shape[0], trace_labels[i], color = 'w')


				# #then extracted Qp, Qm, Up, Um
				# size = len(data.source_list[0].trace_spectra[0,1,:])
				# ax[2,0].clear()
				# ax[2,0].plot(data.source_list[0].trace_spectra[0,1,:], 'b', label = '%s (Qp)'%trace_labels[0])
				# ax[2,0].plot(data.source_list[0].trace_spectra[1,1,:], 'r', label = '%s (Qm)'%trace_labels[1])
				# ax[2,0].set_xlim([size/2-50, size/2+50])
				# ax[2,0].legend()

				# ax[2,1].clear()
				# ax[2,1].plot(data.source_list[0].trace_spectra[2,1,:], 'b', label = '%s (Up)'%trace_labels[2])
				# ax[2,1].plot(data.source_list[0].trace_spectra[3,1,:], 'r', label = '%s (Um)'%trace_labels[3])
				# ax[2,1].set_xlim([size/2-50, size/2+50])
				# ax[2,1].legend()

				#also plot in the collective plot. 
				# ax[0].plot(data.source_list[0].trace_spectra[0,1,:], 'b', label = '%s (Qp)'%trace_labels[0])
				# ax[0].plot(data.source_list[0].trace_spectra[1,1,:], 'r', label = '%s (Qm)'%trace_labels[1])
				# ax[1].plot(data.source_list[0].trace_spectra[2,1,:], 'b', label = '%s (Up)'%trace_labels[2])
				# ax[1].plot(data.source_list[0].trace_spectra[3,1,:], 'r', label = '%s (Um)'%trace_labels[3])

				#save extraction results

				# data.save_wirc_object(base_dir+date+'/'+object_name+'_%.1fs_auto/'%exp_time+file_name.split('.')[0]+'_auto_extracted.fits')#, full_image = True)
				first_file = int(fn_string[-4:])+1


				#DELETE file to not blow up memory usage
				del data 
				gc.collect()
			# plt.tight_layout()
				plt.pause(0.0001)
		#at the end of the for loop, set first_file to the last file, and start again
		plt.savefig("/scr/wircpol/WIRC-Pol-Website/flask/app/static/quick_look_img.png",bbox_inches="tight")
			
	time.sleep(5)
#There's no end in sight!
