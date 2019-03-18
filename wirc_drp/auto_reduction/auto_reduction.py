# -*- coding: utf-8 -*-
"""
Created on Fri June 2 2017

@author: Kaew Tinyanont, Max Millar-Blanchaer, Ricky Nilsson

Utilities to reduce WIRC+Pol data automatically. These are designed to be used
during and/or after an observing run.

To run the auto reduction loop, call python auto_reduction.py, followed by the date
of the observation and optional the object name. 

To start: python auto_reduction.py yyyymmdd object_name sky_ref first_file x_coord y_coord

auto souce finding will be implemented

"""
import astropy.io.fits as fits
import wirc_drp.wirc_object as wo 
import matplotlib.pyplot as plt 
import numpy as np 

import matplotlib
matplotlib.use('Qt4Agg')

import sys, os, glob, gc, time

if __name__ == "__main__":
	prefix = 'image'
	#First, define a base directory. 
	#base_dir = "/Users/kaew/work/wircpol/trials/auto_reduction/data/data_stage/"
	#base_cal = "/Users/kaew/work/wircpol/trials/auto_reduction/data/temp_cals/"
	base_dir = '/scr/data/'
	out_dir =  '/scr/data/auto_reduction/'
	base_cal = '/scr/data/calibrations/20190317/'
	os.chdir(base_dir)


	if len(sys.argv) == 8: #argument given
		date = sys.argv[1]
		object_name = sys.argv[2]
		sky_ref = sys.argv[3] #frame number of the sky reference image for sky subtraction for automatic source finder
		first_file= sys.argv[4]
		last_file = sys.argv[5]
		x_coord = sys.argv[6]
		y_coord = sys.argv[7]
		#if len(sys.argv) > 2: 
		#	obj_name = sys.argv[2] #set object name

	else: 
		date = input('Type in date in yyyymmdd format: ')
		object_name = input('Object name: ')
		sky_ref = input('image name of sky reference (e.g. 0012): ')
		first_file = input('image name of first science frame (e.g. 0013)" ')
		x_coord = input('X coordinate: ')
		y_coord = input('Y coordinate: ')

	#check if this date exists
	if os.path.isdir(date):
		print("Reducing data for date ", date)
	else: 
		print("Data for date "+ str(date)+ " not available. Please run download_data.py in the data directory")

	#this is for the on-the-fly automatic reduction
	#get into the date directory 
	os.chdir(base_dir+date)
	
	#ask for the first file number to process: use default
	#first_file = str(input('First file number to process: '))

	if len(first_file) <= 5:
		first_file = first_file.zfill(4)
	else:
		first_file = input("File name %s not valid, type in new first file name e.g. 0015: "%first_file)

	#loop to get new files
	all_files = sorted(glob.glob('*.fits')) #get all files in the directory
	plt.ion()
	#plt.tight_layout()
	fig, ax = plt.subplots(3 ,4, figsize = (10,10)) #this is to be updated for every image
	# fig2, ax2 = plt.subplots(1, 4, figsize = (12,3)) #this is accumulated
	"""This is the live output plot, it will show for 4 traces, the raw cutouts, the rotated and background subtracted
	cutouts, the extracted spectrum, and the calculated q, u.
	"""
	#now the actual loop
	while True:
		all_files = sorted(glob.glob('%s????.fits'%prefix)) #get all files in the directory
		# all_files = all_files[int(first_file) :] #cut down on prior files, assuming we start from 0001, with some margin

		print(all_files)
		time.sleep(1)
		for file_name in all_files:
			#print("First file is", first_file)
			if int(file_name[-9:-5]) < int(first_file) or int(file_name[-9:-5])>int(last_file): #before first file
				pass
				#print(file_name, first_file)
			else: #after first file, do something
				#print('Processing %s'%file_name)
				#get exposure time and filter
				header = fits.getheader(file_name)
				exp_time = header['EXPTIME']
				#check if a reduced directory exists
				print(os.path.isdir(out_dir+date+'/'+object_name+'_%.1fs_auto'%exp_time))
				if ~os.path.isdir(out_dir+date+'/'+object_name+'_%.1fs_auto'%exp_time):
					os.system("mkdir "+out_dir+date+'/'+object_name+'_%.1fs_auto'%exp_time)

				filter_name = header['AFT'][0] #just the name, J or H
				if filter_name not in ['J', 'H']:
					filter_name = input('Invalid filter name (%s) from header, type J or H'%s)
				#grab calibration files from the archive.
				#flat_fn = base_cal+"cleaned_PG_flat_%s.fits"%filter_name
				#bp_fn =   base_cal+"bad_pix_map.fits"
				#dark_fn = base_cal+"archive_dark_%.1fs.fits"%exp_time
				flat_fn = base_cal+"image0233_master_PG_flat.fits"
				bp_fn =   base_cal+"image0143_bp_map.fits"
				#dark_fn = base_cal+"image0021_master_dark.fits" #1s
				dark_fn = base_cal+"image0093_master_dark.fits" #5s
				#dark_fn = base_cal+"image0063_master_dark.fits" #15s
				#load in data as wirc object, and calibrate
				data = wo.wirc_data(raw_filename = file_name, flat_fn = flat_fn, dark_fn = dark_fn, bp_fn = bp_fn)
				data.calibrate(mask_bad_pixels = False, verbose = False)

				#after calibration, get thumbnails and extract spectra!

				#add source at the given location x,y 
				data.source_list.append(wo.wircpol_source([int(y_coord),int(x_coord)], 'slitless', data.n_sources + 1))
				data.n_sources += 1

				#get cutouts
				data.source_list[0].get_cutouts(data.full_image, data.filter_name, True, cutout_size = 150) 
				#extract spectra
				data.source_list[0].extract_spectra(plot=False, sub_background = True, bkg_sub_shift_size = 45, \
							method = 'optimal_extraction', bad_pix_masking = 1, verbose = False)
				data.source_list[0].rough_lambda_calibration(method=2)

				#plot them on the axis, first the calibrated images
				trace_labels = ['Top - Left', 'Bottom - Right', 'Top - Right', 'Bottom - Left']

				fig.suptitle('%s %s %s'%(date, object_name, file_name))
				for i,j in enumerate(data.source_list[0].trace_images):
					ax[0,i].clear()
					ax[0,i].imshow(j, origin = 'lower')
					ax[0,i].text(0.1*j.shape[1],0.8*j.shape[0], trace_labels[i], color = 'w')

				#then rotated and background subtracted images
				for i,j in enumerate(data.source_list[0].trace_images_extracted):
					ax[1,i].clear()
					ax[1,i].imshow(j, origin = 'lower')
					#ax[1,i].text(0.1*j.shape[1],0.8*j.shape[0], trace_labels[i], color = 'w')

				#then extracted Qp, Qm, Up, Um
				size = len(data.source_list[0].trace_spectra[0,1,:])
				ax[2,0].clear()
				ax[2,0].plot(data.source_list[0].trace_spectra[0,1,:], 'b', label = '%s (Qp)'%trace_labels[0])
				ax[2,0].plot(data.source_list[0].trace_spectra[1,1,:], 'r', label = '%s (Qm)'%trace_labels[1])
				ax[2,0].set_xlim([size/2-50, size/2+50])
				ax[2,0].legend()

				ax[2,1].clear()
				ax[2,1].plot(data.source_list[0].trace_spectra[2,1,:], 'b', label = '%s (Up)'%trace_labels[2])
				ax[2,1].plot(data.source_list[0].trace_spectra[3,1,:], 'r', label = '%s (Um)'%trace_labels[3])
				ax[2,1].set_xlim([size/2-50, size/2+50])
				ax[2,1].legend()

				#also plot in the collective plot. 
				# ax2[0].plot(data.source_list[0].trace_spectra[0,1,:], 'b', label = '%s (Qp)'%trace_labels[0])
				# ax2[0].plot(data.source_list[0].trace_spectra[1,1,:], 'r', label = '%s (Qm)'%trace_labels[1])
				# ax2[1].plot(data.source_list[0].trace_spectra[2,1,:], 'b', label = '%s (Up)'%trace_labels[2])
				# ax2[1].plot(data.source_list[0].trace_spectra[3,1,:], 'r', label = '%s (Um)'%trace_labels[3])

				#save extraction results, write to a temporary file first
				data.save_wirc_object(out_dir+date+'/'+object_name+'_%.1fs_auto/'%exp_time+file_name.split('.')[0]+'_auto_extracted_tmp.fits')#, full_image = True)
				#then change the name to a proper one. 
				os.rename(out_dir+date+'/'+object_name+'_%.1fs_auto/'%exp_time+file_name.split('.')[0]+'_auto_extracted_tmp.fits', \
					out_dir+date+'/'+object_name+'_%.1fs_auto/'%exp_time+file_name.split('.')[0]+'_auto_extracted.fits')

				#reset first file to the current file. 
				first_file = int(file_name[-9:-5])+1


				#DELETE file to not blow up memory usage
				del data 
				gc.collect()
			# plt.tight_layout()
				plt.pause(0.0001)
		#at the end of the for loop, set first_file to the last file, and start again
			
	time.sleep(5)
#There's no end in sight!










