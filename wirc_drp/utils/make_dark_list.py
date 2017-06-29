"""
This is a short script to go through all files in the directory, find dark frames, and write lists of darks
based on exposure times. 
"""
import glob, os
import astropy.io.fits as fits
import numpy

os.chdir('/hcig1-nas/wircpol/data/20170613/')

#Grab all files
all_files = glob.glob('*.fits')

darks = {} #empty dictionary 
#go through and select files with BrGamma__(2.17) and J__(1.25) in 'FORE' and 'AFT'
for i in all_files:
	if fits.getheader(i)['FORE'] == 'BrGamma__(2.17)' and fits.getheader(i)['AFT'] == 'J__(1.25)':
		#found a dark file, check if the exp time is new
		exp_time = fits.getheader(i)['EXPTIME']
		if exp_time in darks:
			darks[exp_time] += [i] #add file name to that index
		else:
			darks[exp_time] = [i] #create an index with that exposure time

print(darks)

#go through each exposure time and write file
for i in darks.keys():
	out_file = open('dark_{}s.list'.format(i), 'w')
	for item in darks[i]:
		out_file.writelines("{}\n".format(item))

