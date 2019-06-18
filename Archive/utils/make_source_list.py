#This is a helper script to create a list of sources quickly.
#Call this in a data directory, which includes a file called sources.list, which contains 5 columns: source, filter, type, start index, stop index
#comma separated. type can be 'sci' or 'sky', and filter is 'J' or 'H'. 
import sys, os
from astropy.io import ascii as asci
prefix = 'image'

flat_table = asci.read('sources.list')
for i in flat_table:
	name = '%s_%s_%s.list'%(i['source_name'], i['type'], i['filter'])
	f = open(name, mode = 'a')
	for j in range(i['start'], i['stop']+1):
		filename = prefix+'%04d'%j+'.fits'
		if os.path.isfile(filename):
			f.write(filename+'\n')
#except:
#	print('First create a flats.list file with 4 columns (comma separated): type, filter, start index, stop index')

