#This is a helper script to create a list of files quickly 
#call python make_list.py start_number stop_number list_name
#if there are different chunks, calling this on the same list_name will append it
import sys, os

start = int(sys.argv[1])
stop = int(sys.argv[2])
name = sys.argv[3]

prefix = 'image'

f = open(name, mode = 'a')

for i in range(start, stop+1):
    #print('%04d'%i)
    filename = prefix+'%04d'%i+'.fits'
    if os.path.isfile(filename):
        f.write(filename+'\n')
    else:
        print(filename+' not found')


