import numpy as np
from astropy.io import fits
import sys


flist_fn = sys.argv[1]

flist=[]

with open(flist) as f:
    for line in f:
        flist.append(line)


print(flist)