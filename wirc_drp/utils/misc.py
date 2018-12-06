###Miscellaneous functions to help with WIRC+Pol data reduction efforts
from astropy import coordinates
from astropy.time import Time
import astropy.units as u 
import numpy as np 

def compute_object_position(list_of_obs_time, obj_coords, observatory = 'Palomar'):
    """
    This function takes a list of observation times, or just an observation time, taken from FITS header, the object coordinates
    in astropy SkyCoord format, and compute the altitude, hour angle, and parallactic angle for each of the given time. 

    Input:
    	list_of_obs_time: a list of astropy Time object, one Time object is also acceptable 
    	obj_coords: coordinates of the object in astropy SkyCoord format
    	observatory: Default to Palomar, but can be changed to any string recognized by astropy: see astropy.coordinates.EarthLocation.get_site_names()
    """
#     alt = []
#     ha = []
#     pa = []
    #If single time object given, make it into a list
    if type(list_of_obs_time) != list:
        list_of_obs_time = [list_of_obs_time]

    #check if provided location is recognized
    if observatory not in coordinates.EarthLocation.get_site_names():
    	observatory = 'Palomar'
    	print('%s not recognized as a site, default to Palomar')

    #Set location so we can get sidereal time
    list_of_obs_time = [Time(x, location = coordinates.EarthLocation.of_site(observatory)) for x in list_of_obs_time]
    #First we compute the local sidereal time
    LST = [x.sidereal_time('apparent') for x in list_of_obs_time]
    #Then hour angle by subtracting RA from this
    HA = np.array([(x - obj_coords.ra).radian for x in LST]) #now in numpy array, with radian unit
    
    #Compute altitude
    lat = coordinates.EarthLocation.of_site(observatory).lat.radian #Get observatory latitude, default is Palomar
    dec = obj_coords.dec.radian
    
    alt = np.arcsin(np.sin(dec)*np.sin(lat)+np.cos(dec)*np.cos(lat)*np.cos(HA))
    
    #Now we can compute Parallactic Angle
    PA = np.arcsin(np.cos(lat)/np.cos(alt)*np.sin(HA))
    
    #Don't forget the Azimuth
    az = np.arcsin(-np.sin(HA)*np.cos(dec)/np.cos(alt))
    #Return everything IN RADIAN
    return alt, az, HA, PA