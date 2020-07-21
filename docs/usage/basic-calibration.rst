Basic Calibration
==================

The first step of the DRP is to apply dark subtraction and field flattening to science images. Once an image is loaded as a ``wirc_data`` object, the calibration is performed by the method ``wirc_data.calibrate``. To do this, we need the following calibration files. 

1. Master dark frame. This is a median-combined dark frames taken at the same exposure time and coadd as the science image. In case of exposure time mismatch, the DRP will automatically scale the master dark. However, this is much less reliable at subtracting numerous hot pixels in WIRC data. We typically use about 21 dark images to construct the master dark frame. This is done using the ``wirc_drp.calibration.masterDark`` function.

2. Master flat frame. This is a median-combined flat frames taken using the same filter as the science image. The best practice is to use flat with the PG and the focal plane mask out of the beam. The half-wave plate is also typically removed. Twenty one 1-second exposures with the low lamp is sufficient. This is done using the ``wirc_drp.calibration.masterFlat`` function. 

3. Bad pixel map. In addition to the master flat, the ``wirc_drp.calibration.masterFlat`` function also produces a bad pixel map by finding pixels that deviate significantly in gain from the rest of the array.  This map is used later on in the reduction process to mask them out of spectral extraction.

The basic calibration steps are shown in :doc:`../tutorials/WIRC+Pol_Tutorial_1-Single_File_Spectral_Extraction`.
