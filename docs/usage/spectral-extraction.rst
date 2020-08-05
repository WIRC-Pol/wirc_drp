Spectral Extraction
======================

In this step, the 2D spectrum in WIRC+Pol data is extracted into 1D spectrum. The spectral extraction step is performed by the method ``wircpol_source.extract_spectra``, which directly calls the function ``spec_utils.spec_extraction``. See :doc:`../tutorials/WIRC+Pol_Tutorial_1-Single_File_Spectral_Extraction` and :doc:`../tutorials/WIRC+Pol_Tutorial_3-Reducing_a_Modulation_Sequence` for examples of how to run this function. Here we describe the inputs and each step of spectral extraction. 

Inputs for Spectral Extraction
---------------------------------
1. Science thumbnails: an array with shape [4, width, width] with four small thumbnails of the four spectral traces
2. Background thumbnails: same-shape array with background thumbnails
3. Data quality thumbnails: same-shape array containing bad pixel information 

Key Options for Spectral Extraction
------------------------------------
1. Optimal extraction (``method = 'optimal_extraction'``): the optimal extraction algorighm of `Horne 1986 <https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H/abstract>`_. This algorithm weighs the significant of each pixel by its signal to noise ratio, thereby suppressing noise from low SNR pixels. This is the default extraction algorighm. 

2. Simple extraction (``method = 'sum_across_trace'``): in this simpler method, all pixels in the spatial direction are simply summed with equal weight. It may produce noisier 1D spectrum, but suffers less biases from the weighing algorithm. 

3. Bad pixel masking (``bad_pix_masking``): The optimal extraction algorighm allows us to easily mask bad pixels out from spectral extraction. 

4. Extraction Range (``spatial_sigma``): The DRP automatically detect the trace inside each thumbnail and measure its spatial profile. User can select how many "sigmas" in each direction to extract the spectrum. For example, if the DRP determines that the trace has a Gaussian profile with 2 pixel standard deviation, ``spatial_sigma = 5`` (default) will extract the spectrum +-10 pixels from the center of the trace. 


Steps of Operation
--------------------
1. Find Spectral Trace: the function ``wirc_drp.image_utils.findTrace`` is called to find the trace within each thumbnail. This function a Gaussian profile to each column in the thumbnail to find the peak pixel. It then fits a straight line through it to find the trace. It returns the location, the first estimate of the width of the trace, and the angle of the trace. 

2. Rotate Spectral Trace: WIRC+Pol produce spectral traces that are about 45 degrees with respect to the pixel grid. To extract the spectra, we first rotate this to align with the pixel grid. The angle used is either what measured by the previous step, or what supplied by the user via a keyword ``trace_angle``, which is a list of four angles in degree of the four traces (per convention, the order of the four traces are always upper left, lower right, upper right, lower left). The default rotation algorithm used is openCV bicubic interpolation. (To be consistent, the rotation method can only be changed by adjusting the DRP code. See the ``wirc_drp.spec_utils.frame_rotate`` function for the keywords.)

3. Determine Extraction Range: After rotation, the function ``image_utils.traceWidth_after_rotation`` is called to recompute the trace width. Then the function ``spec_utils.determine_extraction_range`` is called to robustly find the center of the trace in the rotated thumbnail, and determine the range (in pixel) in the spatial direction over which the spectrum is extracted. 

4. Run Spectral Extraction: Once the thumbnails are prepared and the extraction range determined, the ``spec_utils.optimal_extraction`` or ``spec_utils.sum_across_trace`` is called to extract the spectrum. The output from the function are: array of extracted spectra, array of flux standard deviation, spectral trace widths, spectral trace angle, and the thumbnails used for extraction (background subtracted). The DRP incorporates these outputs into the ``wircpol_source`` object from which the ``extract_spectra`` method was called. 


