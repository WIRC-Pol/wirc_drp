# wirc_drp - The WIRC+POL

Object oriented data reduction pipeline for the WIRC+POL instrument at Palomar Observatory

## Installation

Download or clone the repository and then run:
> python setup.py develop

## Dependencies

The WIRC_DRP relies on several outside packages:
- The image registration package from:
<https://github.com/keflavich/image_registration>
many thanks to the authors.
- The photuilts package: 
https://photutils.readthedocs.io/en/stable/install.html
(please check their citation request: https://photutils.readthedocs.io/en/stable/index.html)

- The OpenCV package. You can probably install it like this: ```pip install opencv-python```. (This should be installed when you run setup.py)

- The scikit-image package: http://scikit-image.org/docs/dev/install.html (This should be installed when you run setup.py)
