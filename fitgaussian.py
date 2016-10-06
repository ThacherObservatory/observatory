# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:57:09 2016

@author: gorge
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import glob
import matplotlib.mlab as mlab

def makeGaussian(fwhm = 20., center=None, dir="/Users/george/Dropbox/Astronomy/AllSky Cam/30Sept2016/",filename="sky1.FIT"):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    testimage = glob.glob(dir+filename)
    hdu = fits.open(testimage[0])[0]
    xd = hdu.header['NAXIS1']
    size = (fwhm/180.)*xd

    x = np.arange(1, size, 1, float)
    y = x[:,np.newaxis]
    print x, y

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    gaus = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    std = np.std(gaus)
    med = np.median(gaus)
    mean = np.mean(gaus)

    plt.figure(1)
    plt.plot(gaus,mlab.normpdf(gaus, np.mean(gaus), std))
    plt.show()
    return gaus
