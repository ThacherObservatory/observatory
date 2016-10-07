# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:57:09 2016

@author: görg
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import glob
import matplotlib.mlab as mlab
from scipy.stats import norm

def makeGaussian(fwhm=20., arcppx=383.65, center=None, dir="/Users/george/Dropbox/Astronomy/AllSky Cam/30Sept2016/", filename="sky1.FIT"):
    """
    fwhm: full with half max
    arcppx: sqarcsec per pixel
    center: if gaussian to be centered around certain points, say so in tuple/list
    dir: directory of image
    filename: name of image

    Takes image, makes a 2D gaussian of selected portion of sky
    """
    #Calculates sigma and the number of pixels wide the circle will be
    sig = fwhm/(2*np.sqrt(2*np.log(2)))
    pxnum = (fwhm*3600)/arcppx
    #Creates two arrays: array of image and array of zeros w/ same dimensions
    hdu = fits.open(dir+filename)[0]
    xd = hdu.header['NAXIS1']
    yd = hdu.header['NAXIS2']
    img = fits.getdata(dir+filename)
    imgaus = np.zeros((yd, xd))
    #Creates variables for x and y centers
    if center:
        ycent = center[1]
        xcent = center[0]
    else:
        ycent = yd//2
        xcent = xd//2
    #Add the pixels for the gaussian to the zeros array
    xadd = xcent-pxnum//2
    yadd = ycent-pxnum//2
    imgaus[yadd:yadd+pxnum,xadd:xadd+pxnum] = img[yadd:yadd+pxnum,xadd:xadd+pxnum]
    mean = np.mean(imgaus[yadd:yadd+pxnum,xadd:xadd+pxnum])
    #Make gaussian data set of it and plot
    gaus = gausFunc(imgaus,sig,mean)
    plt.plot(gaus[yadd:yadd+pxnum,xadd:xadd+pxnum],mlab.normpdf(gaus[yadd:yadd+pxnum,xadd:xadd+pxnum],mean,sig))
    plt.axis([2000,5000,0,.05])
    plt.show()
    #print np.max(gaus)
    """
    comb outliers
    """
    return mean


def gausFunc(imgaus, sigma, x0):
    """
    Actual 1D gaussian function
    """
    for line in np.nditer(imgaus,op_flags=['readwrite']):
        for px in np.nditer(line,op_flags=['readwrite']):
            if int(px) != 0:
                px = (1.0/(np.sqrt(2.0*np.pi)*sigma))*(np.e**((-1*(px-x0)**2)/(2*sigma**2)))
    return imgaus
