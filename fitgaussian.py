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

def makeGaussian(deg=20.,arcppx=383.65, center=None, dir="/Users/george/Dropbox/Astronomy/AllSky Cam/30Sept2016/", filename="sky1.FIT"):
    """
    fwhm: full width half max
    arcppx: sqarcsec per pixel
    center: if gaussian to be centered around certain points, say so in tuple/list
    dir: directory of image
    filename: name of image

    Takes image, makes a 2D gaussian of selected portion of sky
    """
    #Calculates sigma and the number of pixels wide the circle will be
    fwhm = (deg*3600)/arcppx
    sig = fwhm/(2*np.sqrt(2*np.log(2)))
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
    # domain and range of values you want selected out of table for gaussian
    xmin = xcent-fwhm//2
    xmax = xcent+fwhm//2
    ymin = ycent-fwhm//2
    ymax = ycent+fwhm//2
    # Add all datapoints that you want from image to gaussian zeros array, run thru a gaussian function to make data set of gaussian values
    mean = np.mean(img[ymin:ymax,xmin:xmax])
    imgaus[ymin:ymax,xmin:xmax] = (1.0/(np.sqrt(2.0*np.pi)*sig))*(np.e**((-1*(img[ymin:ymax,xmin:xmax]-mean)**2)/(2*sig**2)))
    # plot gaussian distribution of points from original image (norm.pdf normalizes points for you, sig calculated earlier relative to fwhm)
    mean = np.mean(img[ymin:ymax,xmin:xmax])
    plt.clf()
    plt.figure(1)
    plt.plot(img[ymin:ymax,xmin:xmax],norm.pdf(img[ymin:ymax,xmin:xmax],mean,sig))
    plt.show()
    """
    comb outliers?
    """
    return imgaus[ymin:ymax, xmin:xmax]
