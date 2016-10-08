# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:57:09 2016

@author: görg
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import glob
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
    #Make circle range to extract circular section of datapoints from main img (rad = fwhm/2 b/c fwhm = width of portion, rad = 1/2width)
    rad = fwhm/2
        #np.ogrid returns np array of yvalues going down indexed at [0] ([0][1][2]...), xvalues grid indexed at [1] ([0,1,2,3...])
    y,x = np.ogrid[ :yd, :xd]
    circ = (x-xcent)**2 + (y-ycent)**2 <= rad*rad
    # Add all datapoints that you want from image to gaussian zeros array, run thru a gaussian function to make data set of gaussian values
    mean = np.mean(img[circ])
    imgaus[circ] = (1.0/(np.sqrt(2.0*np.pi)*sig))*(np.e**((-1*(img[circ]-mean)**2)/(2*sig**2)))
    # plot gaussian distribution of points from original image (norm.pdf normalizes points for you, sig calculated earlier relative to fwhm)
    mean = np.mean(img[circ])
    plt.clf()
    plt.figure(1)
    plt.plot(img[circ],norm.pdf(img[circ],mean,sig))
    plt.show()
    return imgaus[circ]
