# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:57:09 2016

@author: görg,syao,astrolub
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import glob
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

def makeGaussian(m0,plot=True, fwhm=20.,arcppx=383.65, center=None,siglo=5, sighi=.25, dir="/Users/sara/python/25Oct2016/IMG00069.FIT"):
    """
    m0: reading from photometer
    fwhm: full width half max
    arcppx: sqarcsec per pixel
    center: if gaussian to be centered around certain points, say so in tuple/list
    dir: directory of image and image

    Takes image, makes a 2D gaussian of selected portion of sky
    """
    #Calculates sigma and the number of pixels wide the circle will be
    fwhm *= 3600/arcppx
    sig = fwhm/(2*np.sqrt(2*np.log(2)))
    #Creates two arrays: array of image and array of zeros w/ same dimensions
    hdu = fits.open(dir)[0]
    xd = hdu.header['NAXIS1']
    yd = hdu.header['NAXIS2']
    #Creates variables for x and y
    x = np.arange(0, xd, 1, float)
    y = np.arange(0, yd, 1, float)
    y = y[:,np.newaxis]
    # Asssume center of brightness is at center of image unless specified otherwise
    if center:
        y0 = center[1]
        x0 = center[0]
    else:
        y0 = yd//2
        x0 = xd//2
    # Gaussian
    Z = (1.0/(np.sqrt(2.0*np.pi)*sig))*(np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / (2.0*sig**2)))
    # image
    N = fits.getdata(dir)
    # Weighted Mean
    wmean = np.sum(N*Z)/np.sum(Z)
    # Image in magnitudes
    img_mag = m0 - 2.5*np.log(N/wmean)
    # Plot image
    if plot:
        sig = np.std(img_mag)
        med = np.median(img_mag)
        vmin = med - siglo*sig
        vmax = med + sighi*sig

        plt.ion()
        plt.figure(1)
        plt.imshow(img_mag, vmin=vmin, vmax=vmax, cmap='CMRmap_r')
        plt.colorbar()
        plt.title("Sky brightness")
        plt.show()
    return img_mag
"""
m-m0 = -2.5log(F/F0)
m0 = 20.78
F0 = wmean

img_magnitude = m0-2.5log10(img/wmean)
"""
