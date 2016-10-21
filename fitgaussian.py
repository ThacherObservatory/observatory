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

def makeGaussian(fwhm=20.,arcppx=383.65, center=None, dir="/Users/sara/python/30Sept2016/", filename="sky1.FIT"):
    """
    fwhm: full width half max
    arcppx: sqarcsec per pixel
    center: if gaussian to be centered around certain points, say so in tuple/list
    dir: directory of image
    filename: name of image

    Takes image, makes a 2D gaussian of selected portion of sky
    """
    #Calculates sigma and the number of pixels wide the circle will be
    fwhm *= 3600/arcppx
    sig = fwhm/(2*np.sqrt(2*np.log(2)))
    #Creates two arrays: array of image and array of zeros w/ same dimensions
    hdu = fits.open(dir+filename)[0]
    xd = hdu.header['NAXIS1']
    yd = hdu.header['NAXIS2']
    #img = fits.getdata(dir+filename)
    #imgaus = np.zeros((yd, xd))
    #Creates variables for x and y
    
    x = np.arange(0, xd, 1, float)
    y = np.arange(0, yd, 1, float)
    y = y[:,np.newaxis]

    if center:
        y0 = center[1]
        x0 = center[0]
    else:
        y0 = yd//2
        x0 = xd//2
    #[X,Y] = np.meshgrid(x,y)        
    #Z = (1.0/(np.sqrt(2.0*np.pi)*sig))*(np.exp(-((X-x0)**2+(Y-y0)**2)/(2.0*sig**2))
    Z = (1.0/(np.sqrt(2.0*np.pi)*sig))*(np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / (2.0*sig**2)))
    
    imgZ = 2.5*np.log(Z)
    
    plt.ion()
    plt.clf()
    plt.figure(99,figsize=(15,8))
    plt.imshow(Z,cmap='pink',interpolation='nearest',origin='upper')
    #fig = plt.figure()
    #ax=fig.add_subplot(111,projection='3d')
    #ax.plot_surface(x,y,Z,)
    plt.colorbar()
    #plt.show() 
    
    #def weightedMean(dir="/Users/sara/python/30Sept2016/", filename="sky1.FIT"):
    N = 2.5*np.log(fits.getdata(dir+filename)[0])
    W = imgZ
    
    #mean = np.sum(N)/np.float(len(N))
    wmean = np.sum(N*W)/np.sum(W)
    
    #plt.figure(99,figsize=(15,8))
    #plt.imshow(wmean,cmap='pink',interpolation='nearest',origin='upper')
    #plt.colorbar()
    #plt.show()
    
    return wmean
    