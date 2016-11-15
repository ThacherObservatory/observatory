
"""

Created on Wed Oct  5 20:57:09 2016

@author: gorg,syao,astrolub

"""
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

def makeGaussian(m0,xcent,ycent,plot=True,hist=False, plotCirc=False, fwhm=20.,arcppx=383.65, center=None,vmin=19.2, vmax=21.0, dir="/Users/sara/python/25Oct2016/IMG00074.FIT"):
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
    sig = fwhm/(2*np.sqrt(2*np.log10(2)))
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

    #George's Circle!
    ycirc, xcirc = np.ogrid[:yd, :xd]

    #Sulfur Mt. Centers x:710 y:885
    #x:1115, y:500 for Thach Obs
    r = 50
    circ = (x-xcent)**2 + (y-ycent)**2 <= r*r
    plot_circ_big = (x-xcent)**2 + (y-ycent)**2 <= r*r + 100
    plot_circ_small = (x-xcent)**2 + (y-ycent)**2 >= r*r - 100
    plot_circ = np.where(plot_circ_big == plot_circ_small)
    mean = np.mean(N[circ])
    std = np.std(N[circ])
    median = np.median(N[circ])

    # Image in magnitudes
    img_mag = m0 - 2.5*np.log10(N/wmean)
    # Plot image
    
    if plot:
        plt.clf()
        plt.ion()
        plt.figure()
        plt.title("Sky brightness")
        if plotCirc:
            img_mag[plot_circ]=0
            plt.imshow(img_mag, vmin=vmin, vmax=vmax, cmap='CMRmap_r')
        else:
            plt.imshow(img_mag, vmin=vmin, vmax=vmax, cmap='CMRmap_r')
        #plt.scatter(xd/2,yd/2,s=30)
        #plt.scatter(xcent,ycent,s=30)
        #plt.plot([xd/2,xcent],[yd/2,ycent],linewidth=1)
        plt.axis('off')
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
        plt.show()

    if hist:
        plt.clf()
        plt.ion()
        plt.figure()
        n, bins, patches = plt.hist(N[circ],bins = 100,histtype='bar')
        y = mlab.normpdf(bins,mean,std)
        plt.plot(bins, y, 'r--')
        plt.xlim(2000,5000)
        plt.ylim(0,2000)
        plt.show()
        return 
    return mean, std, median, N[circ]
    
"""
m-m0 = -2.5log(F/F0)
m0 = 20.78
F0 = wmean

img_magnitude = m0-2.5log10(img/wmean)
"""
TOhist = makeGaussian(20.74,1115,500,plot=False,dir="/Users/sara/python/25Oct2016/IMG00069.FIT")[3]
SMhist = makeGaussian(20.64,710,885,plot=False,dir="/Users/sara/python/25Oct2016/IMG00074.FIT")[3]
dif = SMhist-TOhist
plt.ion()
plt.clf()
plt.figure(1)
data = np.vstack([TOhist,SMhist,dif]).T
plt.xlim(-1000,5000)
plt.ylim(0,2000)
plt.hist(data, bins=1000,label=['TO','SM','dif'])
plt.legend(loc='upper right')
plt.show()
v = np.where(dif<=0)
pcent = len(v)/dif
print dif, pcent
