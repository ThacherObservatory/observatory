
"""

Created on Wed Oct  5 20:57:09 2016

@author: gorg,syao,astrolub

"""
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
import robust as rb

def makeGaussian(m0,xcent,ycent,plot=True,hist=False, plotCirc=False, fwhm=20.,arcppx=383.65, center=None,vmin=19.2, vmax=21.0, dir="/Users/george/Dropbox/Astronomy/Oculus/25Oct2016/IMG00074.FIT"):
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
        plotHist()
    return mean, std, median, N[circ], wmean

"""
m-m0 = -2.5log(F/F0)
m0 = 20.78
F0 = wmean

img_magnitude = m0-2.5log10(img/wmean)
"""
def plotHist(file1, m01, file2, m02, xcent1=1024, ycent1=1024, xcent2=1024,
             ycent2=1024, dir="/Users/georgelawrence/python/Astronomy/data/",
             ttest=False, **kwargs):
    """
    Compares brightness of two sky brightness images

    Parameters:
    -----------
    file1: string
        name of the first .FITS file
    m01: double
        photometer reading for first file
    file2: string
        name of the second .FITS file
    m01: double
        photometer reading for second file
    xcent1,2; ycent1,2: int
        x, y center coordinates for gaussian magnitude reading for
        first and second image respectively
    dir: string
        directory which contains the images
    ttest: boolean
        if ttest, label the ttest value on the plot
    kwargs:
        bins: anything to pass to bins in plt.hist
        label1: what you want file1 to be labeled as
        label2: what you want file2 to be labeled as


    Returns:
    -----------
    tuple: difference in images (mags/arcsec^2), students ttest statistc
    """
    # Run makeGaussian over the input images
    img1,wmn1= makeGaussian(m01, xcent1, ycent1, plot=False, dir=dir+file1)[3:]
    img2,wmn2= makeGaussian(m02, xcent2, ycent2, plot=False, dir=dir+file2)[3:]

    # Get the photometer magnitude reading for each image, find difference
    mag1 = m01-2.5*np.log10(img1/wmn1)
    mag2 = m02-2.5*np.log10(img2/wmn2)
    dif = np.median(mag1-mag2)
    # Calculate student's t-test statistic for two distributions
    pval= stats.ttest_ind(img2, img1)[1]

    # Plot the results
    plt.figure(1)
    plt.ion()
    plt.clf()

        # Plot histograms of data
    plt.hist(img1, label=kwargs.get("label1", "Image 1"), alpha=.5, bins=
             kwargs.get("bins", 10), color="midnightblue")
    plt.hist(img2, label=kwargs.get("label2", "Image 2"), alpha=.5, bins=
             kwargs.get("bins", 10), color="darkgreen")

        # Plot means of distributions
    plt.axvline(x=rb.mean(img1), color ='red', linewidth = 2)
    plt.axvline(x=rb.mean(img2), color = 'red', linewidth = 2)
        # Annotate the graph:
        # difference
    plt.annotate(r'$dif$=%.2f mags/arcsec'u'\u00B2' %dif, [.01,.93],
                 horizontalalignment='left', xycoords='axes fraction',
                 fontsize='large', backgroundcolor='white')
        # Means
    plt.annotate(r'$\bar{\sigma}$=%.2f flux/px' %rb.mean(img1),
                 [.01,0.86], horizontalalignment='left', xycoords=
                 'axes fraction', fontsize="large", color='midnightblue',
                 backgroundcolor="white")
    plt.annotate(r'$\bar{\sigma}$=%.2f flux/px'%rb.mean(img2), [0.01,0.79],
                 horizontalalignment='left', xycoords='axes fraction',
                 fontsize="large", color='darkgreen', backgroundcolor="white")
    if ttest:
            # Student's T-Test statistic
        plt.annotate(r'$T-Test pval$=%.2E' %pval, [.01,.72],
                     horizontalalignment='left', xycoords='axes fraction',
                     fontsize='large')

    plt.title("Sky brightness")
    plt.xlabel("Flux Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()

    print dif, pval
