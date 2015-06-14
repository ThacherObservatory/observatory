import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
from fitgaussian import *
import robust as rb
import sys, os, time, glob
#import scipy as sp
#import matplotlib.patheffects as PathEffects
import pdb
#import djs_phot_mb as djs
#from select import select
#from astropysics.coords import AngularCoordinate as angcor
#import astropy.io.fits
#from astropy import wcs

######################################################################
# All Sky Module:
# ---------------
# Contains tools for the visualization and analysis of images produced
# by the Starlight Xpress Oculus All-Sky Camera
#
# V.0 Written in March 2015 by J. Swift
# V.1 In process
#
# Updates:
# --------
# jswift 3/25/15: Created progress counter for CreateMovie
# jswift 4/10/15: Changed data_look to show_image
#               - Updated CreateMovie for current version of ffmpeg
#                 (version 2.6.1)
#               - Updated for display of 180 degree lens by cropping
#               - Changed show_image to movie_image and internalized
#                 it into CreateMovie
#
# To do:
# -----
#
#
######################################################################



#----------------------------------------------------------------------#
# done_in:                                                             #
#----------------------------------------------------------------------#

def done_in(tmaster):

    """
    Overview:
    ---------
    Simple routine to print out the time elapsed since input time

    Calling sequence:
    -----------------
    import time
    tstart = time.time()
    (stuff happens here)
    done_in(tstart)

    """

    t = time.time()
    hour = (t - tmaster)/3600.
    if np.floor(hour) == 1:
        hunit = "hour"
    else:
        hunit = "hours"

    minute = (hour - np.floor(hour))*60.
    if np.floor(minute) == 1:
        munit = "minute"
    else:
        munit = "minutes"

    sec = (minute - np.floor(minute))*60.

    if np.floor(hour) == 0 and np.floor(minute) == 0:
        tout = "done in {0:.2f} seconds"
        print tout.format(sec)
    elif np.floor(hour) == 0:
        tout = "done in {0:.0f} "+munit+" {1:.2f} seconds"
        print tout.format(np.floor(minute),sec)
    else:
        tout = "done in {0:.0f} "+hunit+" {1:.0f} "+munit+" {2:.2f} seconds"
        print tout.format(np.floor(hour),np.floor(minute),sec)

    print " "

    return

#----------------------------------------------------------------------#
# get_files:                                                           #
#----------------------------------------------------------------------#

def get_files(prefix='IMG',dir="/Users/jonswift/Dropbox (Thacher)/Observatory/AllSkyCam/Data/",
              suffix='.FIT'):
    
    """
    Overview:
    ---------
    Returns list of files with a user defined prefix and suffix withing a
    specified directory
    
    
    Calling sequence:
    -----------------
    files = get_files('HATp33b',dir='/home/users/bob/stuff/')
    
    """

    files = glob.glob(dir+prefix+"*"+suffix)
    
    fct = len(files)

    return files,fct


#----------------------------------------------------------------------#
# check_ast                                                            #
#----------------------------------------------------------------------#
def check_ast(file):
    """ 

    Overview:
    ---------
    Takes an input file and tests to see if there is astrometry in the
    header. 

    Calling sequence:
    -----------------
    status = check_ast(file)

    status is 1 if there is no astrometry, else status = 0

    """

    image, header = pf.getdata(file, 0, header=True)
    status = 0
    try:
        crval1 = header["CRVAL1"]
    except:
        print "Image has inadequate astrometry information"
        status = 1

    return status



def show_image(file,lowsig=1,hisig=4,skyonly=False):

    # Get image and header
    image, header = pf.getdata(file, 0, header=True)

    # Keep region determined by eye
    image = image[:,200:1270]

    # Region of sky to determine statistics determined by eye
    if skyonly:
        region = image[280:775,200:900]
    else:
        region = image
        
    sig = rb.std(region)
    med = np.median(region)
    mean = np.mean(region)
    vmin = med - lowsig*sig
    vmax = med + hisig*sig

    plt.ion()
    plt.figure(99,figsize=(15,8))
    plt.clf()
    plt.imshow(image,vmin=vmin,vmax=vmax,cmap='gray',
               interpolation='nearest',origin='upper')
    plt.axis('off')
    plt.colorbar()

    return

    
def CreateMovie(FrameFiles, fps=30, filename='AllSkyMovie',lowsig=1.0,hisig=4.0,
                windows=False):
    """
    CreateMovie(FrameFiles)
    
    This function creates a movie called movie.mp4 in the current working directory
    which has a frame rate of 10 frames per second.
    
    Parameters:
    	plotter: This parameter is a function of the following form:
    				def plotter(frame_number)
    			 where frame_number is the current frame that needs to be plotted
    			 using the matplotlib.pyplot library.
    	FrameFiles: list of file names that will be used to make the movie
    	fps: The frames per second. The default is 30.
    	
    Output:
    	The function will create a movie called AllSkyMovie.mp4. Make sure that you don't
    	have any files called movie.mp4 and _tmp*.png in the current working
    	directory because they will be deleted.
    """

    def movie_image(file,lowsig=1,hisig=4):

        # Get image and header
        image, header = pf.getdata(file, 0, header=True)        

        date = header['DATE-OBS']
        time = header['TIME-OBS']
        
        # Keep region determined by eye
        image = image[:,200:1270]

        # Do statistics for image display
        sig = rb.std(image)
        med = np.median(image)
        mean = np.mean(image)
        vmin = med - lowsig*sig
        vmax = med + hisig*sig

        # Plot image
        plt.imshow(image,vmin=vmin,vmax=vmax,cmap='gray',
                   interpolation='nearest',origin='upper')

        plt.annotate(date,[0.08,0.92],horizontalalignment='left',
                     xycoords='figure fraction',fontsize=14,color='white')

        plt.annotate(time,[0.92,0.92],horizontalalignment='right',
                     xycoords='figure fraction',fontsize=14,color='white')

        # Turn axis labeling off
        plt.axis('off')
        
        return
    
    print "Creating movie from all-sky images"
    t = time.time()
    i = 0
    plt.ioff()
    plt.clf()
    plt.figure(987)
    nframes = len(FrameFiles)
    for file in FrameFiles:
        sys.stdout.write(" ...Starting frame no. %i of %i \r" % (i,nframes))
        sys.stdout.flush()

        movie_image(file,lowsig=lowsig,hisig=hisig)
        fname = '_tmp%05d.png'%i

        plt.savefig(fname,bbox_inches='tight',transparent=True, pad_inches=0,frameon=False,
                    dpi=150)

        # System command to make background of PNG file black
        os.system("convert "+fname+" -background black -flatten +matte "+fname)

        plt.clf()

        i += 1

    if windows:
        os.system("del "+filename+".mp4")
    else:
        os.system("rm "+filename+".mp4")
    os.system("ffmpeg -r "+str(fps)+" -i _tmp%05d.png -b:v 20M -vcodec libx264 -pix_fmt yuv420p -s 808x764 "+filename+".mp4")
    if windows:
        os.system("del _tmp*png")
    else:
        os.system("rm _tmp*.png")
    done_in(t)
    
    return

    
def get_lst():
    pass

