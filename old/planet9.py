# Nick Edwards
# 09/21/2016
# This code is for all your Planet 9 needs
# K. O'Neill 9/27 minor changes
# K. O'Neill 9/29 additions to findArea and started numPoint
# J. Swift 9/30: - Added some useful functions including plate_scale,
#                  npix, and get_path
#                - Should probably program more of what is in my notebook
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pdb
import matplotlib as mpl


user = 'nick' #'nick' and 'katie' are also defined users

"""
SNR EQUATION
"""
def plot_params(fontsize=16,linewidth=1.5):
    """
    Procedure to set the parameters for this suite of plotting utilities
    """

    global fs,lw

    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    fs = fontsize
    lw = linewidth

    return

plot_params()

def plate_scale(mu=13.5,f=6.486,d=0.7):
    """
    Returns arcseconds per pixel for camera and telescope
    specifications

    mu = size of a pixel in microns
    f = focal ratio of the telescope
    d = diameter of the telescope in meters
    """

    flen = f*d
    arcsec_per_radian = 180.0/np.pi*3600.0
    P = arcsec_per_radian*mu/(1e6*flen)
    return P

# Make sure vatiables are floats to avoid int division
# snr should be between 5 - 20
# mlim should be between 22.5 - 25

# Standard variables
# number of pixels
ps = plate_scale()

def npix(fwhm=3.0,ps=ps):
    numpix = np.pi/4.0 * (fwhm/ps)**2
    return numpix

numpix = npix(fwhm=3.5)


# number of electrons per sec due to background
# corresponds to 21 mag per square arcsec
# Should program function to calculate this
Fb = 3.0

# camera gain
g = 1.9

# zero point magnitude
mzp = 22.5

# signal to noise
SNR = 5
# everyone write their own directory in a comment
# and just uncomment it when you use the code

def get_path(user='katie'):
    if user == 'katie':
        dir = '/Users/ONeill/Astronomy/'
    if user == 'nick':
        dir = '/Users/nickedwards/Downloads/'
    if user == 'jswift':
        dir = '/Users/jonswift/Thacher/Teaching/Classes/Astronomy X Block/Fall 2016/'

    return dir

dir = get_path(user=user)

def integrationTime(snr,mlim):
    snr = np.array(snr)
    mlim = np.array(mlim)
    t = (numpix*Fb)/(((g/snr)*(10.0**(-.4*(mlim-mzp))))**2)
    return t

def Mlim(time,seeing):
    #to put time into seconds for equation
    time = np.array(time)*60
    seeing = np.array(seeing).flatten()
    numpix = np.array([])
    for i in range(len(seeing)):
        numpix = np.array(np.append(numpix,npix(fwhm=seeing[i])))
        print i
    numpix = np.reshape(numpix,(100,100))
    seeing = np.reshape(seeing,(100,100))
    #pdb.set_trace()
    mlim = mzp - 2.5*np.log10((SNR/g)*np.sqrt(numpix*Fb/time))
    return mlim

def calcSNR(time,mlim):
    time = np.array(time)
    mlim = np.array(mlim)
    snr = np.sqrt(time/(numpix*Fb))*g*(10.0**(-.4*(mlim-mzp)))
    return snr

def contourTime(s=[5.0,40.0],m=[18,24.0]):
    s = np.linspace(s[0],s[1],1000)
    m = np.linspace(m[0],m[1],1000)
    snr, mlim = np.meshgrid(s,m)
    time = integrationTime(snr,mlim)
    timeMins = time/60
    plt.ion()
    plt.figure()
    plt.clf()
    levels = np.array([1,5,10,15,30,60,90,120])
    plot = plt.contour(snr,mlim,timeMins,levels,linewidth=2,colors='k')
    plt.clabel(plot,fontsize=14,fmt='%1.0f',inline=True)
    #plt.colorbar(plot,label='Time (mins)')
    plt.xlabel('SNR',fontsize=18)
    plt.ylabel(r'$m_{\rm{lim}}$',fontsize=21)
    plt.savefig("SNR_Mlim.png",dpi=300)
    #plt.title('Integration time as a\nfunction of SNR and Mlim in V band',font)

def contourMlim(s=[.5,5.0],t=[5.0,60.0]):
    s = np.linspace(s[0],s[1],100)
    t = np.linspace(t[0],t[1],100)
    fwhm, time = np.meshgrid(s,t)
    mlim = Mlim(time,fwhm)
    plt.ion()
    plt.figure()
    plt.clf()
    plot = plt.contour(time,fwhm,mlim,25,cmap='inferno')
    plt.clabel(plot, inline=True, fontsize=10)
    plt.colorbar(plot,label='Mlim')
    plt.xlabel('Time (mins)')
    plt.ylabel('FWHM')
    plt.title('Limiting magnitude as a\nfunction of FWHM and time')

"""
SKY AREA
"""
RA = np.loadtxt(dir+"P9_GP_only.txt")[:,0]
RA = np.append(RA[0:6],RA[8:len(RA)])
Dec = np.loadtxt(dir+"P9_GP_Only.txt")[:,1]
Dec = np.append(Dec[0:6],Dec[8:len(Dec)])

def p9Region():
    plt.ion()
    plt.figure('p9')
    plt.clf()
    plt.plot(Dec,RA,'.')
    plt.xlabel('Dec')
    plt.ylabel('RA')
    plt.title('Region, in Dec and RA\nwhere Planet 9 could be')


def findArea(n):
    #improved but ISSUES EVERYWHERE
    #for certain values of n (ex:10) seems to work, but for others error
    #"A value in x_new is above the interpolation range"

    # These were chosen incorrectly
    # Additionally, these arrays need to be ordered properly for
    # the interpolation to work
    lowerDec = Dec[5:16]
    lowerRA = RA[5:16]
    i = np.argsort(lowerDec)
    # N: Why do we need to sort the Dec and why not sorting RA?
    # Dec is the independent variable.
    lowerDec = lowerDec[i] ; lowerRA = lowerRA[i]
    # N: What is going on here?
    # sorting the dec values so they are monotically increasing. Must also
    # rearrange the corresponding RA values

    upperDec = np.append(Dec[15:],Dec[0:6])
    upperRA = np.append(RA[15:],RA[0:6])
    i = np.argsort(upperDec)
    upperDec = upperDec[i] ; upperRA = upperRA[i]
    # N: Same questions as above

    lower_interpolate = interp1d(lowerDec, lowerRA, kind='linear')
    upper_interpolate = interp1d(upperDec, upperRA, kind='linear')


    #define delta x - constant
    delta_x = (np.max(upperDec)-np.min(upperDec))/n
    # create equally spaced values between low and high point
    n_Dec = np.linspace(np.min(upperDec),np.max(upperDec),n)

    # create delx variation for use in loop (midpoint of rectangle)
    del_x2 = delta_x/2
    Area = []
    # height = (upper_interp - lower_interp)
    # width = delta_x * cos(dec)
    # Sum = np.sum(height * width)

    width = delta_x

    # Many little things were wrong here, starting with cosine takes
    # an argument in radians
    # Width doesn't change throughout the Riemann Sum
    # Index must go to n-1, not n, for the way you defined your
    # rectangles.
    for i in range(n-1): # N: Why n-1? and not just n?
        # rectangles can't extend beyond the boundary of the region

        #width = np.cos(n_Dec[i]) * delta_x
        # cos(Dec) correction should be on the height and evaluated
        # at the midpoint of the rectangles.
        height = (upper_interpolate(n_Dec[i]+del_x2) - \
                 lower_interpolate(n_Dec[i]+del_x2)) * \
                 np.cos(np.radians(n_Dec[i]+del_x2))
        Area = np.append(Area,height*width)
    TotalArea = np.sum(Area)

    return TotalArea


def numPoint(n=100):
    area = findArea(n)
    field = (20.9/60.0)**2
    pointings = area/field

    return pointings
