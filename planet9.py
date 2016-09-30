# Nick Edwards
# 09/21/2016
# This code is for all your Planet 9 needs
# K. O'Neill 9/27 minor changes
# K. O'Neill 9/29 additions to findArea and started numPoint

import numpy as np
import SNR
import matplotlib.pyplot as plt
from scipy import interpolate

"""
SNR EQUATION
"""


# Make sure vatiables are floats to avoid int division
# snr should be between 5 - 20
# mlim should be between 22.5 - 25

# Standard variables
# number of pixels
npix = 25.9
# number of electrons per sec
Fb = 3.0
# gain on camera
g = 1.9
# zero point magnitude
mzp = 22.5

# everyone write their own directory in a comment
# and just uncomment it when you use the code

dir = '/Users/ONeill/Astronomy/'
#dir = '/Users/nickedwards/Downloads/'

def integrationTime(snr,mlim):
    snr = np.array(snr)
    mlim = np.array(mlim)
    t = npix*Fb/np.power((g/snr)*np.power(10.0,-.4*(mlim-mzp)),2)
    return t

def contourTime(s=[5.0,10.0],m=[22.5,25.0]):
    s = np.linspace(s[0],s[1],1000)
    m = np.linspace(m[0],m[1],1000)
    snr, mlim = np.meshgrid(s,m)
    time = integrationTime(snr,mlim)
    timeMins = time/60
    plt.ion()
    plt.figure()
    plt.clf()
    plot = plt.contour(snr,mlim,timeMins,200,cmap='inferno')
    plt.clabel(plot, inline=True, fontsize=10)
    plt.colorbar(plot,label='Time (mins)')
    plt.xlabel('SNR')
    plt.ylabel('Mlim')
    plt.title('Integration time as a\nfunction of SNR and Mlim')

"""
SKY AREA
"""

RA = np.loadtxt(dir+"P9BlackOnly.txt")[:,0]
RA = np.append(RA[0:6],RA[8:len(RA)])
Dec = np.loadtxt(dir+"P9BlackOnly.txt")[:,1]
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
    upperDec = Dec[6:18]
    upperRA = RA[6:18]
    lowerDec = np.append(Dec[0:6],Dec[17:len(Dec)])
    lowerRA = np.append(RA[0:6],RA[17:len(RA)])
    lower_interpolate = interpolate.interp1d(lowerDec, lowerRA, kind='linear')
    upper_interpolate = interpolate.interp1d(upperDec, upperRA, kind='linear')

    np.array(Dec)
    
    #define delta x - constant
    delta_x = ((Dec[15]-Dec[6])/n)
    # create equally spaced values between low and high point
    n_Dec = np.linspace(Dec[6],Dec[15],n)
    #create delx variation for use in loop
    del_x2 = delta_x/2
    Area = []
    # height = (upper_interp - lower_interp)
    # width = delta_x * cos(dec)
    # Sum = np.sum(height * width)
    
    for i in range(n):
        width = np.cos(n_Dec) * delta_x        
        #to correct for upper/lower sum
        height = upper_interpolate(i+del_x2)-lower_interpolate(i+del_x2)
        Area = np.append(Area,height*width[i])
    TotalArea = np.sum(Area)

    return TotalArea
    
def numPoint(n=100):
    area = findArea(n)
    field = 0.17
    pointings = area/field
    
    return pointings
