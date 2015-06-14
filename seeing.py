######################################################################
# Seeing module written for the SBIG seeing monitor at the Thacher
# Observatory. Individual routines (should) have header comments.
#
# To do:
# ------
# - Data vetting (zeros)
# - Convert the date and time into "datetime" objects in "get_data"
# - Develop simple plotting routines for data including time series
#   FWHM, histograms, and peak to peak intensity fluctuations.
# - Develop routine to read multiple days of data
# 
# History:
# --------
# jswift  2/8/2015: First versions of get_data, fwhm_ave, fwhm_all,
#                   and fwhm_stats written
#
# jswift  4/1/2015: Vetted all FWHM data
# jswift 4/15/2015: Added simple histogram plotting function.
#
######################################################################

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import sigmaclip
from scipy.stats.kde import gaussian_kde
from scipy.interpolate import interp1d
import pdb
import matplotlib as mpl
import datetime

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

def distparams(dist):
    """
    Description:
    ------------
    Return robust statistics of a distribution of data values

    Example:
    --------
    med,mode,interval,lo,hi = distparams(dist)
    """

    from scipy.stats.kde import gaussian_kde
    from scipy.interpolate import interp1d
    vals = np.linspace(np.min(dist)*0.5,np.max(dist)*1.5,1000)
    kde = gaussian_kde(dist)
    pdf = kde(vals)
    dist_c = np.cumsum(pdf)/np.sum(pdf)
    func = interp1d(dist_c,vals,kind='linear')
    lo = np.float(func(math.erfc(1./np.sqrt(2))))
    hi = np.float(func(math.erf(1./np.sqrt(2))))
    med = np.float(func(0.5))
    mode = vals[np.argmax(pdf)]

    disthi = np.linspace(.684,.999,100)
    distlo = disthi-0.6827
    disthis = func(disthi)
    distlos = func(distlo)
    
    interval = np.min(disthis-distlos)

    return med,mode,interval,lo,hi



def get_data(year=2015,month=3,day=6,
             path='/Users/jonswift/Dropbox (Thacher)/Observatory/Seeing/Data/'):

    """
    Description:
    ------------
    Fetch data from the seeing monitor for a given date
    

    Example:
    --------
    data = get_data(year=2015,month=3,day=6)


    To do:
    ------
    Make data reading more robust

    """

    # Set up path and filename
    prefix = 'seeing_log_'
    root = str(year)+'-'+str(month)+'-'+str(day)
    suffix = '.log'

    file = path+prefix+root+suffix

    # Read first section of data (tab delimited and uniform)
    d1 = np.loadtxt(file, dtype=[('time', '|S8'), ('date', '|S10'), ('Fmin', 'i6'),
                                 ('Fmax', 'i6'), ('FWHMave', 'f6'), ('npts', 'i2')],usecols=(0,1,2,3,4,5))

    # Read in second section of data (; delimited and not uniform)
    d2raw = np.loadtxt(file, delimiter=[0],dtype='str')
    d2 = []
    for i in np.arange(len(d2raw)):
        d2.append(d2raw[i][37:90].split(';')[0:-1])

    # create null vectors of interest
    yearv = [] ; monthv = [] ; dayv = [] ; doyv = [] ; time24v = [] ; dt =[]

    # parse data
    
    date = d1['date']
    time = d1['time']
    
    for i in range(len(date)):
        yr = np.int(date[i].split('/')[2])
        yearv = np.append(yearv,yr)

        month = np.int(date[i].split('/')[0])
        monthv = np.append(monthv,month)
        
        day = np.int(date[i].split('/')[1])
        dayv = np.append(dayv,day)
        
        hr  = np.int(time[i].split(':')[0])
        mn  = np.int(time[i].split(':')[1])
        sec = np.int(time[i].split(':')[2])

        time24v = np.append(time24v,hr+mn/60.0+sec/3600.0)

        d = datetime.datetime(yr,month,day,hr,mn,sec,0)
        dt = np.append(dt,d)
        doyv = np.append(doyv,d.timetuple().tm_yday+np.float(hr)/
                         24.0+mn/(24.0*60.0)+sec/(24.0*3600.0))

        
    # Put all data together into a dictionary
    data = {"datetime": dt, "doy": doyv, "timefloat": time24v,
            "time": d1["time"], "date": d1["date"],
            "Fmin": d1["Fmin"], "Fmax": d1["Fmax"],
            "FWHMave": d1["FWHMave"], "npts": d1["npts"],
            "FWHMraw": d2}

    return data


def FWHM_all(data):
    """
    Description:
    ------------
    Extract a numpy array of all FWHM measurements vetted against zeros and 0.08 values
    which seem to be a saturation effect

    """

    raw = data["FWHMraw"]
    new = []
    for r in raw:
        new = np.append(new,r)

    inds, = np.where(new == '')

    if inds:
        new[inds] = '0.0'

    FWHM = new.astype('float')

    keep, = np.where((FWHM != 0.0) & (FWHM != 0.08))
    
    return FWHM[keep]


def FWHM_ave(data,clip=False,sigmas=False):
    """
    Description:
    ------------    
    Computes the average FWHM
    Expects data dictionary in format of "get_data"

    """

    raw = data['FWHMraw']
    sz = len(raw)
    FWHM = np.ones(sz)* np.nan
    sigma = np.ones(sz)* np.nan
    for i in range(sz):
        vals = np.array(raw[i]).astype('float')
        nvals = len(vals)
        minval = np.min(vals)
        maxval = np.min(vals)
        if clip:
            newvals, low, high = sigmaclip(vals,low=clip,high=clip)
            FWHM[i] = np.mean(newvals)
            sigma[i] = np.std(newvals)
        else:
            FWHM[i] = np.mean(vals)
            sigma[i] = np.std(vals)

    if sigmas:
        return [FWHM,sigma]
    else:
        return FWHM


def FWHM_stats(data,all=True,clip=False):
    """
    Description:
    ------------
    Return basic FWHM stats
    """
    
    if all:
        fwhm =  FWHM_all(data)
    elif clip:
        fwhm = FWHM_ave(data,clip=clip)
    else:
        fwhm = FWHM_ave(data)


    # Basic stats
    med = np.median(fwhm)
    mean = np.mean(fwhm)
    fwhm_clip, low, high = sigmaclip(fwhm,low=3,high=3)
    meanclip = np.mean(fwhm_clip)

    # Get mode using kernel density estimation (KDE)
    vals = np.linspace(0,30,1000)
    fkde = gaussian_kde(fwhm)
    fpdf = fkde(vals)
    mode = vals[np.argmax(fpdf)]

    std = np.std(fwhm)
    
    return [mean,med,mode,std,meanclip]
    

    

def vet_FWHM_series(time,raw):

    """
    Description:
    ------------
    Extract a numpy array of all FWHM measurements vetted against zeros and 0.08 values
    which seem to be a saturation effect
    
    Timestamp for data is also input so that vetted data contains corresponding
    timestamp.

    """

    new = []
    newt = []
    for r in raw:
        new = np.append(new,round(r,2))

    inds, = np.where(new == '')

    if inds:
        new[inds] = '0.0'

    keep1, = np.where(new != 0.0)
    new = new[keep1]
    newt = time[keep1]

    keep2, = np.where(new != 0.08)
    new = new[keep2]
    newt = time[keep2]
    
    FWHM = new.astype('float')

    return newt, FWHM

def fwhm_hist(vec):

    plot_params()
    plt.ion()
    plt.figure(33)
    maxval = np.max(vec)
    minval = np.min(vec)
    std = np.std(vec)
    med = np.median(vec)

    vec = vec[vec < 5*std+med]
    plt.hist(vec,bins=100)
    plt.xlabel('FWHM (arcsec)',fontsize=fs)
    plt.ylabel('Frequency',fontsize=fs)

    mpl.rcdefaults()
    return

