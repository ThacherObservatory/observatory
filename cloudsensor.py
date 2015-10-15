###################################################
# Module written to read Boltwood Cloud sensor data
#
# To do:
# ------
# - Read data
# 
# History:
# --------
# jswift  10/20/2015: Initial version
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



def get_data(year=2015,month=10,day=1,
             path='/Users/jonswift/Dropbox (Thacher)/Observatory/CloudSensor/Data/'):

    """
    Description:
    ------------
    Fetch data from the cloud sensor for a given date
    

    Example:
    --------
    data = get_data(year=2015,month=3,day=6)


    To do:
    ------
    Make data reading more robust

    """

    # Set up path and filename
    file = str(year)+'-'+str(month).zfill(2)+'-'+str(day).zfill(2)+'.txt'
    filename = path+file

    # Read first section of data (tab delimited and uniform)
    d1 = np.loadtxt(filename, dtype=[ ('date', '|S10'), ('time', '|S11'), ('tag1', '|S1'),
                                 ('tag2', '|S2')], usecols=(0,1,2,3))

                                      # ('FWHMave', 'f6'), ('npts', 'i2')],usecols=(0,1,2,3,4,5))

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
