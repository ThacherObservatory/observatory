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
import time,datetime,glob,pdb
from scipy.stats import sigmaclip
from scipy.stats.kde import gaussian_kde
from scipy.interpolate import interp1d
import matplotlib as mpl
import pandas as pd

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



def get_day_data(year=2015,month=10,day=1,
             path='/Users/jonswift/Dropbox (Thacher)/Observatory/CloudSensor/Data/'):

    """
    Description:
    ------------
    Fetch data from the cloud sensor for a given date
    

    Example:
    --------
    data = get_day_data(year=2015,month=3,day=6)


    To do:
    ------

    """

    # Set up path and filename
    file = str(year)+'-'+str(month).zfill(2)+'-'+str(day).zfill(2)+'.txt'
    filename = path+file

    test = glob.glob(filename)
    if len(test) == 0:
        print 'File '+file+' not found in directory'
        print '     '+path
        return []

    colspec = [(0,10),(11,22),(23,24),(26,28),(29,30),(31,32),(33,34),(35,36),(37,38),(39,40),
               (42,47),(49,53),(55,59),(60,61),(62,63),(64,67),(69,73),(74,79),(80,83),
               (84,89),(90,91),(92,96),(97,102),(103,108),(109,114),(115,120),(169,174),
               (175,176),(177,181)]
    names =   ['Date','Time','M','RecordType','Error','CloudVal','WindVal','RainVal','SkyVal','Roof',
               'SkyTemp','AmbientTemp','Wind','Wet','Rain','Humidity','DewPoint','CaseTemp','Heater',
               'BLKT','H','Voltage','TipTemp','WetDrop','WetAvg','WetDry','RawWetCt',
               'DayFlag','Daylight']
    data = pd.read_fwf(filename,colspecs=colspec,names=names)

    data = data[data['RecordType'] == '~D']
    fmt = '%Y-%m-%d %H:%M:%S.%f'
    datetimestr = data['Date']+' '+data['Time']
    dtv = []
    doyv = []
    print '... converting times into datetime objects'
    for date in datetimestr:
        if np.int(date[17:19]) == 60:
            date = date[:17]+'00'+date[19:]
            min = np.int(date[14:16])
            min += 1
            date = date[:14]+str(min)+date[16:]

        dt = datetime.datetime.strptime(date,fmt)
        dtv =  np.append(dtv,dt)
        doyv = np.append(doyv,dt.timetuple().tm_yday+np.float(dt.hour)/
                         24.0+dt.minute/(24.0*60.0)+dt.second/(24.0*3600.0)+
                         dt.microsecond/(24.0*3600.0*1e6))

    data['datetime'] = dtv
    data['DOY'] = doyv

    return data





def get_longterm_data(path='/Users/jonswift/Dropbox (Thacher)/Observatory/CloudSensor/Data/'):

    """
    Description:
    ------------
    Fetch long term log data from the cloud sensor
    

    Example:
    --------
    data = get_longterm_data(path='./')


    To do:
    ------


    """

    # Set up path and filename
    file = 'longtermlog.txt'
    filename = path+file

    test = glob.glob(filename)
    if len(test) == 0:
        print 'File '+file+' not found in directory'
        print '     '+path
        return []

    colspec = [(0,10),(11,22),(23,24),(26,28),(29,30),(31,32),(33,34),(35,36),(37,38),(39,40),
               (42,47),(49,53),(55,59),(60,61),(62,63),(64,67),(69,73),(74,79),(80,83),
               (84,89),(90,91),(92,96),(97,102),(103,108),(109,114),(115,120),(169,174),
               (175,176),(177,181)]
    names =   ['Date','Time','M','RecordType','Error','CloudVal','WindVal','RainVal','SkyVal','Roof',
               'SkyTemp','AmbientTemp','Wind','Wet','Rain','Humidity','DewPoint','CaseTemp','Heater',
               'BLKT','H','Voltage','TipTemp','WetDrop','WetAvg','WetDry','RawWetCt',
               'DayFlag','Daylight']
    data = pd.read_fwf(filename,colspecs=colspec,names=names)

    data = data[data['RecordType'] == '~D']
    fmt = '%Y-%m-%d %H:%M:%S.%f'
    datetimestr = data['Date']+' '+data['Time']
    dtv = []
    doyv = []
    print '... converting times into datetime objects'
    for date in datetimestr:
        if np.int(date[17:19]) == 60:
            date = date[:17]+'00'+date[19:]
            min = np.int(date[14:16])
            min += 1
            date = date[:14]+str(min)+date[16:]

        dt = datetime.datetime.strptime(date,fmt)
        dtv =  np.append(dtv,dt)
        doyv = np.append(doyv,dt.timetuple().tm_yday+np.float(dt.hour)/
                         24.0+dt.minute/(24.0*60.0)+dt.second/(24.0*3600.0)+
                         dt.microsecond/(24.0*3600.0*1e6))

    data['datetime'] = dtv
    data['DOY'] = doyv

    return data

