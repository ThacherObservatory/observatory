##########################################
#
#    To Do (WEDNESDAY):
# Check for weirdness in OutHum
#
#
#   MEETING (MONDAY):
# Comparing datetime objects
#
#
##########################################

#

import weather as w
import seeing as s
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE
import robust as rb
import matplotlib as mpl
from scipy.interpolate import interp1d
import calendar,sys
import pdb
from scipy.stats.stats import pearsonr
from matplotlib.ticker import NullFormatter
import cloudsensor as cs
import datetime
import glob

#Make this more robust
def interpolate(plot_interp=False,plot_corr=False,weather_key='OutHum'):

    dt_seeing = sdatetime
    seeing = FWHMave
    # Weather data
    weather = weather_data[weather_key]

    #changes weather time data to datetime to work with seeing time data
    dt_weather = np.array([t.to_datetime() for t in weather_data['datetime']])

    # Make sure data overlap in time...
    # select out the indices of the weather data that overlap
    # with the seeing data
    # inds is elements in the original order ORDER MATTERS
    inds, = np.where((dt_weather >= np.min(dt_seeing)) &
                     (dt_weather <= np.max(dt_seeing)))
    dt_weather = dt_weather[inds]
    weather = weather[inds]

    inds, = np.where((dt_seeing >= np.min(dt_weather)) &
                     (dt_seeing <= np.max(dt_weather)))
    dt_seeing = dt_seeing[inds]
    seeing = seeing[inds]

    def toTimestamp(d):
        return calendar.timegm(d.timetuple())

    tseeing  = np.array([toTimestamp(d) for d in dt_seeing])
    tweather = np.array([toTimestamp(d) for d in dt_weather])

    interp_func = interp1d(tweather,weather,kind='linear')
    weather_interp = interp_func(tseeing)

    if plot_interp:
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.plot(dt_seeing,seeing,'o',label='FWHM')
        plt.plot(dt_weather,weather,'o',label=weather_key)
        plt.xlim(np.min(dt_seeing),np.max(dt_seeing))
        plt.plot(dt_seeing,weather_interp,'-',label='interpolated humidity')
        plt.legend(loc='best')

    #FINISH THIS MAKING SEPERATE PLOT FOR SEEING VS WEATHER_INTERP
    if plot_corr:
        plt.ion()
        plt.figure(2)
        plt.clf()
        plt.plot(seeing,weather_interp,'o')
        plt.xlabel('FWHM (arcsec)')
        plt.ylabel('Humidity (%)')
    return pearsonr(seeing, weather_interp)

#cloudsensor data function
# 5/11/15 4:20 - 4:35 p
# np.where(weather_data['OutHum'] == '---')
def load_data_seeing(year=[2016],month=[3],day=[20]):
    spath = '/Users/nickedwards/Dropbox (Thacher)/Observatory/Seeing/Data/'
    year = np.array(year)
    month = np.array(month)
    day = np.array(day)

    if len(year)==1:
        start_yr = year
        stop_yr = year

    if len(month)==1:
        start_mn = month
        stop_mn = month

    if len(day)==1:
        start_dy = day
        stop_dy = day

    if len(year) > 2 or len(month) > 2 or len(day) > 2:
        print "Fix year, month, or day amount"
        return []

    if len(year)==2:
        start_yr = year[0]
        stop_yr = year[1]

    if len(month)==2:

        start_mn = month[0]
        stop_mn = month[1]

    if len(day)==2:
        start_dy = day[0]
        stop_dy = day[1]

    start_dt = datetime.datetime(start_yr,start_mn,start_dy)
    stop_dt = datetime.datetime(stop_yr,stop_mn,stop_dy)

    # if one day is loaded this is what happens

    if start_dt - stop_dt == datetime.timedelta(0):
        sdata = s.get_data(path=spath,year=start_dt.year,month=start_dt.month,day=start_dt.day)
        FWHMave = sdata['FWHMave']
        sdatetime = sdata['datetime']
        sdatetime,FWHMave = s.vet_FWHM_series(sdatetime,FWHMave)
        return FWHMave, sdatetime

    # telling user to make chronological dates

    if stop_dt - start_dt < datetime.timedelta(0):
        print "Fix either year, month, and/or day so that the dates a chronological"
        return []

    sfiles = glob.glob(spath+'seeing_log_2*')

    dt = []
    for f in sfiles:
        year,month,day = np.array(f.split('/')[-1].split('_')[-1].split('.')[0].split('-')).astype('int')
        dt = np.append(dt,datetime.datetime(year,month,day))

    inds, = np.where((dt>=start_dt)&(dt<=stop_dt))
    goodfiles = np.array(sfiles)[inds]
    gooddates = dt[inds]

    FWHMave = []
    sdatetime = []
    for i in range(len(inds)):
        f = goodfiles[i]
        t = gooddates[i] #(.year .month. day)
        sdata = s.get_data(path=spath,year=t.year,month=t.month,day=t.day)
        #talk to doc swift about indices
        FWHMave = np.append(FWHMave,sdata['FWHMave']) #vet series
        sdatetime = np.append(sdatetime,sdata['datetime'])

    sdatetime,FWHMave = s.vet_FWHM_series(sdatetime,FWHMave)

    return FWHMave, sdatetime

def load_data_weather(year=[2016]):
    wpath = '/Users/nickedwards/Dropbox (Thacher)/Observatory/Weather/Data/'

    if len(year)==1:
        year = year[0]
        weather_data = w.get_data(dpath=wpath,year=year)
        return weather_data

    elif len(year)==2:
        start_yr = year[0]
        stop_yr = year[1]

        if stop_yr < start_yr:
            print "Fix year order, so that they are chronological"
            return []

        #write loop that can deal with year differences bigger than 2
        #np.arange(2014,2018,1)
        #Out[8]: array([2014, 2015, 2016, 2017])
        #year_duration = np.arange(start_yr,stop_yr,1)
        #data = []
        #for i in year_duration[i]:
            #data[]
        if stop_yr-start_yr > 1:
            print "Fix coming soon, please only have year increments of only 1"
            return []

        start_data = w.get_data(dpath=wpath,year=start_yr)
        stop_data = w.get_data(dpath=wpath,year=stop_yr)
        # fix dtypes
        weather_data = {}
        for key in start_data.keys():
            weather_data[key] = np.append(start_data[key],stop_data[key])
        return weather_data

    else:
        print "Fix year amount."
        return []

def vet(data):
    np.delete(np.array(data), np.where(np.array(data) == '---'))