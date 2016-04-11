
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

def load_data_seeing_old(year=2016,month=3,day=20):
    spath = '/Users/nickedwards/Dropbox (Thacher)/Observatory/Seeing/Data/'
    global seeing_data
    seeing_data = s.get_data(path=spath,year=year,month=month,day=day)

def load_data_weather_old(year=2016):
    wpath = '/Users/nickedwards/Dropbox (Thacher)/Observatory/Weather/Data/'
    global weather_data
    weather_data = w.get_data(dpath=wpath,year=year)


#Make this more robust
def interpolate(plot_interp=False,plot_corr=False,seeing='FWHMave',weather_key='OutHum'):

    # FWHM data must be vetted for outliers first
    dt_seeing,seeing = s.vet_FWHM_series(seeing_data['datetime'],seeing_data[seeing])

    # Weather data
    weather   = weather_data[weather_key]

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

    # need to turn datetime objects into numerical dates, first
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

#talk to doc swift about time stamp stuff and how to navigate them
def load_data_seeing(year=2016,month=3,day=20):
    spath = '/Users/nickedwards/Dropbox (Thacher)/Observatory/Seeing/Data/'

    if len(year)==1:
        start_yr = year
        stop_yr = year

    elif len(month)==1:
        start_mn = month
        stop_mn = month

    elif len(day)==1:
        start_dy = day
        stop_dy = day

    elif len(year) > 2 or len(month) > 2 or len(day) > 2:
        print "Fix year, month, or day amount"
        return []


    #elif
    #if date >= dt1 and date <= dt2:
    # append

def load_data_weather(year=2016):
    wpath = '/Users/nickedwards/Dropbox (Thacher)/Observatory/Weather/Data/'

    if len(year)==1:
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
