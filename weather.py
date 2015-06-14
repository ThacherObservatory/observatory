import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import sys
import glob
import os
import pdb
import math

def clean_float(vector):
    """ 
    Simple procedure to produce a float vector from a string vector
    """
    
    bad, = np.where(vector == '--')
    good, = np.where(vector != '--')

    vector[bad] = 'nan'

    return vector.astype('float')


def plot_params(fontsize=20,linewidth=1.5):
    """
    Procedure to set the parameters for this suite of plotting utilities
    """
    
    global fs,lw

    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['ytick.labelsize'] = 18
    fs = fontsize
    lw = linewidth

    return

def distparams(dist):
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


def get_data(year=2012,dpath='./'):
    """
    Procedure to parse data from Davis weather station

    Data file must be named in the following convention:
        WS_data_YYYY.txt
    where YYYY is the year. Also, procedure assumes the file header
    takes up the first 3 lines of the file.

    Inputs are the year of the data and the path to the data file
    """

    filename = 'WS_data_'+str(year)+'.txt'
    test = os.path.exists(dpath+filename)
    if not test:
        print "Cannot find data file!"
        return []
    
    print "Getting weather data for the year of "+str(year)
    # load data
    data = np.loadtxt(filename,dtype='str',skiprows=3)

    # extract data from numpy array
    windhi = data[:,11].astype('float')
    winddir = data[:,12]
    tod = data[:,2]
    time = data[:,1]
    date = data[:,0]

    # unique set of dates
    udate = np.unique(date)

    out = "... {0:.0f} days of data taken in "+str(year)
    print out.format(len(udate))

    # will not consider seconds or smaller denominations of time
    seconds = 0
    milliseconds = 0

    # create null vectors of interest
    yearv = [] ; monthv = [] ; dayv = [] ; doyv = [] ; time24v = [] ; winddir_deg = [] ; dt =[]

    # conversion from compass points to azimuth
    compass   = np.array(['N','NNE', 'NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'])
    direction = np.arange(0,360,22.5)
    
    # parse data
    for i in range(len(date)):
        yr = np.int(date[i].split('-')[2])+2000
        yearv = np.append(yearv,yr)

        month = np.int(date[i].split('-')[0])
        monthv = np.append(monthv,month)
        
        day = np.int(date[i].split('-')[1])
        dayv = np.append(dayv,day)
        
        hr = np.int(time[i].split(':')[0])
        if tod[i] == 'PM' and hr < 12:
            hr += 12
        if tod[i] == 'AM' and hr == 12:
            hr -= 12

        mn = np.int(time[i].split(':')[1])

        time24v = np.append(time24v,hr+mn/60.0)

        d = datetime.datetime(yr,month,day,hr,mn,seconds,milliseconds)
        dt = np.append(dt,d)
        doyv = np.append(doyv,d.timetuple().tm_yday+np.float(hr)/24.0)

        if windhi[i] != 0:
            wi, = np.where(winddir[i] == compass)
            winddir_deg = np.append(winddir_deg,direction[wi])
        else:
            winddir_deg = np.append(winddir_deg,np.nan)

    tout = []
    for i in range(len(time)):
        tout = np.append(tout,time[i]+' '+tod[i])


    dictionary = {'date': date, 'time': tout, 'heat': clean_float(data[:,3]),
                  'temp': clean_float(data[:,4]),'wchill': clean_float(data[:,5]),
                  'hitemp': clean_float(data[:,6]),'lotemp': clean_float(data[:,7]),
                  'humidity': clean_float(data[:,8]), 'dew': clean_float(data[:,9]),
                  'wind':  clean_float(data[:,10]),'windhi': windhi,'winddir': winddir,
                  'rain': clean_float(data[:,13]),'pressure': clean_float(data[:,14]),
                  'temp_in': clean_float(data[:,15]),'humidity_in': clean_float(data[:,16]),
                  'archive':clean_float(data[:,17]), 'winddir_deg': winddir_deg,
                  'year':yearv, 'month':monthv, 'day': dayv, 'doy':doyv, 'time24': time24v,
                  'datetime':dt}
    
    return dictionary


def temp_hi_lo(year=2012):
    """
    Plot the annual high and low temperatures
    """

    from matplotlib.dates import MonthLocator, DateFormatter
    from matplotlib.ticker import NullFormatter


    d = get_data(year=year)

    temp = d["temp"]
    dv = d["datetime"]
    dh = d["time24"]

    date = []
    for i in range(len(dv)):
        date = np.append(date,dv[i].toordinal() + dh[i]/24.0)

    doyv = d["doy"]
    doyu = np.unique(np.floor(doyv)).astype('int')
    
    # create more vectors of interest
    thiv = [] ; tlov = [] ; dhiv = [] ; dlov = []

    for i in range(len(doyu)):
        inds, = np.where(np.floor(doyv) == doyu[i])
        hinds, = np.where((np.floor(doyv) == doyu[i]) & (doyv - np.floor(doyv) > 0.45 ) & (doyv - np.floor(doyv) < 0.65))
        if len(hinds) > 10:
            arg = np.argmax(temp[inds])
            thiv = np.append(thiv,np.max(temp[inds[arg]]))
            dhiv = np.append(dhiv,date[inds[arg]])
        linds, = np.where((np.floor(doyv) == doyu[i]) & (doyv - np.floor(doyv) > 0.1 ) & (doyv - np.floor(doyv) < 0.3))
        if len(hinds) > 10:
            arg = np.argmin(temp[inds])
            tlov = np.append(tlov,np.min(temp[inds[arg]]))
            dlov = np.append(dlov,date[inds[arg]])

            
    plot_params()
    plt.ion()
    plt.figure(1,figsize=(11,8.5))
    plt.clf()
    ax = plt.subplot(111)
    ax.plot_date(dhiv,thiv,'-r',linewidth=lw,label='Daily Highs')
    ax.plot_date(dlov,tlov,'-b',linewidth=lw,label='Daily Lows')
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
    
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(DateFormatter('%b'))

    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')
        
    ax.set_xlabel(r'Temperature ($^\circ$F)',fontsize=fs)
    plt.legend(loc='best',fontsize=fs-2,frameon=False)
    
    imid = len(dhiv)/2
    ax.set_xlabel(str(year),fontsize=fs)

    ax.set_ylim(20,120)

    plt.savefig('His_Los_'+str(year)+'.png',dpi=300)
    mpl.rcdefaults()
   
    return

def wind_speed_direction(year=2013,peak=False):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE
    import robust as rb

    min1 = 0.0
    max1 = 360.0
    min2 = 0
    sigfac = 3
    sigsamp = 5

    d = get_data(year=year)
    if peak:
        wind = d['windhi']
        tag = 'peak'
        word = 'Peak '
    else:
        wind = d["wind"]
        tag = 'ave'
        word = 'Average '
        
    wdir = d["winddir_deg"]
    
    wind_rand = wind + np.random.normal(0,0.5,len(wind))
    wdir_rand = wdir + np.random.normal(0,12,len(wdir))
    bad = np.isnan(wdir_rand)
    wdir_rand[bad] = np.random.uniform(0,360,np.sum(bad))

    dist1 = wdir_rand
    dist2 = wind_rand
    
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    datamin1 = np.min(dist1)
    datamax1 = np.max(dist1)
    

    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    datamin2 = np.min(dist2)
    datamax2 = np.max(dist2)
    max2 = min(med2 + sigfac*sig2,datamax2)
    
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])
    
    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)
    
    aspect = (max1-min1)/(max2-min2) * 8.5/11.0

    plot_params()
    plt.ion()
    plt.figure(2,figsize=(11,8.5))
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(np.rot90(Z), cmap=plt.cm.CMRmap_r,aspect=aspect, \
              extent=[min1, max1, min2, max2],origin='upper')
    ax.yaxis.labelpad = 12
    ax.set_xlabel('Wind Direction (degrees)',fontsize=fs)
    ax.set_ylabel(word+'Wind Speed (mph)',fontsize=fs)
    plt.title('Wind Patterns at Thacher Observatory in '+str(year),fontsize=fs)
    
    plt.savefig('Wind'+tag+'_Speed_Direction_'+str(year)+'.png',dpi=300)
    mpl.rcdefaults()

    return


def wind_speed_pressure(year=2013,peak=False):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE
    import robust as rb

    min2 = 0
    sigfac = 3
    sigsamp = 5

    d = get_data(year=year)
    if peak:
        wind = d['windhi']
        tag = 'peak'
        word = 'Peak '
    else:
        wind = d["wind"]
        tag = 'ave'
        word = 'Average '

    wind_rand = wind + np.random.normal(0,0.5,len(wind))
    press = d["pressure"]
    
    dist1 = press
    dist2 = wind_rand
    
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    datamin1 = np.min(dist1)
    datamax1 = np.max(dist1)
    min1 = np.min(dist1)
    max1 = np.max(dist1)


    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    datamin2 = np.min(dist2)
    datamax2 = np.max(dist2)
    max2 = min(med2 + sigfac*sig2,datamax2)
    
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])
    
    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)
    
    aspect = (max1-min1)/(max2-min2) * 8.5/11.0

    plot_params()
    plt.ion()
    plt.figure(5,figsize=(11,8.5))
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(np.rot90(Z), cmap=plt.cm.CMRmap_r,aspect=aspect, \
              extent=[min1, max1, min2, max2],origin='upper')
    ax.yaxis.labelpad = 12
    ax.set_xlabel('Atmospheric Pressure (in-Hg)',fontsize=fs)
    ax.set_ylabel(word+'Wind Speed (mph)',fontsize=fs)
    plt.title('Wind Speed and Pressure at Thacher Observatory in '+str(year),fontsize=fs)
    
    plt.savefig('Wind'+tag+'_Pressure_'+str(year)+'.png',dpi=300)
    mpl.rcdefaults()

    return



def wind_dir_pressure(year=2013):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE
    import robust as rb

    min2 = 0
    sigfac = 3
    sigsamp = 5

    d = get_data(year=year)
    wdir = d["winddir_deg"]
    
    wdir_rand = wdir + np.random.normal(0,12,len(wdir))
    bad = np.isnan(wdir_rand)
    wdir_rand[bad] = np.random.uniform(0,360,np.sum(bad))
    
    press = d["pressure"]
    
    dist1 = wdir_rand
    dist2 = press
    
    med1 = np.median(dist1)
    sig1 = rb.std(dist1)
    datamin1 = np.min(dist1)
    datamax1 = np.max(dist1)
    min1 = 0.0
    max1 = 360.0


    med2 = np.median(dist2)
    sig2 = rb.std(dist2)
    datamin2 = np.min(dist2)
    datamax2 = np.max(dist2)
    min2 = np.min(dist2)
    max2 = np.max(dist2)
    
    X, Y = np.mgrid[min1:max1:100j, min2:max2:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([dist1, dist2])
    
    kernel = KDE(values,var_type='cc',bw=[sig1/sigsamp,sig2/sigsamp])
    Z = np.reshape(kernel.pdf(positions).T, X.shape)
    
    aspect = (max1-min1)/(max2-min2) * 8.5/11.0

    plot_params()
    plt.ion()
    plt.figure(5,figsize=(11,8.5))
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(np.rot90(Z), cmap=plt.cm.CMRmap_r,aspect=aspect, \
              extent=[min1, max1, min2, max2],origin='upper')
    ax.yaxis.labelpad = 12
    ax.set_ylabel('Atmospheric Pressure (in-Hg)',fontsize=fs)
    ax.set_xlabel('Wind Direction (degrees)',fontsize=fs)
    plt.title('Wind Direction and Pressure at Thacher Observatory in '+str(year),fontsize=fs)
    
    plt.savefig('Wind_Direction_Pressure_'+str(year)+'.png',dpi=300)
    mpl.rcdefaults()

    return


def ave_diurnal_plot(year=2013,months=[1,2]):
#    from matplotlib.dates import MonthLocator, DateFormatter
#    from matplotlib.ticker import NullFormatter


    d = get_data(year=year)

    temp = d["temp"]
    mn = d["month"]

    inds = []
    for month in months:
        inds = np.append(inds, np.where(mn == month))

    inds = inds.astype('int')
    temp = temp[inds]
    mn = mn[inds]
    dh = d["time24"][inds]
    dv = d["datetime"][inds]
    
    times  = np.sort(np.unique(dh))
    
    temps = np.zeros(len(times))
    this =  np.zeros(len(times))
    tlos =  np.zeros(len(times))
    for i in range(len(times)):
        inds, = np.where(dh == times[i])
        params = distparams(temp[inds])
        temps[i] = np.mean(temp[inds])
        this[i] = params[4]
        tlos[i] = params[3]
        

            
    plot_params()
    plt.ion()
    plt.figure(4,figsize=(11,8.5))
    plt.clf()
    plt.plot(times,temps,'-k',linewidth=lw)
    plt.plot(times,this,'--k',linewidth=lw)
    plt.plot(times,tlos,'--k',linewidth=lw)
    plt.xlim(0,23.75)
    mpl.rcdefaults()
    

    pass

    


