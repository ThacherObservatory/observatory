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
    data = get_data(year=2015,month=3,day=6)


    To do:
    ------
    Make data reading more robust

    """

    # Set up path and filename
    file = str(year)+'-'+str(month).zfill(2)+'-'+str(day).zfill(2)+'.txt'
    filename = path+file

    test = glob.glob(filename)
    if len(test) == 0:
        print 'File '+file+' not found in directory'
        print '     '+path
        return []
    
    # Read data line by line
    f = open(filename)
    content = [x.strip('\n').strip('\r') for x in f.readlines()]
    date = []; time = []; hval = [] ; Dval = [] ;Eval = [] ;Cval = [] ;Wval = []
    Rval = [] ;oneval = [] ;cval = []; SKY = []; AMB = []; WIND = []; wval = []
    rval = []; HUM = [] ;DEW = [] ;CASE = [] ; HEA = []; BLKT = []; Hval = []; PWR = []
    WNDTD = []; WDROP = []; WAVG = []; WDRY = []; RHT = []; AHT = []; ASKY = []
    ACSE = []; APSV = []; ABLK = []; AWND = []; AVNE = [] ; DKMPH = [] ; VNE = []
    RWOSC = []; D = []; ADAY = []; PH = [] ; CN = []; T = []; S = []
    f.close()
    
    counter = 0
    for c in content:
        if c[26:28] == '~D':
            counter += 1
            date.append(c[0:10])
            time.append(c[11:22])
            hval.append(c[23])
            Dval.append(c[26:28])
            Eval.append(c[29])
            Cval.append(c[31])
            Wval.append(c[33])
            Rval.append(c[35])
            oneval.append(c[37])
            cval.append(c[39])
            SKY = np.append(SKY,np.float(c[41:47]))
            AMB = np.append(AMB,np.float(c[48:53]))
            WIND = np.append(WIND,np.float(c[54:59]))
            wval.append(c[60])
            rval.append(c[62])
            HUM = np.append(HUM,np.float(c[64:67]))
            DEW = np.append(DEW,np.float(c[68:73]))
            CASE = np.append(CASE,np.float(c[74:79]))
            HEA = np.append(HEA,np.float(c[80:83]))
            BLKT = np.append(BLKT,np.float(c[84:89]))
            Hval.append(c[90])
            PWR = np.append(PWR,np.float(c[92:96]))
            WNDTD = np.append(WNDTD,np.float(c[97:102]))
            if counter > 1:
                ADAY = np.append(ADAY,np.float(c[176:181]))
            else:
                ADAY = np.append(ADAY,np.nan)
        else:
            counter = 0

    pdb.set_trace()

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
