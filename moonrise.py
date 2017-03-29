import ephem # See http://rhodesmill.org/pyephem/
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt

#################################################################################
# Created by jswift 9/25/2015 in preparation for the lunar eclipse on 9/27/2015 #
# Default inputs created to produce moon positions for 9/27/2015 as observed    #
# from the Thacher Observatory.                                                 #
#                                                                               #
# Simply run the following commands from a python or ipython console:           #
# import moonrise as mr                                                         #
# location = mr.make_location()                                                 #
# times,alt,az = mr.moon_pos_range(location,plot=True)                          #
#                                                                               #
#################################################################################
def moon_radec(year=2017,month=1,day=1,epoch=2000):
    """
    Returns the apparent geocentric Right Ascension and Declination of the 
    Moon for the given year, month and day
    """
    m = ephem.Moon()
    datestr = str(year)+'/'+str(month)+'/'+str(day)
    m.compute(datestr, epoch=str(epoch))

    return str(m.g_ra),str(m.g_dec)



def get_timezone():
    """
    Get timezone in your current location based on your computer time and UTC
    """

    tz = (dt.datetime.utcnow() - dt.datetime.now())

    return np.round((tz.seconds + tz.microseconds/1e6)/3600.0)



def make_location(lat='34:28:01.30',lon='-119:10:38.43',elevation=502.9):
    """ 
    Generate a pyephem location that can be appended to
    """
    
    loc = ephem.Observer()
    loc.long = ephem.degrees(lon)
    loc.lat = ephem.degrees(lat)

    return loc


def moon_pos(location,year=2015,month=9,day=28,hour=0,min=0,sec=0,
             microsec=0,timezone=None,verbose=False):
    """
    Get moon position for a given year, month, day, hour, minute, second
    and microsecond (all integers)
    """
    
    if not timezone:
        timezone = get_timezone()

    indate = dt.datetime(year,month,day,hour,min,sec,microsec)
    date = indate + dt.timedelta(hours=timezone)
    
    location.date = ephem.Date(date)

    # Create moon object for given location 
    moon = ephem.Moon(location)

    if verbose:
        az = np.degrees(moon.az)
        alt = np.degrees(moon.alt)
        print 'Output for '+str(indate)
        print 'Moon altitude = %.2f degrees' % alt
        print 'Moon azimuth = %.2f degrees' % az

    return moon.alt,moon.az


def moon_pos_range(location,dstart=[2015,9,27,18],dstop=[2015,9,28,6],timezone=None,plot=False):
    """
    Get the position of the moon for the input location within a range set by two dates:
    [year, month, day, hour]
    """
    
    start = dt.datetime(dstart[0],dstart[1],dstart[2],dstart[3]) 
    stop  = dt.datetime(dstop[0],dstop[1],dstop[2],dstop[3]) 

    istart = start
    
    if not timezone:
        timezone = get_timezone()
        start += dt.timedelta(hours=timezone)
        stop  += dt.timedelta(hours=timezone)

    # Set increment of one minute
    inc   = ephem.minute
    tfull = (stop-start).seconds/(3600.0*24.0)
    nincs = np.int(np.round(tfull/inc))

    # Create null vectors for quantities of interest
    moonalt = []
    moonaz  = []
    times   = []
    
    for i in range(0,nincs+1):
        day = start + dt.timedelta(days=inc*i)
        iday = istart+ dt.timedelta(days=inc*i)
        location.date = day
        moon = ephem.Moon(location)
        moonalt = np.append(moonalt,np.degrees(moon.alt))
        moonaz  = np.append(moonaz,np.degrees(moon.az))
        times.append(iday)

    if plot:
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.plot(times,moonalt)
        plt.gcf().autofmt_xdate()
        plt.title('Start Date: '+str(dstart[1])+'/'+str(dstart[2])+'/'+str(dstart[0]))
        plt.ylabel('Altitude (degrees)')
        if np.min(moonalt) < 0:
            plt.axhline(y=0,color='red',linestyle='--')
        
    return times,moonalt,moonaz
