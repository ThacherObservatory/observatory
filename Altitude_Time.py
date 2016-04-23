# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 19:43:17 2016

@author: sara
"""
import matplotlib.pyplot as pl
import datetime
import numpy as np
import ephem
from scipy import interpolate

# Hi


def riseset(year, month, day, position):
    
    thob = ephem.Observer()
    thob.long = ephem.degrees("-119.1773417")
    thob.lat = ephem.degrees("34.467028")
    thob.elevation = 504.4 
    #thob.date = "2017/8/21" 
    #sun = ephem.Sun(thob)
    
    a = []  
    t = range(0,1440) #np.array([datetime.datetime(2016,4,6,i,m,0) for i in range(24) for m in range(60)])
        
    for d in [datetime.datetime(year,month,day,i,m,0) for i in range(24) for m in range(60)]:
        thob.date = d
        sun = ephem.Sun(thob) #put coordinate of star here
        a.append(sun.alt + np.radians(0.25))
        
    a = np.array(a)
    t = np.array(t)
    
    pl.title('The Altitude of the Sun')
    pl.ylabel('Time in minutes',)
    pl.xlabel('Altitude')
    
    
    pl.plot(a, t, 'ro')#, xnew, ynew, '-')
    pl.axvline(x=0, color = 'b',linestyle = 'dashdot')
    pl.axvline(x=-0.20943951,color = 'b',linestyle = 'dashdot')
    pl.axvline(x=-0.314159265, color = 'b',linestyle = 'dashdot')
    
    
    inds, = np.where((a>-0.4) & (a<0.1))
    
    pl.plot(a[inds],t[inds],'ko')
    pl.show()
    
    m = np.diff(a[inds])
    rinds = np.where(m > 0)
    risea = a[inds][rinds]
    riset = t[inds][rinds]
    risefunc = interpolate.interp1d(risea, riset)
    risetime = risefunc(np.radians(position))
    risehour = risetime//60
    riseminute = risetime % 60
    risesecond = int((risetime - int(risetime))*60)
    sunrise = '%d:%d:%d' % (risehour, riseminute, risesecond)
    
    sinds = np.where(m < 0)
    seta = a[inds][sinds]
    sett = t[inds][sinds]
    setfunc = interpolate.interp1d(seta, sett)
    settime = setfunc(np.radians(position))
    sethour = settime//60
    setminute = settime % 60
    setsecond = int((settime - int(settime))*60)
    sunset = '%d:%d:%d' % (sethour, setminute, setsecond)
    
    return sunrise, sunset 
    
print riseset(2016,4,14,-12.0)