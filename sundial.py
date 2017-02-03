import ephem
import numpy as np
from datetime import datetime
from pytz import timezone
import matplotlib.pyplot as plt

# Longitude and Latitude of Poco Farm
lon = '-119:16:32.11'
lat = '34:26:47.17'
loc = ephem.Observer()
loc.long = ephem.degrees(lon)
loc.lat = ephem.degrees(lat)

# Elevation in meters (740 ft)
loc.elevation = 225.55 

plt.ion()
plt.figure(1)
plt.clf()
plt.grid()


# Height of gnomon in meters
gheight = 0.25

# Date array
ystart = '2015'
mstart = '1'
dstart = '1'
datestr = str(ystart)+'/'+str(mstart)+'/'+str(dstart)
stdate = ephem.Date(datestr) - 0.5
tzone = -8
tzone_d = tzone*ephem.hour

inc = ephem.minute
inc_in_day = np.int(np.round(1.0/inc))

# Starting time
loc.date = stdate

# Loop over year
for j in range(0,12):
    date = stdate + np.float(j)*30

    # Loop over day
    loctime = [] ; xshadow = [] ; yshadow = []
    for i in range(0,inc_in_day+1):
        day = date + np.float(i)*inc
        loc.date = day
        sun = ephem.Sun(loc)
        if np.degrees(sun.alt) < 10.0:
            xshadow = np.append(xshadow,np.nan)
            yshadow = np.append(yshadow,np.nan)
            loctime = np.append(loctime,str(loc.date))
        else:
            l = gheight/np.tan(sun.alt)
            theta = 3*np.pi/2.0 - sun.az
            if theta < 0:
                theta += 2.0*np.pi
            xshadow = np.append(xshadow,l*np.cos(theta))
            yshadow = np.append(yshadow,l*np.sin(theta))
        

        plt.plot(xshadow,yshadow)


# Times to plot on shadow traces
uts = [' 16:00:00',' 18:00:00',' 20:00:00',' 22:00:00',' 24:00:00']

# Loop over year
size = len(uts)
xtime = np.empty((size,366))
ytime = np.empty((size,366))
for j in range(0,366):
    date = stdate + np.float(j)
    dstem = ('%s' % ephem.Date(date)).split(' ')[0]
    for k in range(0,size):
        d =  ephem.Date(('%s' % ephem.Date(date)).split(' ')[0]+uts[k])
        loc.date = d
        sun = ephem.Sun(loc)
        l = gheight/np.tan(sun.alt)
        theta = 3*np.pi/2.0 - sun.az
        xtime[k,j] = l*np.cos(theta)
        ytime[k,j] = l*np.sin(theta)

# Plot analemmas
for i in range(0,size):
    plt.plot(xtime[i,:],ytime[i,:],'k-')
    
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.plot([0],[0],'ko',markersize=10)
plt.xlabel('East-West Distance (m)')
plt.ylabel('North-South Distance (m)')
plt.title('Shadow Traces for Poco Farm')
plt.axes().set_aspect('equal')
plt.savefig('PocoFarmShadows.png',dpi=300)
