import ephem
import matplotlib.pyplot as pl
from datetime import date, time, datetime, timedelta

thob = ephem.Observer()
thob.long = ephem.degrees("-119.1773417")
thob.lat = ephem.degrees("34.467028")
thob.elevation = 504.4 
#thob.date = "2017/8/21 00:00:00" 
#sun = ephem.Sun(thob)

def timeloop(startDate, endDate, delta=timedelta(days=1)):
    currentDate = startDate
    while currentDate < endDate:
        yield currentDate
        currentDate += delta

x = [] #azimuth
y = [] #altitude

m = [] #right ascension
n = [] #declination

for t in timeloop(datetime(2016, 3, 30, 01, 00, 00), 
                          datetime(2016, 3, 31, 01, 00, 00), 
                          delta=timedelta(hours=1)):
    thob.date = t
    sun = ephem.Sun(thob)
    x.append(sun.az) 
    y.append(sun.alt)
    m.append(sun.ra) 
    n.append(sun.dec)
    print"%s %f %f %f %f" % (thob.date, sun.alt, sun.az, sun.ra, sun.dec)
    
horizontal_coordinates = pl.figure(1)
#pl.subplot(222)
pl.title('Path of our SUN in one day')
pl.xlabel('Azimuth')
pl.ylabel('Altitude')
pl.plot(x,y,'yo')
pl.show()

Equatorial_coordinates = pl.figure(2)
pl.title('Path of our SUN in one day')
pl.xlabel('Right Ascension')
pl.ylabel('Declination')
pl.plot(m,n,'ro')
pl.show()
