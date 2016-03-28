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


for t in timeloop(datetime(2007, 3, 30, 00, 00), 
                          datetime(2007, 3, 30, 00, 00), 
                          delta=timedelta(hours=1)):
    thob.date = t
    sun = ephem.Sun(thob)
 #   x = sun.alt 
  #  y = sun.az
    
 #   pl.plot(x,y,'ro')
    print"%s %f %f %f %f" % (thob.date, sun.alt, sun.az, sun.ra, sun.dec)
