import numpy as np
import matplotlib.pyplot as plt
import thacherphot as tp
import matplotlib as mpl
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
import matplotlib.gridspec as gridspec
import datetime

path = '/Users/jonswift/Astronomy/ThacherObservatory/data/21Nov2016/'

files,fct = tp.get_files(dir=path, prefix='SA20-297',suffix='.fits')


    
data = tp.batch_total_flux(files,object='SA 20-297',mag=8.907,SpT='b9v',
                           brightest=False,camera='U16m',filter='G')

# make datetime vector from JD

dtime = [jdutil.jd_to_datetime(jd) for jd in data['jd']]

#for i in range(len(data['jd'])):
#    dt = np.append(dt,jdutil.jd_to_datetime(data['jd'][i]))
dates = mpl.dates.date2num(dtime)
    
mx = np.median(xpos)
my = np.median(ypos)
xpos = data['xpos']
ypos = data['ypos']

plt.ion()
plt.figure(1,figsize=(8.5,11))
gs = gridspec.GridSpec(5, 1,wspace=0)

fig,ax = plt.subplots()
ax1 = plt.subplot(gs[0, 0])    
ax1.plot(dates,xpos-mx,'o')
ax1.set_xticklabels(())
ax1.set_ylabel(r'$\Delta$X (pixels)')
ax1.set_xlim(np.min(dates),np.max(dates))
#ax1.xaxis.set_major_locator(DayLocator())
#ax1.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 6)))
#ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
#ax1.fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')
#fig.autofmt_xdate()


ax2 = plt.subplot(gs[1, 0])    
ax2.plot(dates,ypos-my,'o')
ax2.set_xticklabels(())
ax2.set_ylabel(r'$\Delta$Y (pixels)')
ax2.set_xlim(np.min(dates),np.max(dates))

d = np.sqrt((xpos[0]-xpos)**2 + (ypos[0]-ypos)**2)
ax3 = plt.subplot(gs[2, 0])    
ax3.plot(dates,d,'o')
ax3.set_xticklabels(())
ax3.set_ylabel(r'$\Delta$r (pixels)')
ax3.set_xlim(np.min(dates),np.max(dates))

ax4 = plt.subplot(gs[3, 0])    
ax4.plot(dates,data['fwhm'],'o')
ax4.set_xticklabels(())
ax4.set_ylabel(r'FWHM (pixels)')
ax4.set_xlim(np.min(dates),np.max(dates))

ax5 = plt.subplot(gs[4, 0])    
ax5.plot(dates,data['chisq'],'o')
ax5.set_ylabel(r'$\chi^2_r$')
ax5.set_xlim(np.min(dates),np.max(dates))
plt.gcf().autofmt_xdate()
