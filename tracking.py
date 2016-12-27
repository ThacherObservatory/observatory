import numpy as np
import matplotlib.pyplot as plt
import thacherphot as tp
import matplotlib as mpl
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
import matplotlib.gridspec as gridspec
import datetime
import jdutil
from astropysics.coords import AngularCoordinate as angcor

path = '/Users/jonswift/Astronomy/ThacherObservatory/data/22Dec2016/'


#path = '/Users/ONeill/astronomy/Data/04Dec2016/'

files,fct = tp.get_files(dir=path, prefix='WASP-50b',suffix='solved.fits')
bfiles,bct = tp.get_files(dir=path,prefix='WASP-50b',suffix='bias.fit')
master_bias = tp.master_bias(bfiles)

data = tp.batch_total_flux(files,object='WASP-50b',mag=11.6,SpT='g8v',bias=master_bias,
                           nearest=True,camera='iKON-L',filter='V',gain=3.0,network='swift')


#tau = data['tau']
#secz = data['secz']
#plt.clf()
#plt.ion()
#plt.figure()
#plt.plot(secz,tau,'o')
#plt.ylabel('tau')
#plt.xlabel('secz')
#plt.xlim(1.85,1.95)
#plt.ylim(0.64,0.67)


# make datetime vector from JD

dtime = [jdutil.jd_to_datetime(jd) for jd in data['jd']]

#for i in range(len(data['jd'])):
#    dt = np.append(dtime,jdutil.jd_to_datetime(data['jd'][i]))
dates = mpl.dates.date2num(dtime)
    
xpos = data['xpos']
ypos = data['ypos']
mx = np.median(xpos)
my = np.median(ypos)

plt.ion()
fig = plt.figure(1,figsize=(8.5,11))
gs = gridspec.GridSpec(5, 1,wspace=0)

ax1 = plt.subplot(gs[0, 0])    
ax1.plot(dates,xpos-mx,'o')
ax1.set_xticklabels(())
ax1.set_ylabel(r'$\Delta$X (pixels)')
ax1.set_xlim(np.min(dates),np.max(dates))
#ax1.xaxis.set_major_locator(DayLocator())
#ax1.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 6)))
#ax1.xaxis.set_major_formatter(DateFormatter('%H-%M-%S'))
#ax1.fmt_xdata = DateFormatter('%H:%M:%S')
#fig.autofmt_xdate()


ax2 = plt.subplot(gs[1, 0])    
ax2.plot(dates,ypos-my,'o')
ax2.set_xticklabels(())
ax2.set_ylabel(r'$\Delta$Y (pixels)')
ax2.set_xlim(np.min(dates),np.max(dates))
#ax2.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 6)))
#ax2.xaxis.set_major_formatter(DateFormatter('%H-%M-%S'))
#ax2.fmt_xdata = DateFormatter('%H:%M:%S')
#fig.autofmt_xdate()

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
ax5.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 6)))
#ax5.xaxis.set_minor_locator(HourLocator())
ax5.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
ax5.fmt_xdata = DateFormatter('%H:%M:%S')
fig.autofmt_xdate()
plt.xlabel('Universal Time (23 December 2016)')

plt.savefig('22Dec2016_tracking.png',dpi=300)
