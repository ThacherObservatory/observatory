"""
I need some help with the path manager thing
It's not recognizing weather and seeing
"""

import weather as w
import seeing as s
import numpy as np
import matplotlib.pyplot as plt


wpath = '/Users/nickedwards/python/data/weather/'
spath = '/Users/nickedwards/python/data/seeing/'
weather_data = w.get_data(dpath=wpath)
seeing_data = s.get_data(path=spath)


# Following code was Frankensteined from weather.py to be used as
# as template for a 2D correlation plot
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
