"""
To Do:
Match weather data to seeing data by time. 
Start to make graphs and fix plot limits

History:

11/21/15
nedwards: Changed dist variables, added fs and year variable

"""

import weather as w
import seeing as s
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_density import KDEMultivariate as KDE
import robust as rb
import matplotlib as mpl


wpath = '/Users/nickedwards/python/data/weather/'
spath = '/Users/nickedwards/python/data/seeing/'
weather_data = w.get_data(dpath=wpath)
seeing_data = s.get_data(path=spath)

#These are plot limits

#min1 = 0.0
#max1 = 360.0
#min2 = 0
#sigfac = 3
#sigsamp = 5


dist1 = seeing_data['FWHMave']
dist2 = weather_data['humidity']
year = weather_data['year']
fs = 20

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

w.plot_params()
plt.ion()
plt.figure(2,figsize=(11,8.5))
plt.clf()
ax = plt.subplot(111)
ax.imshow(np.rot90(Z), cmap=plt.cm.CMRmap_r,aspect=aspect, \
          extent=[min1, max1, min2, max2],origin='upper')
ax.yaxis.labelpad = 12
ax.set_xlabel(str(dist1),fontsize=fs)
ax.set_ylabel(str(dist2),fontsize=fs)
plt.title('Weather and Seeing Corralation at Thacher Observatory in '+str(year),fontsize=fs)

plt.savefig(str(dist1)+'by'+str(dist2)+str(year)+'.png',dpi=300)
mpl.rcdefaults()
