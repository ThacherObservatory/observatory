#from astropy.time import Time
#from astropy.coordinates import SkyCoord, EarthLocation, AltAz
#from astropy import units as u
#import sys
import pyfits as pf
import pytz,datetime,ephem
from astropy import wcs
from astropy.io.fits import open
import robust as rb
from djs_photfrac_mb import *
from scipy.ndimage import gaussian_filter 

do_smooth = False

# Specifics of the Thacher Observatory
lat = 34.467028
lon = -119.1773417
deczen = lat

# Create pyephem observatory object 
thob = ephem.Observer()
thob.long = ephem.degrees("-119.1773417")
thob.lat = ephem.degrees("34.467028")
thob.elevation = 504.4 

# Data directory
dir = '/Users/jonswift/Dropbox (Thacher)/Observatory/AllSkyCam/Data/13April2015/'
# Image with astrometric solution from astrometry.net
file = 'Image_15_crop4_astrom.fits'

rate = 0.1500
darkrate = 0.0157 # from D. McKenna
mv = -2.5*np.log10(rate - darkrate)+18.865 # from D. McKenna

# Get image and header
image, header = pf.getdata(dir+file, 0, header=True)

# Get image info
date = header["DATE-OBS"]
# From observer log (header time is not right)
time = '00:14:20'
local = pytz.timezone ("America/Los_Angeles")
naive = datetime.datetime.strptime (date+" "+time, "%Y-%m-%d %H:%M:%S")
local_dt = local.localize(naive, is_dst=None)
utc_dt = local_dt.astimezone (pytz.utc)

# All dates and times in pyephem are UTC
thob.date = utc_dt
ra = thob.sidereal_time()
dec = ephem.degrees(np.radians(deczen))


image = np.array(image)
ysz, xsz = np.shape(image)

# Get image astrometry    
hdulist = open(dir+file)
w = wcs.WCS(hdulist['PRIMARY'].header)
radeg  = np.degrees(ra)
decdeg = np.degrees(dec)
xpix,ypix = w.wcs_world2pix(radeg,decdeg,1) # Pixel coordinates of (RA, DEC)



# Get indices of 5 degree field
pixsz  = np.sqrt(header['CD1_1']**2 + header['CD1_2']**2)
radpix = 2.5/pixsz 
ap = djs_photfrac(ypix,xpix,radpix,xdimen=xsz,ydimen=ysz)

# Smooth image
sigma = 2*radpix/2.355
if do_smooth:
    smooth = gaussian_filter(image,sigma)
else:
    smooth = image
    
# Image characteristics and plot
sig = rb.std(smooth)
med = np.median(smooth)
vmin = med - 3*sig
vmax = med + 5*sig
plt.figure(1)
plt.clf()
plt.imshow(smooth,vmin=vmin,vmax=vmax,cmap='gist_heat',interpolation='nearest', \
           origin='lower')
plt.scatter(xpix,ypix,marker='+',s=100,facecolor='none',edgecolor='yellow', \
            linewidth=1.5)
plt.xlim(0,xsz)
plt.ylim(0,ysz)
plt.axis('off')
plt.title('Field Center')
plt.savefig('Center.png',dpi=300)

image2 = smooth*0
image2[ap['xpixnum'],ap['ypixnum']] = smooth[ap['xpixnum'],ap['ypixnum']]

plt.figure(2)
plt.clf()
plt.imshow(image2,vmin=vmin,vmax=vmax,cmap='gist_heat',interpolation='nearest', \
           origin='lower')
plt.scatter(xpix,ypix,marker='+',s=100,facecolor='none',edgecolor='yellow', \
            linewidth=1.5)
plt.xlim(0,xsz)
plt.ylim(0,ysz)
plt.axis('off')
plt.title('Photometer Field of View')
plt.savefig('FOV.png',dpi=300)

# Get median value in apeture
#ftot = np.sum(image2[ap['xpixnum'],ap['ypixnum']])
#npix = np.sum(ap['fracs'])
#avgval = ftot/npix
avgval = np.median(image2[ap['xpixnum'],ap['ypixnum']])


#avgval = np.median(image[ap['xpixnum'],ap['ypixnum']])
#avgval = np.min(image[ap['xpixnum'],ap['ypixnum']])

logavg = -2.5*np.log10(avgval)
calval = 21.029 - logavg
fullfile = 'img00015_ds.fit'

# Get image and header
fullim, fullh = pf.getdata(dir+fullfile, 0, header=True)

# Smooth full image
if do_smooth:
    fsmooth = gaussian_filter(fullim,sigma)
else:
    fsmooth = fullim
    
# Select out horizon
#if do_smooth:
inds = np.where(fullim <= 500)
fullim[inds] = 1e-5
fsmooth[inds] = fullim[inds]

finalim = fsmooth
logim = -2.5*np.log10(finalim)
calim = logim + calval

ysz, xsz = np.shape(calim)
calap = djs_photfrac(ysz/2,xsz/2+40,ysz/2.0,xdimen=ysz,ydimen=xsz)
newcal = calim*np.inf
newcal[calap['xpixnum'],calap['ypixnum']] = calim[calap['xpixnum'],calap['ypixnum']]
cmap='CMRmap_r'
vmin = 19.0
vmax = 22.0
plt.figure(3)
plt.clf()
plt.imshow(newcal,vmin=vmin,vmax=vmax,cmap=cmap,interpolation='nearest', \
           origin='upper')
plt.axis('off')
plt.colorbar(shrink=0.65,aspect=10,ticks=[19,19.5,20,20.5,21,21.5,22], \
    orientation='horizontal',pad=0.075)
plt.annotate('N',[0.53,0.91],horizontalalignment='center',xycoords='figure fraction',\
    fontsize=18)
plt.annotate('S',[0.53,0.24],horizontalalignment='center',xycoords='figure fraction',\
    fontsize=18)
plt.annotate('E',[0.28,0.58],horizontalalignment='center',xycoords='figure fraction',\
    fontsize=18)
plt.annotate('W',[0.785,0.58],horizontalalignment='center',xycoords='figure fraction',\
    fontsize=18)
plt.annotate(r'mags/arcsec$^2$',[0.5,0.07],horizontalalignment='center', \
             xycoords='figure fraction',fontsize=12)
plt.savefig('SkyBrightness.png',dpi=1200)