import matplotlib.pyplot as plt
import numpy as np



def test_image(center=None,snr=None,sigma=20,floor=500,size=200,seeing=3,
               flux=100):

    flux *= sigma

    image = np.random.normal(floor,sigma,(size,size))

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    star = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / seeing**2)
    
    npix = max(np.pi*(seeing)**2,1)
    
    noise = np.sqrt(npix)*sigma
    
    if snr:
        bright = noise*snr
    elif flux:
        bright = flux
        snr = flux/noise
        print 'Total SNR = %.0f' % snr
        
    star  = star/np.sum(star) * bright
    
    image += star
    
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.imshow(image,cmap='gray')
    
    slice = image[y0,:]
    plt.figure(2)
    plt.clf()
    plt.plot(slice)
    
