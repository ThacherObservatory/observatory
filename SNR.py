import matplotlib.pyplot as plt
import numpy as np



def test_image(center=None,snr=None,sigma=20,floor=500,size=200,seeing=3,
               flux=100):

    # Interpret flux in terms of noise level
    flux *= sigma

    # Make an image of given size with specified noise properties.
    # Floor is bias, sigma is noise
    image = np.random.normal(floor,sigma,(size,size))

    # Create arrays that correspond to image pixels
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    # Decide where the star will go. If no input, place the star in the
    # middle of the image
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    # This turns the star into a Gaussian
    star = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / seeing**2)

    # This is approximately how many pixels the star covers (needed
    # to compute the total signal to noise)
    npix = max(np.pi*(seeing)**2,1)

    # Total noise within the star image
    noise = np.sqrt(npix)*sigma

    # Adjust brightness of the star if SNR is specified in the inputs
    if snr:
        bright = noise*snr
    elif flux:
        bright = flux
        snr = flux/noise
        print 'Total SNR = %.0f' % snr
    star  = star/np.sum(star) * bright

    # Add the star into the image
    image += star

    # Make a plot
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.imshow(image,cmap='gray')
    

    # Show a cross section of the star in the image
    slice = image[y0,:]
    plt.figure(2)
    plt.clf()
    plt.plot(slice)

    # Return the image
    return image
    
