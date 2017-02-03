import numpy as np
import matplotlib.pyplot as plt
from quick_image import display_image
import thacher_phot as tp


files = glob.glob('Mantis*[0-9]'+band+'.fit')
    zsz = len(files)
    image0,header0 = readimage(files[0])
    ysz,xsz = np.shape(image)
    
    stack = np.zeros((xsz,ysz,zsz))
    for i in range(zsz):
        image,header = readimage(files[i])
        
        stack[:,:,i] = image
        
    final = np.median(stack,axis=2)

    if band == 'V':
        tag = 'Blue'
        
    if band == 'R':
        tag = 'Green'
    
    if band == 'ip':
        tag = 'Red'
        
    fits.writeto(tag+'.fit', final, header)
    
 
