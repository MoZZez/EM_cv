from os.path import join
from glob import glob
import cv2

import numpy as np

from matplotlib import pyplot as plt


def log_scaling(img, ret_int=True):
    '''
    Scaling given np array of numeric type from 0 to 255
    with logarithmic transform.
    So whatever values were there before, after we may treat it as usual image.
    
    Args:
         img - np.array of numeric type.
         ret_int - bool whether to convert to uin8 output before returning it.
                   Default is True.
                   
    returns: scaled(and converted to np.uin8) np.array.
    '''
    
    abs_img = np.absolute(img)
    max_value = np.amax(abs_img)
    
    scale_coef = 255. / (np.log(max_value + 1))
    
    scaled_img = scale_coef * np.log(1 + abs_img)
    if ret_int == True:
        scaled_img = scaled_img.astype(np.uint8)
        
    return scaled_img

def linear_scaling(img, ret_int=True, max_bright=255.):
    '''
    Scaling given np array of numeric type from 0 to max_bright(255).
    with linear transform.
    So whatever values were there before, after we may treat it as usual image.
    
    Args:
         img - np.array of numeric type.
         ret_int - bool whether to convert to uin8 output before returning it.
                   Default is True.
         max_bright - numeric scalar, which set upper bound of scaling.
                      Default is 255.
    
    returns: scaled(and converted to np.uin8) np.array.
    '''
    max_value = np.amax(img)
    min_value = np.amin(img)
    
    scale_coef = max_bright / (max_value - min_value)
    bias = -scale_coef * min_value
    
    scaled_img = scale_coef * img + bias
    if ret_int == True:
        #scaled_img[scaled_img > 255] = 255
        scaled_img = scaled_img.astype(np.uint8)
    
    return scaled_img

def img_summary(img):
    '''
    Debuging and research func.
    Prints most basic info about given numpy array(image)
    
    Args:
         img - numeric numpy array.
    
    returns: None.
    '''
    print(img.shape,img.dtype)
    print('Max: ', np.amax(img))
    print('Min: ', np.amin(img))
    print('Mean: ',np.mean(img))
    
def fourier_filter(img,win_size=(60,60)):
    '''
    Performing noise filtration of a given grayscale image(shape == (H,W))
    via nullifying high-freq components of image FFT and then recovering image
    from corrected FFT.
    
    Args:
         img - 2D numeric numpy array.
         win_size - 2-tuple of ints size of window
                    set to the senter of img.Inside frequencies will be preserved.
                    Outside nullified.
    returns 2D numeric numpy array of filtered image.
    '''
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    
    mask = np.zeros((rows,cols,2),np.uint8)
    
    mask[crow-int(win_size[0]/2):crow+int(win_size[0]/2), ccol-int(win_size[1]/2):ccol+int(win_size[1]/2)] = 1
    
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    
    return img_back