from os.path import join
from glob import glob

import cv2
import numpy as np

import ice_basic_api as iba

def preprocess_img(img, dataset_format=True):
    '''
    Performs full processing from images to
    binary masks of specific tone rings on imgs
    FFT obtained by compex filtering and adaptive thresholding.
    
    Args:
         img - 2D numpy array(img)
    
    returns: 2D numpy array binary mask of FFT of image.
    '''
    #Custom kernel for convolutional contrast filter.
    kernel = np.array([[-2,-2,-2,-2,-2],
                       [-2,-1,-1,-1,-2],
                       [-2,-1,40,-1,-2],
                       [-2,-1,-1,-1,-2],
                       [-2,-2,-2,-2,-2]])
    #doing magnitude of imgs FFT.
    dft = np.fft.fft2(img)
    dft = np.fft.fftshift(dft)
    magn = np.absolute(dft)
    
    #scaling magnitude
    scaled = iba.log_scaling(magn)
    #reducing noise 
    scaled = cv2.medianBlur(scaled, 7)
    
    #reducing noise with fourier high-freq filtering.
    small = cv2.resize(scaled, (512, 512))
    small = iba.fourier_filter(small)
    #scaling image back to 0,255
    #in this case logscaling does`not work.
    small = iba.linear_scaling(small)
    
    #bluring -> improving contrast
    small = cv2.blur(small, (5, 5))
    small = cv2.filter2D(small, -1, kernel)
    
    #again killing noise
    small = cv2.medianBlur(small, 11)
    
    #finally tresholding and last median to kill rest of noise.
    small = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    small = cv2.medianBlur(small ,5)
    
    if dataset_format == True:
        small = cv2.resize(small,(28,28)).ravel().astype(bool).astype(np.uint8).tolist()
    
    return small



        