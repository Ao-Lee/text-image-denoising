import numpy as np
from skimage.io import imread
from tqdm import tqdm
import cv2
from os.path import join, isdir
from os import makedirs, listdir

import cfg
model_name = 'HighPassFilter'



def PredictOnImg(img_path):
    img_origin = imread(img_path)                   # (H, W)
    img_origin = img_origin/255
    # Fourier transform the input image
    imfft = np.fft.fft2(img_origin)
    	
    # Apply a high pass filter to the image. 
    # Note that since we're discarding the k=0 point, we'll have to add something back in later to match the correct white value for
    # the target images
    
    for i in range(imfft.shape[0]):
        # Fourier transformed coordinates in the array start at kx=0 and increase to pi, then flip around to -pi and increase towards 0
        kx = i/float(imfft.shape[0])
        if kx>0.5: kx = kx-1
        
        for j in range(imfft.shape[1]):
            ky = j/float(imfft.shape[1])
            if ky>0.5: ky = ky-1
            
            # Get rid of all the low frequency stuff - in this case, features whose wavelength is larger than about 20 pixels
            if (kx*kx + ky*ky < 0.015*0.015): imfft[i,j] = 0

    # Transform back
    img_prediction = 1.0*((np.fft.ifft2(imfft)).real)+0.9
    img_prediction = np.minimum(img_prediction, 1.0)
    img_prediction = np.maximum(img_prediction, 0.0)
    
    img_prediction = (img_prediction*255).astype(np.uint8)
    return img_prediction
    
def PredictOnFolder(src, dst):
    print('read date from {} and make predictions on {}'.format(src, dst))
    if not isdir(dst): makedirs(dst)
    
    for name in tqdm(listdir(src)):
        name_src = join(src, name)
        name_dst = join(dst, name)
        result = PredictOnImg(name_src)
        cv2.imwrite(name_dst, result)
        
def Run():
    print(model_name + ': ' + 'predict training data')
    PredictOnFolder(src=cfg.path_train, dst= join(model_name, 'train'))
        
    print(model_name + ': ' + 'predict testing data')
    PredictOnFolder(src=cfg.path_test, dst=join(model_name, 'test'))

if __name__=='__main__':
    Run()
    