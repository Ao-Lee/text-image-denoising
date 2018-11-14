import numpy as np
from scipy import signal, ndimage
from skimage.io import imread
from tqdm import tqdm
import cv2
from os.path import join, isdir
from os import makedirs, listdir


import cfg
from metrics import mse
model_name = 'BackGround'

######################## Background Noise removal #########################
#Let's break image into back ground and foreground 
#At the same time, we want to keep the background look similar to original 
#book paper.

#Also lets remove small chunks of splattered ink by closing the gaps
#######################################################################


def PredictOnImg(img_path):
    img_origin = imread(img_path)                   # (H, W)
    img_origin = img_origin/255
    
    # estimate 'background' color by a median filter
    background = signal.medfilt2d(img_origin, 11)
    #save('background.png', background)

    # compute 'foreground' mask as anything that is significantly darker than
    # the background
    foreground = img_origin < background - 0.1    
    #save('foreground_mask.png', foreground)
    back = np.average(background);
    
    # Lets remove some splattered ink
    mod = ndimage.filters.median_filter(foreground,2);
    mod = ndimage.grey_closing(mod, size=(2,2));
       
    # either return forground or average of background
       
    img_prediction = np.where(mod, img_origin, back)  ## 1 is pure white 
    
    img_prediction = (img_prediction*255).astype(np.uint8)
    background = (background*255).astype(np.uint8)
    foreground = (foreground*255).astype(np.uint8)
    
    return img_prediction, background, foreground
    
    
def PredictOnFolder(src, dst):
    print('read date from {} and make predictions on {}'.format(src, dst))
    if not isdir(dst): makedirs(dst)
    
    for name in tqdm(listdir(src)):
        name_src = join(src, name)
        result, background, foreground = PredictOnImg(name_src)
        cv2.imwrite(join(dst, name), result)
        #cv2.imwrite(join(dst, 'back_' + name), background)
        #cv2.imwrite(join(dst, 'fore_' + name), foreground)
        
def Eval():
    err = 0
    for name in cfg.names_val:
  
        img_predicton = imread(join(model_name, 'train', name))
        img_predicton = img_predicton/255
        
        img_label = imread(join(cfg.path_label, name))
        img_label = img_label/255
        
        err += mse(prediction=img_predicton, label=img_label)
    return err/len(cfg.names_val)
        
        
def Run():
    print(model_name + ': ' + 'predict training data')
    PredictOnFolder(src=cfg.path_train, dst= join(model_name, 'train'))
        
    print(model_name + ': ' + 'predict testing data')
    PredictOnFolder(src=cfg.path_test, dst=join(model_name, 'test'))
    
    err = Eval()
    print('mse error is {}'.format(err))
    return err
    
if __name__=='__main__':
    err = Run()

    