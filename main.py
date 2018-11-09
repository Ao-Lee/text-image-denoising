from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


import cfg
import train
import predict
import viz
'''
https://www.kaggle.com/c/denoising-dirty-documents
'''

    
if __name__=='__main__':
    # model = train.Train()
    split = int(len(cfg.names)*0.9)
    tests = cfg.names[split:]
    result_path = 'result'
    
    
    for name in tests:
        img_origin = imread(join(cfg.path_train, name))
        img_gt = imread(join(cfg.path_label, name))
        img_prediction = predict.GetPrediction(model, img_origin)
        
        img_origin = viz.Gray2Rgb(img_origin)
        img_gt = viz.Gray2Rgb(img_gt)
        img_prediction = viz.Gray2Rgb(img_prediction)
        
        imgs = [img_origin, img_gt, img_prediction]
        imgs = viz.MergeImage(imgs, color=(40,0,40))
        viz.ShowImg(imgs, save_path=join(result_path, name))
        
