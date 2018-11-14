from os.path import join, isdir
from os import makedirs, listdir
from skimage.io import imread
from tqdm import tqdm
import cv2
import numpy as np


import cfg
import train
import data
import predict
import model as m
model_name = 'AutoEncoder'

def GetModelPath():
    return join(cfg.path_models, model_name + '.h5')
    
def Train(model):
    reader_tr_x = data.PathReader(cfg.path_train, cfg.names_tr)
    reader_tr_y = data.PathReader(cfg.path_label, cfg.names_tr)
    gen_tr = data.DataGenerator([reader_tr_x], reader_tr_y).GetGenerator()  
    #if cfg.debug: data.DebugGenerator(gen_tr)

    
    reader_val_x = data.PathReader(cfg.path_train, cfg.names_val)
    reader_val_y = data.PathReader(cfg.path_label, cfg.names_val)
    gen_val = data.DataGenerator([reader_val_x], reader_val_y).GetGenerator()
    #if cfg.debug: data.DebugGenerator(gen_val) 
    
    train.Train(model, gen_tr, gen_val)
    model.save_weights(GetModelPath())
    
def PredictOnImg(model, img_path):
    img = imread(img_path)
    img = img[..., np.newaxis]
    result = predict.GetPrediction(model, img)
    return result
    
    
def PredictOnFolder(model, src, dst):
    print('read date from {} and make predictions on {}'.format(src, dst))
    if not isdir(dst): makedirs(dst)
    for name in tqdm(listdir(src)):
        name_src = join(src, name)
        name_dst = join(dst, name)
        result = PredictOnImg(model, name_src)
        cv2.imwrite(name_dst, result)
                              
def Run(pretrained=False):
    model = m.GetAutoenocder(cfg.reg)
    if pretrained: 
        model.load_weights(GetModelPath())
    else:
        Train(model)
    
    print(model_name + ': ' + 'predict training data')
    PredictOnFolder(model, src=cfg.path_train, dst= join(model_name, 'train'))
        
    print(model_name + ': ' + 'predict testing data')
    PredictOnFolder(model, src=cfg.path_test, dst=join(model_name, 'test'))
    
if __name__=='__main__':
    Run(pretrained=False)
    
    