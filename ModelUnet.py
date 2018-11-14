from os.path import join, isdir
from os import makedirs, listdir
from skimage.io import imread
from tqdm import tqdm
import cv2
import numpy as np
import gzip

from metrics import mse
import cfg
import train
import data
import predict
import model as m
import viz    
import ModelAutoEncoder, ModelHighPassFilter, ModelBackGround

model_name = 'Unet'

others = []
others.append(ModelAutoEncoder.model_name)
others.append(ModelBackGround.model_name)
# others.append(ModelHighPassFilter.model_name)


# imgs comming from each model, plus the original training data together forms the 'depth' of the data
input_channel = len(others) + 1


def GetModelPath():
    return join(cfg.path_models, model_name + '.h5')
    
def Train(model):
    reader_tr_xs = []
    # this is the training data comming from other models
    for other in others:
        reader = data.PathReader(join(other, 'train'), cfg.names_tr)
        reader_tr_xs.append(reader)
        
    reader_tr_x_origin = data.PathReader(cfg.path_train, cfg.names_tr)
    # this is the original training data
    reader_tr_xs.append(reader_tr_x_origin)
    reader_tr_y = data.PathReader(cfg.path_label, cfg.names_tr)
    gen_tr = data.DataGenerator(reader_tr_xs, reader_tr_y).GetGenerator()
        
    
    reader_val_xs = []
    # this is the training data comming from other models
    for other in others:
        reader = data.PathReader(join(other, 'train'), cfg.names_val)
        reader_val_xs.append(reader)
    reader_val_x_origin = data.PathReader(cfg.path_train, cfg.names_val)
    # this is the original training data
    reader_val_xs.append(reader_val_x_origin)
    reader_val_y = data.PathReader(cfg.path_label, cfg.names_val)
    gen_val = data.DataGenerator(reader_val_xs, reader_val_y).GetGenerator()
        
    train.Train(model, gen_tr, gen_val)
    model.save_weights(GetModelPath())
    
    
def PredictOnImg(model, img_paths):
    assert len(img_paths) == input_channel
    
    x = []
    for path in img_paths:
        img = imread(path)                          # (H, W)
        x.append(img)
    x = np.stack(x, axis=-1)                        # (H, W, C)
        
    result = predict.GetPrediction(model, x)        # (H, W)
    return result
    
# srcs is a list of folders
def PredictOnFolder(model, srcs, dst):
    print('make predictions on {}'.format(dst))
    if not isdir(dst): makedirs(dst)
    assert len(srcs) == input_channel

    for img in tqdm(listdir(srcs[0])):
        names_src = [join(src, img) for src in srcs]
        name_dst = join(dst, img)
        result = PredictOnImg(model, names_src)
        cv2.imwrite(name_dst, result)
        
def Eval():
    err = 0
    for name in cfg.names_val:
  
        img_predicton = imread(join(model_name, 'train', name))
        img_predicton = img_predicton/255
        
        img_label = imread(join(cfg.path_label, name))
        img_label = img_label/255
        
        err += mse(prediction=img_predicton, label=img_label)
    return err/len(cfg.names_val)
    
def Run(pretrained=False):
    model = m.GetUNet(cfg.reg, input_channel=input_channel)
    if pretrained: 
        model.load_weights(GetModelPath())
    else:
        Train(model)
    
    print(model_name + ': ' + 'predict training data')
    srcs = [join(other, 'train') for other in others]
    srcs.append(cfg.path_train)
    PredictOnFolder(model, srcs, join(model_name, 'train'))
        
    print(model_name + ': ' + 'predict testing data')
    srcs = [join(other, 'test') for other in others]
    srcs.append(cfg.path_test)
    PredictOnFolder(model, srcs, join(model_name, 'test'))
    

    
    print('generate visualization examples')
    for name in tqdm(cfg.names_val):
        path_origin = join(cfg.path_train, name)
        path_gt = join(cfg.path_label, name)
        path_prediction = join(model_name, 'train', name)
        
        img_origin = viz.Gray2Rgb(imread(path_origin))
        img_gt = viz.Gray2Rgb(imread(path_gt))
        img_prediction = viz.Gray2Rgb(imread(path_prediction))
        
        imgs = [img_origin, img_gt, img_prediction]
        imgs = viz.MergeImage(imgs, color=(40,0,40))
        viz.ShowImg(imgs, save_path=join(cfg.path_show, name))
        
    err = Eval()
    print('mse error is {}'.format(err))
    return err

    
def Submit():
    path_submit = 'MySubmission.csv.gz'
    submission = gzip.open(path_submit,'wt')
    submission.write('id, value\n')

    print('generating submit data')
    for name in tqdm(cfg.names_te):
        path = join(model_name,'test', name)
        img = imread(path)
        img = img/255
        for j in range(img.shape[1]):
            for i in range(img.shape[0]):
                submission.write("{}_{}_{},{}\n".format(name, i+1, j+1, img[i,j]))
    submission.close()
    
if __name__=='__main__':
    err = Run(pretrained=False)
    Submit()
    
    

