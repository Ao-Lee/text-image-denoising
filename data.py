import os
import numpy as np
from skimage.io import imread

import cfg
import viz

class PathReader(object):
    
    def __init__(self, root, names):
        self.root = root
        self.paths = [os.path.join(self.root, name) for name in names]
        self.len = len(self.paths)
    def Get(self, idx):
        assert idx < self.len
        return self.paths[idx]
        
class DataGenerator(object):
    def __init__(self, reader_x, reader_y, batch_size=16):
        self.x = reader_x
        self.y = reader_y
        assert self.x.len == self.y.len
        self.len =self.x.len
        self.batch_size = batch_size
        
    @staticmethod
    def Read(filepath):
        return imread(filepath)

    @staticmethod
    def PreprocessInput_X(image):
        image = image[..., np.newaxis]
        image = image/255 - 0.5
        return image
        
    @staticmethod
    def PreprocessInput_Y(image):
        image = image[..., np.newaxis]
        image = image/255
        return image
    
    @staticmethod
    def Crop(img_x, img_y, crop_size):
        assert len(img_x.shape) == 2 #binary image
        assert len(img_y.shape) == 2 #binary image
        assert img_x.shape == img_y.shape # same size
        h, w = img_x.shape
        
        dy, dx = crop_size
        x = np.random.randint(0, w - dx + 1)
        y = np.random.randint(0, h - dy + 1)
        return img_x[y:(y+dy), x:(x+dx)], img_y[y:(y+dy), x:(x+dx)]

    def GetGenerator(self):
        while True:
            batch_x = []
            batch_y = []
            for _ in range(self.batch_size):
                idx = np.random.randint(low=0, high=self.len)
                path_x = self.x.Get(idx)
                path_y = self.y.Get(idx)
                
                img_x = self.Read(path_x)
                img_y = self.Read(path_y)
                
                img_x, img_y = self.Crop(img_x, img_y, crop_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH))
                batch_x.append(img_x)
                batch_y.append(img_y)
                
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
                
            batch_x = self.PreprocessInput_X(batch_x)
            batch_y = self.PreprocessInput_Y(batch_y)
            yield (batch_x, batch_y)
            
def DebugGenerator(gen):  
    data = next(gen)
    imgs_x = data[0]
    imgs_y = data[1]

    batch_size = imgs_x.shape[0]
    for idx in range(batch_size):
        img_x = imgs_x[idx][:,:,0]
        img_y = imgs_y[idx][:,:,0]
        viz.ShowBinaryImg(img_x)
        viz.ShowBinaryImg(img_y)
        break