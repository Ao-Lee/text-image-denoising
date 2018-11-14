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
    def __init__(self, readers_x, reader_y, batch_size=cfg.batch_size):
        self.xs = readers_x
        self.y = reader_y

        self.len =self.y.len
        for x in self.xs:
            assert x.len == self.y.len
        
        self.batch_size = batch_size
        
    @staticmethod
    def Read(filepath):
        return imread(filepath)

    @staticmethod
    def PreprocessInput_X(image):
        image = image/255 - 0.5
        return image
        
    @staticmethod
    def PreprocessInput_Y(image):
        image = image[..., np.newaxis]
        image = image/255
        return image
    
    @staticmethod
    def Crop(img_x, img_y, crop_size):
        assert len(img_x.shape) == 3
        assert len(img_y.shape) == 2 #binary image
        assert img_x.shape[:-1] == img_y.shape # same size
        h, w = img_y.shape
        
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
                
                path_xs = [x.Get(idx) for x in self.xs]
                path_y = self.y.Get(idx)
                
                img_xs = [self.Read(path_x) for path_x in path_xs]
                img_y = self.Read(path_y)
                
                # stack images from each reader together as 'channels'
                img_x = np.stack(img_xs, axis=-1)
                
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