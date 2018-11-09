import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers

'''
kaggle project
https://www.kaggle.com/c/denoising-dirty-documents
'''

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
path_train = 'data/train/'
path_label = 'data/train_cleaned'
path_test = 'data/test/'
names = os.listdir(path_train)


SAVE_LOAD = False
debug = True

reg = 0.0005
seed = 231
# random.seed = seed
# np.random.seed = seed

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
                
                img_x, img_y = self.Crop(img_x, img_y, crop_size=(IMG_HEIGHT, IMG_WIDTH))
                batch_x.append(img_x)
                batch_y.append(img_y)
                
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
                
            batch_x = self.PreprocessInput_X(batch_x)
            batch_y = self.PreprocessInput_Y(batch_y)
            yield (batch_x, batch_y)
            
def ShowBinaryImg(img, save_path=None):
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
    if save_path is not None:
        plt.imsave(save_path, img, cmap = plt.cm.gray)

def GetUNetModel(reg):
    # Build U-Net model
    
    r = regularizers.l2(reg)
    
    # (128, 128, 1)
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)
    # (128, 128, 3) -> (128, 128, 16)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (s)
    c1 = Dropout(0.1) (c1)
    # (128, 128, 16) -> (128, 128, 16)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c1)
    # (128, 128, 16) -> (64, 64, 16)
    p1 = MaxPooling2D((2, 2)) (c1)
    # (64, 64, 16) -> (64, 64, 32)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (p1)
    c2 = Dropout(0.1) (c2)
    # (64, 64, 32) -> (64, 64, 32)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c2)
    # (64, 64, 32) -> (32, 32, 32)
    p2 = MaxPooling2D((2, 2)) (c2)
    # (32, 32, 32) -> (32, 32, 64)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (p2)
    c3 = Dropout(0.2) (c3)
    # (32, 32, 64) -> (32, 32, 64)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c3)
    # (32, 32, 64) -> (16, 16, 64)
    p3 = MaxPooling2D((2, 2)) (c3)
    # (16, 16, 64) -> (16, 16, 128)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (p3)
    c4 = Dropout(0.2) (c4)
    # (16, 16, 128) -> (16, 16, 128)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c4)
    # (16, 16, 128) -> (8, 8, 128)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    # (8, 8, 128) -> (8, 8, 256)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (p4)
    c5 = Dropout(0.3) (c5)
    # (8, 8, 256) -> (8, 8, 256)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c5)
    
    # (8, 8, 256) -> (16, 16, 128)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=r) (c5)
    # (16, 16, 256)
    u6 = concatenate([u6, c4])
    # (16, 16, 256) -> (16, 16, 128)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (u6)
    c6 = Dropout(0.2) (c6)
    # (16, 16, 128) -> (16, 16, 128)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c6)
    # (16, 16, 128) -> (32, 32, 64)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=r) (c6)
    # (32, 32, 128)
    u7 = concatenate([u7, c3])
    # (32, 32, 128) -> (32, 32, 64)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (u7)
    c7 = Dropout(0.2) (c7)
    # (32, 32, 64) -> (32, 32, 64)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c7)
    # (32, 32, 64) -> (64, 64, 32)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=r) (c7)
    # (64, 64, 64)
    u8 = concatenate([u8, c2])
    # (64, 64, 64) -> (64, 64, 32)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (u8)
    c8 = Dropout(0.1) (c8)
    # (64, 64, 32) -> (64, 64, 32)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c8)
    # (64, 64, 32) -> (128, 128, 16)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=r) (c8)
    # (128, 128, 16) -> (128, 128, 32)
    u9 = concatenate([u9, c1], axis=3)
    # (128, 128, 32) -> (128, 128, 16)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (u9)
    c9 = Dropout(0.1) (c9)
    # (128, 128, 16) -> (128, 128, 16)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (c9)
    # (128, 128, 16) -> (128, 128, 1)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[])
    model.summary()
    return model
    

def DebugGenerator(gen):  
    data = next(gen)
    imgs_x = data[0]
    imgs_y = data[1]

    batch_size = imgs_x.shape[0]
    for idx in range(batch_size):
        img_x = imgs_x[idx][:,:,0]
        img_y = imgs_y[idx][:,:,0]
        ShowBinaryImg(img_x)
        ShowBinaryImg(img_y)
        break

def TestPrediction(generator, model):
    data = next(generator)
    imgs_x = data[0]
    imgs_y = data[1]

    batch_size = imgs_x.shape[0]
    results = model.predict_on_batch(imgs_x)
    for idx in range(batch_size):
        img_x = imgs_x[idx][:,:,0]
        img_y = imgs_y[idx][:,:,0]
        result = results[idx][:,:,0]
        print('origin')
        ShowBinaryImg(img_x)
        print('label')
        ShowBinaryImg(img_y)
        print('prediction')
        ShowBinaryImg(result)
        break
    
def GetStartPosition(h, w):
    results = []
    dh, dw = int(IMG_HEIGHT/2), int(IMG_WIDTH/2)

    possible_ys = list(np.arange(0, h - IMG_HEIGHT + 1, dh))
    possible_ys.append(h - IMG_HEIGHT)
    
    possible_xs = list(np.arange(0, w - IMG_WIDTH + 1, dw))
    possible_xs.append(w - IMG_WIDTH)
    
    for start_y in possible_ys:
        for start_x in possible_xs:
            results.append([start_y, start_x])
    return np.array(results)
    
    
def GetPrediction(model, img):
    
    h, w = img.shape
    positions = GetStartPosition(h, w)
    count = np.zeros(shape=[h, w])
    batch = []
    for idx in range(len(positions)):
        start_y = positions[idx, 0]
        start_x = positions[idx, 1]
        end_y = start_y + IMG_HEIGHT
        end_x = start_x + IMG_WIDTH
        content = img[start_y:end_y, start_x:end_x]
        batch.append(content)
        count[start_y:end_y, start_x:end_x] += 1

    batch = np.stack(batch, axis=0)     #(Batch, IMG_HEIGHT, IMG_WIDTH)
    batch = batch[..., np.newaxis]      #(Batch, IMG_HEIGHT, IMG_WIDTH, 1)
    batch = batch/255 - 0.5
    
    assert np.sum(count==0) ==0

    prediction = model.predict(batch)   #(Batch, IMG_HEIGHT, IMG_WIDTH, 1)
    
    summed_prob = np.zeros(shape=[h, w])
    # iterate over batch 
    for idx in range(len(positions)):
        start_y = positions[idx, 0]
        start_x = positions[idx, 1]
        end_y = start_y + IMG_HEIGHT
        end_x = start_x + IMG_WIDTH
        summed_prob[start_y:end_y, start_x:end_x] += prediction[idx, :, :, 0]

    average_prob = summed_prob / count
    return average_prob
    
def Train():
    split = int(len(names)*0.9)
    names_tr = names[:split]
    names_te = names[split:]

    reader_tr_x = PathReader(path_train, names_tr)
    reader_tr_y = PathReader(path_label, names_tr)
    gen_tr = DataGenerator(reader_tr_x, reader_tr_y).GetGenerator()
    if debug: DebugGenerator(gen_tr)

    reader_te_x = PathReader(path_train, names_te)
    reader_te_y = PathReader(path_label, names_te)
    gen_te = DataGenerator(reader_te_x, reader_te_y).GetGenerator()
    if debug: DebugGenerator(gen_te)  
    
    
    model = GetUNetModel(reg)
    
    

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    callbacks = [earlystopper]
    if SAVE_LOAD:  callbacks.append(checkpointer)
    
    model.fit_generator(  gen_tr, 
                          validation_data=gen_te,  
                          epochs=10, 
                          verbose=1, 
                          workers=4,
                          steps_per_epoch=300, 
                          validation_steps=50)
    
    if debug: TestPrediction(gen_te, model)
    return model
    
    

if __name__=='__main__':
    # model = Train()
    split = int(len(names)*0.9)
    tests = names[split:]

    result_path = 'result'
    for name in tests:
        img = imread(os.path.join(path_train, name))
        result = GetPrediction(model, img)
        save_path = os.path.join(result_path, name)
        ShowBinaryImg(result, save_path=save_path)
    
        plt.savefig('test2.png')
