import os

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