from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
# from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers

import cfg

def GetUNetModel(reg):
    # Build U-Net model
    
    r = regularizers.l2(reg)
    
    # (128, 128, 1)
    inputs = Input((cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS))
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
    
    
    