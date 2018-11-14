from keras.models import Model
from keras.layers import Input, Dropout, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers.merge import concatenate
# from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
import cfg


def GetUNet(reg, input_channel):    
    r = regularizers.l2(reg)    
    # (128, 128, input_channel)
    inputs = Input((cfg.IMG_HEIGHT, cfg.IMG_WIDTH, input_channel))
    # (128, 128, input_channel) -> (128, 128, 16)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', kernel_regularizer=r) (inputs)
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
    if not cfg.debug: model.summary()
    return model
    
    
def GetAutoenocder_V1_Origin(reg):
    input_img = Input(shape=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS), name='image_input')
    
    #enoder 
    x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2,2), padding='same', name='pool1')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same', name='pool2')(x)
    
    #decoder
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2), name='upsample1')(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2), name='upsample2')(x)
    x = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
    
    #model
    autoencoder = Model(inputs=input_img, outputs=x)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    if not cfg.debug: autoencoder.summary()
    return autoencoder
  
def GetAutoenocder_v2(reg):
    # train_loss 0.1193 - val_loss: 0.1075
    # # (32, 32, 1)
    inputs = Input(shape=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS))
    x = inputs
    # (32, 32, 1) -> (16, 16, 32)
    x = Conv2D(32, kernel_size=3, strides=2, activation='elu', padding='same')(x)
    # (16, 16, 32) -> (8, 8, 64)
    x = Conv2D(64, kernel_size=3, strides=2, activation='elu', padding='same')(x)
    # (8, 8, 64) -> (4, 4, 128)
    x = Conv2D(128, kernel_size=3, strides=2, activation='elu', padding='same')(x)
    # (4, 4, 128) -> (4, 4, 16)
    x = Conv2D(8, kernel_size=3, activation='elu', padding='same')(x)
    
    # (4, 4, 16) -> (4, 4, 128)
    x = Conv2DTranspose(128, kernel_size=3, activation='relu', padding='same')(x)
    # (4, 4, 128) -> (8, 8, 64)
    x = Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    # (8, 8, 64) -> (16, 16, 32)
    x = Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    # (16, 16, 32) -> (32, 32, 16)
    x = Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    # (32, 32, 16) -> (32, 32, 1)
    x = Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')(x)

    
    #model
    autoencoder = Model(inputs=inputs, outputs=x)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    if not cfg.debug: autoencoder.summary()
    return autoencoder
    
def GetAutoenocder(reg):
    inputs = Input(shape=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_CHANNELS))
    x = inputs
    # (32, 32, 1) -> (16, 16, 64)
    x = Conv2D(64, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    # (16, 16, 64) -> (8, 8, 128)
    x = Conv2D(128, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    # (8, 8, 128) -> (4, 4, 256)
    x = Conv2D(256, kernel_size=3, strides=2, activation='relu', padding='same')(x)

    # (4, 4, 256) -> (8, 8, 128)
    x = Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    # (8, 8, 128) -> (16, 16, 64)
    x = Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    # (16, 16, 64) -> (32, 32, 1)
    x = Conv2DTranspose(1, kernel_size=3, strides=2, activation='sigmoid', padding='same')(x)

    #model
    autoencoder = Model(inputs=inputs, outputs=x)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    if not cfg.debug: autoencoder.summary()
    return autoencoder
    
    
if __name__=='__main__':
    model = GetAutoenocder(cfg.reg)
    