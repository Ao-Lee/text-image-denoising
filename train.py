
from keras.callbacks import EarlyStopping, ModelCheckpoint

import cfg
import data
import model as m
import viz

def _TestPrediction(generator, model):
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
        viz.ShowBinaryImg(img_x)
        print('label')
        viz.ShowBinaryImg(img_y)
        print('prediction')
        viz.ShowBinaryImg(result)
        break
    
def Train():
    split = int(len(cfg.names)*0.9)
    names_tr = cfg.names[:split]
    names_te = cfg.names[split:]

    reader_tr_x = data.PathReader(cfg.path_train, names_tr)
    reader_tr_y = data.PathReader(cfg.path_label, names_tr)
    gen_tr = data.DataGenerator(reader_tr_x, reader_tr_y).GetGenerator()
    if cfg.debug: data.DebugGenerator(gen_tr)

    reader_te_x = data.PathReader(cfg.path_train, names_te)
    reader_te_y = data.PathReader(cfg.path_label, names_te)
    gen_te = data.DataGenerator(reader_te_x, reader_te_y).GetGenerator()
    if cfg.debug: data.DebugGenerator(gen_te)  
    
    
    model = m.GetUNetModel(cfg.reg)
    
    

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    callbacks = [earlystopper]
    if cfg.SAVE_LOAD:  callbacks.append(checkpointer)
    
    model.fit_generator(  gen_tr, 
                          validation_data=gen_te,  
                          epochs=10, 
                          verbose=1, 
                          workers=4,
                          steps_per_epoch=300, 
                          validation_steps=50)
    
    if cfg.debug: _TestPrediction(gen_te, model)
    return model