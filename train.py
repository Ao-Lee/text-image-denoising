from keras.callbacks import EarlyStopping
import cfg
import viz

def _TestPrediction(generator, model):
    batch = next(generator)
    imgs_x = batch[0]
    imgs_y = batch[1]

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
    
def Train(model, gen_tr, gen_te):
    earlystopper = EarlyStopping(patience=5, verbose=1)
    callbacks = [earlystopper]
    epochs = 1 if cfg.debug else 15
    steps_per_epoch = 30 if cfg.debug else 200
    model.fit_generator(  gen_tr, 
                          validation_data=gen_te,  
                          epochs=epochs, 
                          verbose=1, 
                          workers=4,
                          steps_per_epoch=steps_per_epoch, 
                          validation_steps=30,
                          callbacks=callbacks
                          )
    
    # if cfg.debug: _TestPrediction(gen_te, model)
    

if __name__=='__main__':
    pass