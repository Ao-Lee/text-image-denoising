import numpy as np

def mse(prediction, label):
    assert prediction.shape == label.shape
    err = (prediction-label)**2
    err = np.sum(err)
    err = err / prediction.size
    return err
    

    
