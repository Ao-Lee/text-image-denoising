import numpy as np
import cfg

def _GetStartPosition(h, w):
    results = []
    dh, dw = int(cfg.IMG_HEIGHT/2), int(cfg.IMG_WIDTH/2)

    possible_ys = list(np.arange(0, h - cfg.IMG_HEIGHT + 1, dh))
    possible_ys.append(h - cfg.IMG_HEIGHT)
    
    possible_xs = list(np.arange(0, w - cfg.IMG_WIDTH + 1, dw))
    possible_xs.append(w - cfg.IMG_WIDTH)
    
    for start_y in possible_ys:
        for start_x in possible_xs:
            results.append([start_y, start_x])
    return np.array(results)
    
    
def GetPrediction(model, img):
    assert len(img.shape) == 3
    h, w, c = img.shape

    positions = _GetStartPosition(h, w)
    count = np.zeros(shape=[h, w])
    batch = []
    for idx in range(len(positions)):
        start_y = positions[idx, 0]
        start_x = positions[idx, 1]
        end_y = start_y + cfg.IMG_HEIGHT
        end_x = start_x + cfg.IMG_WIDTH
        content = img[start_y:end_y, start_x:end_x, :]
        batch.append(content)
        count[start_y:end_y, start_x:end_x] += 1

    batch = np.stack(batch, axis=0)     #(Batch, IMG_HEIGHT, IMG_WIDTH, Channel)
    batch = batch/255 - 0.5
    
    assert np.sum(count==0) ==0

    prediction = model.predict(batch)   #(Batch, IMG_HEIGHT, IMG_WIDTH, 1)
    
    summed_prob = np.zeros(shape=[h, w])
    # iterate over batch 
    for idx in range(len(positions)):
        start_y = positions[idx, 0]
        start_x = positions[idx, 1]
        end_y = start_y + cfg.IMG_HEIGHT
        end_x = start_x + cfg.IMG_WIDTH
        summed_prob[start_y:end_y, start_x:end_x] += prediction[idx, :, :, 0]

    average_prob = summed_prob / count
    
    average_prob = (average_prob*255).astype(np.uint8)
    return average_prob