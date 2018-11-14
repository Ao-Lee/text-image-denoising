from os.path import isdir, join
from os import makedirs, listdir

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 1
debug = False
reg = 0.0005
batch_size = 128


path_models = 'models'
if not isdir(path_models): makedirs(path_models)

path_show = 'show'
if not isdir(path_show): makedirs(path_show)


path_data = 'data'
assert isdir(path_data)

path_train = join(path_data, 'train')
path_label = join(path_data, 'train_cleaned')
path_test = join(path_data, 'test')

def SplitData():
    names = listdir(path_train)
    split = int(len(names)*0.9)
    names_tr = names[:split]
    names_val = names[split:]
    names_te = listdir(path_test)
    return names_tr, names_val, names_te
    
names_tr, names_val, names_te = SplitData()

# training x: [join(path_train, name) for name in names_tr]
# training y: [join(path_label, name) for name in names_tr]

# validation x: [join(path_train, name) for name in names_val]
# validation y: [join(path_label, name) for name in names_val]

# testing x: [join(path_test, name) for name in names_te]
