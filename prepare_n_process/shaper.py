import numpy as np


BATCH_SIZE = 64
dataset_name = 'anim1'
dataset = np.load('../../TGAN_DATASET/' + dataset_name + '/full.npy')
dataset_new = dataset[0:(dataset.shape[0] // BATCH_SIZE) * BATCH_SIZE, :, :, :]
dataset_new = dataset_new.astype('float32')
dataset_shape = dataset_new.shape
print(dataset_shape)
np.save('../../TGAN_DATASET/' + dataset_name + '/{}.npy'.format(dataset_shape[0]), dataset_new)