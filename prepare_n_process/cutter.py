import numpy as np


dataset_name = 'anim1'
dataset_size = 7296
dataset_newsize = 7296 // 2
dataset = np.load('../../TGAN_DATASET/' + dataset_name + '/{}.npy'.format(dataset_size))
dataset_new = dataset[0:dataset_newsize, :, :, :]
np.save('../../TGAN_DATASET/' + dataset_name + '/{}.npy'.format(dataset_newsize), dataset_new)