import numpy as np


dataset = np.load('../../TGAN_DATASET/sunn1_full.npy')
dataset = dataset.astype('float32')
dataset = np.swapaxes(dataset, 2, 3)
dataset = np.swapaxes(dataset, 1, 2)
dataset = np.swapaxes(dataset, 0, 1)
print(dataset.shape)
np.save('../../DGAN_DATASET/sunn1_full.npy', dataset)