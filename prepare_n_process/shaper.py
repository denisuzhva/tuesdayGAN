import numpy as np


dataset = np.load('../../TGAN_DATASET/sunn2/full.npy')
dataset_new = dataset[0:(dataset.shape[0] // 128) * 128, :, :, :]
dataset_new = dataset_new.astype('float32')
dataset_shape = dataset_new.shape
print(dataset_shape)
np.save('../../TGAN_DATASET/sunn2/{}.npy'.format(dataset_shape[0]), dataset_new)