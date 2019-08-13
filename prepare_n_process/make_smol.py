import numpy as np


dataset = np.load('../../DGAN_DATASET/sunn1_full.npy')
dataset_smol = dataset[0:(dataset.shape[0] // 128) * 128, :, :, :]
print(dataset_smol.shape)
np.save('../../DGAN_DATASET/sunn1_1152.npy', dataset_smol)