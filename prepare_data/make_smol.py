import numpy as np


dataset = np.load('../../DGAN_DATASET/sunn1_full.npy')
dataset_smol = dataset[0:200, :, :, :]
print(dataset_smol.shape)
np.save('../../DGAN_DATASET/sunn1_200.npy', dataset_smol)