import numpy as np


dataset = np.load('../../DGAN_DATASET/sunn1_full.npy')
dataset_smol = dataset[0:256, :, :, :]
print(dataset_smol.shape)
np.save('../../DGAN_DATASET/sunn1_256.npy', dataset_smol)