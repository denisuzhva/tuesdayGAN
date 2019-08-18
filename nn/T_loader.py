import numpy as np
import tensorflow as tf



class T_Loader():


    def __init__(self,
                 dataset_path,
                 dataset_size):
        self.__dataset_path = dataset_path
        self.__dataset_size = dataset_size

    
    def T_load(self):   
        dataset_dir =  self.__dataset_path + '{}'.format(self.__dataset_size) + '.npy'
        np_dataset = np.load(dataset_dir)
        return np_dataset
