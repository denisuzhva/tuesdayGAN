import numpy as np
import tensorflow as tf


class T_Loader():

    def __init__(
            self,
            dataset_path,
            dataset_size,
            shuffle_buffer_size,
            batch_size
        ):
        self.__dataset_path = dataset_path
        self.__dataset_size = dataset_size
        self.__shuffle_buffer_size = shuffle_buffer_size
        self.__batch_size = batch_size
    
    def T_load(self):   
        dataset_dir =  self.__dataset_path + '{}'.format(self.__dataset_size) + '.npy'
        np_dataset = np.load(dataset_dir)
        tf_dataset = tf.data.Dataset.from_tensor_slices(np_dataset)
        tf_batch = tf_dataset.shuffle(self.__shuffle_buffer_size).batch(self.__batch_size)
        tf_iterator = tf_batch.make_initializable_iterator()
        return tf_iterator
