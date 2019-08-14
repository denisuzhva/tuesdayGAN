from __future__ import absolute_import, division, print_function, unicode_literals
import time
import os
import tensorflow as tf
import numpy as np
from T_thinker import *
from T_const import *


if __name__ == '__main__':
    thinker = T_Thinker(
                DATASET_PATH,
                DATASET_VER,
                DATASET_SIZE,
                EPOCH,
                TRAIN_BATCH_SIZE,
                SHUFFLE_BUFFER_SIZE,
                FD,
                TD,
                CHANNEL,
                RAND_DIM
            )

    #thinker.T_train(L_RATE, MODEL_PATH, TRAIN_BATCH_SIZE)
    thinker.T_test(MODEL_PATH, EPOCH, TEST_BATCH_SIZE, TEST_PATH, TEST_BATCHES)
