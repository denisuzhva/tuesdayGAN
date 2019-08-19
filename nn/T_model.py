from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
#from T_const import *



class T_Model():


    def __init__(self, 
                 freq_domain, 
                 channel_num):
        self.__channel_num = channel_num
        self.__c_gen = [512, 256, 128, 64, 32]
        self.__s_gen = freq_domain // 32
        self.__lstm_layers = 1
        self.__lstm_direction = 'unidirectional'
        self.__lstm_dropout = 0.5
        
        self.__c_dis = [32, 64, 128, 256, 512]


    def T_gen(self, input, random_dim, is_train, reuse=False):
        output_dim = self.__channel_num
        with tf.variable_scope('gen') as scope:
            if reuse:
                scope.reuse_variables()

            lstm1 = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=self.__lstm_layers, 
                                                  num_units=random_dim, 
                                                  direction=self.__lstm_direction,
                                                  dropout=self.__lstm_dropout if is_train else 0.,
                                                  name='lstm1')
            lstm_out, _ = lstm1(input, initial_state=None, training=is_train)

            w1 = tf.get_variable('w1', 
                                 shape=[random_dim, self.__s_gen * self.__s_gen * self.__c_gen[0]], 
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('b1', 
                                 shape=[self.__s_gen * self.__s_gen * self.__c_gen[0]], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
            flat_conv1 = tf.add(tf.matmul(lstm_out, w1), b1, name='flat_conv1')
            conv1 = tf.reshape(flat_conv1, 
                               shape=[-1, self.__s_gen, self.__s_gen, self.__c_gen[0]], 
                               name='conv1')
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
            act1 = tf.nn.relu(bn1, name='act1')
            conv2 = tf.layers.conv2d_transpose(act1, self.__c_gen[1], kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='conv2')
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
            act2 = tf.nn.relu(bn2, name='act2')
            conv3 = tf.layers.conv2d_transpose(act2, self.__c_gen[2], kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='conv3')
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
            act3 = tf.nn.relu(bn3, name='act3')
            conv4 = tf.layers.conv2d_transpose(act3, self.__c_gen[3], kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='conv4')
            bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
            act4 = tf.nn.relu(bn4, name='act4')
            conv5 = tf.layers.conv2d_transpose(act4, self.__c_gen[4], kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='conv5')
            bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
            act5 = tf.nn.relu(bn5, name='act5')       
            conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='conv6')
            #bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
            act6 = tf.nn.tanh(conv6, name='act6')
            return act6


    def T_dis(self, input, is_train, reuse=False):
        with tf.variable_scope('dis') as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = tf.layers.conv2d(input, self.__c_dis[0], kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                        name='conv1')
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
            act1 = tf.nn.leaky_relu(bn1, name='act1')
            conv2 = tf.layers.conv2d(act1, self.__c_dis[1], kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                        name='conv2')
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
            act2 = tf.nn.leaky_relu(bn2, name='act2')
            conv3 = tf.layers.conv2d(act2, self.__c_dis[2], kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                        name='conv3')
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
            act3 = tf.nn.leaky_relu(bn3, name='act3')
            conv4 = tf.layers.conv2d(act3, self.__c_dis[3], kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                        name='conv4')
            bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
            act4 = tf.nn.leaky_relu(bn4, name='act4')
            conv5 = tf.layers.conv2d(act4, self.__c_dis[4], kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                        name='conv5')
            bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
            act5 = tf.nn.leaky_relu(bn5, name='act5')
            dim = int(np.prod(act5.get_shape()[1:]))
            fc1 = tf.reshape(act5, shape=[-1, dim], name='fc1')
            w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
            logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
            #logits = tf.nn.sigmoid(logits)
            return logits
