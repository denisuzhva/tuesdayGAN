from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
#from T_const import *


class T_Model():

    def __init__(
            self, 
            freq_domain, 
            channel_num
        ):
        self.__freq_domain = freq_domain
        self.__channel_num = channel_num

    def T_gen(self, input, random_dim, is_train, reuse=False):
        c1, c2, c3, c4, c5 = 512, 256, 128, 64, 32
        s0 = self.__freq_domain // 32
        output_dim = self.__channel_num
        with tf.variable_scope('gen') as scope:
            if reuse:
                scope.reuse_variables()
            w1 = tf.get_variable('w1', shape=[random_dim, s0 * s0 * c1], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('b1', shape=[s0 * s0 * c1], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
            flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
            conv1 = tf.reshape(flat_conv1, shape=[-1, s0, s0, c1], name='conv1')
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
            act1 = tf.nn.relu(bn1, name='act1')
            conv2 = tf.layers.conv2d_transpose(act1, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='conv2')
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
            act2 = tf.nn.relu(bn2, name='act2')
            conv3 = tf.layers.conv2d_transpose(act2, c3, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='conv3')
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
            act3 = tf.nn.relu(bn3, name='act3')
            conv4 = tf.layers.conv2d_transpose(act3, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='conv4')
            bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
            act4 = tf.nn.relu(bn4, name='act4')
            conv5 = tf.layers.conv2d_transpose(act4, c5, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
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
        c1, c2, c3, c4, c5 = 64, 128, 256, 512, 1024
        with tf.variable_scope('dis') as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = tf.layers.conv2d(input, c1, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                        name='conv1')
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
            act1 = tf.nn.leaky_relu(bn1, name='act1')
            conv2 = tf.layers.conv2d(act1, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                        name='conv2')
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
            act2 = tf.nn.leaky_relu(bn2, name='act2')
            conv3 = tf.layers.conv2d(act2, c3, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                        name='conv3')
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
            act3 = tf.nn.leaky_relu(bn3, name='act3')
            conv4 = tf.layers.conv2d(act3, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                        name='conv4')
            bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
            act4 = tf.nn.leaky_relu(bn4, name='act4')
            conv5 = tf.layers.conv2d(act4, c5, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
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
            logits = tf.nn.sigmoid(logits)
            return logits
