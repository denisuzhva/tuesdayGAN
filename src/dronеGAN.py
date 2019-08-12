from __future__ import absolute_import, division, print_function, unicode_literals
import time
import os
import tensorflow as tf
import numpy as np
#import cv2
import random
#import scipy.misc
#from utils import *


HEIGHT, WIDTH, CHANNEL = 256, 256, 2
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
EPOCH = 10
L_RATE = 2e-4
VER = 'cyber_dzhhhhhhhhh'
DATASET_LABEL = '256'
new_dzhhh_path = './' + VER

    
def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n)


def loadData():   
    dataset_dir = '../../DGAN_DATASET/sunn1_' + DATASET_LABEL + '.npy'
    np_dataset = np.load(dataset_dir)
    dataset_size = np_dataset.shape[0]
    tf_dataset = tf.data.Dataset.from_tensor_slices(np_dataset)
    tf_batch = tf_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    tf_iterator = tf_batch.make_initializable_iterator()
    return tf_iterator, dataset_size


def generator(input, random_dim, is_train, reuse=False):
    c8, c16, c32, c64, c128 = 512, 256, 128, 64, 32 # channel num
    s8 = 8
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s8 * s8 * c8], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c8 * s8 * s8], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
         #Convolution, bias, activation, repeat! 
        conv1 = tf.reshape(flat_conv1, shape=[-1, s8, s8, c8], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 16*16*256
        #Convolution, bias, activation, repeat! 
        conv2 = tf.layers.conv2d_transpose(act1, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(act3, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(act4, c128, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')
        
        #128*128*2
        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        act6 = tf.nn.tanh(conv6, name='act6')
        return act6


def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16, c32 = 64, 128, 256, 512, 1024  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        #Convolution, activation, bias, repeat! 
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = lrelu(bn1, n='act1')
         #Convolution, activation, bias, repeat! 
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        #Convolution, activation, bias, repeat! 
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
         #Convolution, activation, bias, repeat! 
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')
       
        #Convolution, activation, bias, repeat! 
        conv5 = tf.layers.conv2d(act4, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = lrelu(bn5, n='act5')


        # start from act5
        dim = int(np.prod(act5.get_shape()[1:]))
        fc1 = tf.reshape(act5, shape=[-1, dim], name='fc1')
      
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        acted_out = tf.nn.sigmoid(logits)
        return logits #, acted_out


def train():
    random_dim = 100
    
    with tf.variable_scope('input'):
        real_drone = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_drone')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
    
    # wgan
    fake_drone = generator(random_input, random_dim, is_train)
    
    real_result = discriminator(real_drone, is_train)
    fake_result = discriminator(fake_drone, is_train, reuse=True)

    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.            

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=L_RATE).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=L_RATE).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
    
    batch_size = BATCH_SIZE
    drone_iterator, dataset_size = loadData()
    
    num_o_batches = int(dataset_size / BATCH_SIZE)
    sess = tf.Session()
    sess.run(drone_iterator.initializer)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    #save_path = saver.save(sess, "/tmp/model.ckpt")
    #ckpt = tf.train.latest_checkpoint('./model/' + VER)
    #saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('************************************')
    print('GET READY FOR DZZZZHHHHHHHHHHHHHHHHH')
    print('************************************')
    print('total training sample num: %d' % dataset_size)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, num_o_batches, EPOCH))
    print('start training...')
    for epoch_iter in range(EPOCH):
        print("Running epoch %i/%i..." % (epoch_iter, EPOCH))
        for _ in range(num_o_batches):
            #d_iters = 5
            #g_iters = 1
            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            train_drone = sess.run(drone_iterator.get_next())
            print(train_drone[0, 100, 100, 0])
            sess.run(d_clip)
            _, dLoss = sess.run([trainer_d, d_loss],
                                feed_dict={random_input: train_noise, real_drone: train_drone, is_train: True})
            if (epoch_iter % 5 == 0):
                # train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})
            #print('train:[%d/%d],d_loss:%f,g_loss:%f' % (epoch_iter, batch_iter, dLoss, gLoss))
        sess.run(drone_iterator.initializer)
            
        # save check point every 500 epoch
        #if epoch_iter % 10 == 0:
        #    if not os.path.exists('./model/' + VER):
        #        os.makedirs('./model/' + VER)
        #    saver.save(sess, './model/' + VER + '/' + str(epoch_iter))  
        if epoch_iter % 2 == 0:
            if not os.path.exists(new_dzhhh_path):
                os.makedirs(new_dzhhh_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            dronetest = sess.run(fake_drone, feed_dict={random_input: sample_noise, is_train: False})
            # imgtest = imgtest * 255.0
            # imgtest.astype(np.uint8)
            print(dronetest.shape)
            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    train()