from __future__ import absolute_import, division, print_function, unicode_literals
import time
import os
import tensorflow as tf
import numpy as np
from T_model import *
from T_loader import *


class T_Thinker():
    
    def __init__(
            self, 
            dataset_path,
            dataset_ver,
            dataset_size,
            epoch_num,
            batch_size,
            shuffle_buffer_size,
            freq_domain, 
            time_domain, 
            channel_num,
            random_dim
        ):
        self.__dataset_size = dataset_size
        self.__epoch_num = epoch_num
        self.__freq_domain = freq_domain
        self.__time_domain = time_domain
        self.__channel_num = channel_num
        self.__random_dim = random_dim
        self.__model = T_Model(
                freq_domain, 
                channel_num
            )
        self.__loader = T_Loader(
                dataset_path, 
                dataset_size,
                shuffle_buffer_size,
                batch_size
            )

    def T_train(self, learning_rate, model_path, train_batch_size):
        with tf.variable_scope('input'):
            real_drone = tf.placeholder(tf.float32, 
                                        shape = [None, self.__freq_domain, self.__time_domain, self.__channel_num], 
                                        name='real_drone')
            random_input = tf.placeholder(tf.float32, shape=[None, self.__random_dim], name='rand_input')
            is_train = tf.placeholder(tf.bool, name='is_train')
        fake_drone = self.__model.T_gen(random_input, self.__random_dim, is_train)
        real_result = self.__model.T_dis(real_drone, is_train)
        fake_result = self.__model.T_dis(fake_drone, is_train, reuse=True)
        d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)
        g_loss = -tf.reduce_mean(fake_result)
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'dis' in var.name]
        g_vars = [var for var in t_vars if 'gen' in var.name]
        trainer_d = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=d_vars)
        trainer_g = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=g_vars)
        d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
        batch_iterator = self.__loader.T_load()
        next_batch = batch_iterator.get_next()
        num_o_batches = int(self.__dataset_size / train_batch_size)
        sess = tf.Session()
        sess.run(batch_iterator.initializer)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('************************************')
        print('GET READY FOR DZZZZHHHHHHHHHHHHHHHHH')
        print('************************************')
        print('total training sample num: %d' % self.__dataset_size)
        print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (
                train_batch_size, 
                num_o_batches, 
                self.__epoch_num
            ))
        print('start listening...')
        start_time = time.time()
        for epoch_iter in range(self.__epoch_num):
            print("Running epoch %i/%i..." % (epoch_iter+1, self.__epoch_num))
            for batch_iter in range(num_o_batches):
                #d_iters = 5
                #g_iters = 1
                train_noise = np.random.uniform(-1.0, 1.0, size=[train_batch_size, self.__random_dim]).astype(np.float32)
                train_drone = sess.run(next_batch)
                #print(train_drone[0, 100, 100, 0])
                sess.run(d_clip)
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_drone: train_drone, is_train: True})
                if (epoch_iter + 1) % 5 == 0:
                    # train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                    _, gLoss = sess.run([trainer_g, g_loss],
                                        feed_dict={random_input: train_noise, is_train: True})
                    print('train: [%d ep / %d ba], d_loss:%f, g_loss:%f' % (epoch_iter+1, batch_iter+1, dLoss, gLoss))
            sess.run(batch_iterator.initializer)    
            if (epoch_iter + 1) % 100 == 0:
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                saver.save(sess, model_path + str(epoch_iter+1))  
            print("--- %s seconds passed... ---" % (time.time() - start_time))
        coord.request_stop()
        coord.join(threads)

    def T_test(self, model_path, model_name, test_batch_size, test_path, test_batches):
        #tf.reset_default_graph()
        with tf.variable_scope('input'):
            random_input = tf.placeholder(tf.float32, shape=[None, self.__random_dim], name='rand_input')
            is_train = tf.placeholder(tf.bool, name='is_train')
        fake_drone = self.__model.T_gen(random_input, self.__random_dim, is_train, reuse=True)
        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        restore_path = model_path + str(model_name)
        saver.restore(sess, restore_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('************************************')
        print('GET READY FOR DZZZZHHHHHHHHHHHHHHHHH')
        print('************************************')
        print('start generating dzhhh...')
        for test_iter in range(test_batches):
            print("Generating batch %i/%i..." % (test_iter+1, test_batches)) 
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[test_batch_size, self.__random_dim]).astype(np.float32)
            drone_test = sess.run(fake_drone, feed_dict={random_input: sample_noise, is_train: False})
            np.save(test_path + str(test_iter) , drone_test)
            #print('train:[%d],d_loss:%f,g_loss:%f' % (epoch, dLoss, gLoss))
        coord.request_stop()
        coord.join(threads)
