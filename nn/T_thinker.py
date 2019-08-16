from __future__ import absolute_import, division, print_function, unicode_literals
import time
import os
import tensorflow as tf
import numpy as np
from T_model import *
from T_loader import *



class T_Thinker():


    def __init__(self, 
                 dataset_path,
                 dataset_ver,
                 dataset_size,
                 epoch_num,
                 freq_domain, 
                 time_domain, 
                 channel_num,
                 random_dim):
        self.__dataset_size = dataset_size
        self.__epoch_num = epoch_num
        self.__freq_domain = freq_domain
        self.__time_domain = time_domain
        self.__channel_num = channel_num
        self.__random_dim = random_dim
        self.__reuse_flag = False 
        self.__model = T_Model(freq_domain, channel_num)
        self.__loader = T_Loader(dataset_path, dataset_size)


    def T_train(self, learning_rate, model_path, train_batch_size):
        self.__reuse_flag = True
        
        with tf.variable_scope('input'):
            real_data = tf.placeholder(tf.float32, 
                                        shape = [None, self.__freq_domain, self.__time_domain, self.__channel_num], 
                                        name='real_data')
            random_input = tf.placeholder(tf.float32, shape=[None, self.__random_dim], name='rand_input')
            is_train = tf.placeholder(tf.bool, name='is_train')
        
        num_o_batches = int(self.__dataset_size / train_batch_size)
        np_dataset = self.__loader.T_load()

        dataset = tf.data.Dataset.from_tensor_slices(real_data)
        #dataset = dataset.shuffle(self.__dataset_size)
        dataset = dataset.batch(train_batch_size)
        batch_iterator = dataset.make_initializable_iterator()
        real_batch = batch_iterator.get_next()

        fake_batch = self.__model.T_gen(random_input, self.__random_dim, is_train)
        real_result = self.__model.T_dis(real_batch, is_train)
        fake_result = self.__model.T_dis(fake_batch, is_train, reuse=True)

        d_loss = tf.reduce_mean(fake_result - real_result)
        g_loss = -tf.reduce_mean(fake_result)
        
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'dis' in var.name]
        g_vars = [var for var in t_vars if 'gen' in var.name]
        
        trainer_d = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=d_vars)
        trainer_g = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=g_vars)
        
        d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print('************************************')
            print('GET READY FOR DZZZZHHHHHHHHHHHHHHHHH')
            print('************************************')
            print('total training sample num: %d' % self.__dataset_size)
            print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (train_batch_size, 
                                                                            num_o_batches, 
                                                                            self.__epoch_num))
            print('start listening...')

            start_time = time.time()

            d_iters = 5
            g_iters = 1

            for epoch_iter in range(self.__epoch_num):
                print("Running epoch %i/%i..." % (epoch_iter+1, self.__epoch_num))

                train_noise = np.random.uniform(-1.0, 1.0, size=[train_batch_size, self.__random_dim]).astype(np.float32)

                dis_loss_val = 0
                for d_iter in range(d_iters):
                    sess.run([batch_iterator.initializer, d_clip], 
                         feed_dict={random_input: train_noise, real_data: np_dataset, is_train: True})
                    try:
                        print('d_iter:%i' % d_iter)
                        batch_it = 1
                        while True:
                            print('batch iter:%i' % batch_it)
                            batch_it += 1
                            _, dis_loss_val_inst = sess.run([trainer_d, d_loss], 
                                                            feed_dict={random_input: train_noise, real_data: np_dataset, is_train: True})
                            dis_loss_val += dis_loss_val_inst
                    except tf.errors.OutOfRangeError:
                        pass
                
                
                gen_loss_val = 0
                for g_iter in range(g_iters):
                    sess.run(batch_iterator.initializer, 
                             feed_dict={random_input: train_noise, real_data: np_dataset, is_train: Truerandom_input: train_noise, is_train: True})
                    try:
                        print('g_iter:%i' % g_iter)
                        batch_it = 1
                        while True:
                            print('batch iter:%i' % batch_it)
                            batch_it += 1
                            _, gen_loss_val_inst = sess.run([trainer_g, g_loss], 
                                                            feed_dict={random_input: train_noise, real_data: np_dataset, is_train: True})
                            gen_loss_val += gen_loss_val_inst
                    except tf.errors.OutOfRangeError:
                        pass
                
                print('d_loss:%f, g_loss:%f' % (-dis_loss_val, -gen_loss_val))
                
                if (epoch_iter + 1) % 20 == 0:
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    saver.save(sess, model_path + str(epoch_iter+1))  

                print("--- %s seconds passed... ---" % (time.time() - start_time))
            
            coord.request_stop()
            coord.join(threads)


    def T_test(self, model_path, model_name, test_batch_size, test_path, test_batches):
        with tf.variable_scope('input'):
            random_input = tf.placeholder(tf.float32, shape=[None, self.__random_dim], name='rand_input')
            is_train = tf.placeholder(tf.bool, name='is_train')
        
        fake_batch = self.__model.T_gen(random_input, self.__random_dim, is_train, reuse=self.__reuse_flag)
        
        with tf.Session() as sess:
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
                data_test = sess.run(fake_batch, feed_dict={random_input: sample_noise, is_train: False})
                np.save(test_path + str(test_iter) , data_test)
            
            coord.request_stop()
            coord.join(threads)
