import os
import numpy as np
import tensorflow as tf
import pickle
import pickle

from load_data import loading_data
from img_net import img_net_strucuture
from txt_net import txt_net_strucuture
from utils.calc_hammingranking import calc_map, calc_prc


import datetime
from time import *

# environmental setting: setting the following parameters based on your experimental environment.
select_gpu = '0'
per_process_gpu_memory_fraction = 0.9

Dataset_name = 'FashionVC'
# data parameters
# DATA_DIR = './DataSet/'
DATA_DIR = 'D:/sunyu/dataset/FashionVC/'

TRAINING_SIZE = 16862

QUERY_SIZE = 3000
DATABASE_SIZE = 16862
num_class1 = 8
num_class2 = 27

# hyper-parameters
MAX_ITER =100
num_class = num_class1 + num_class2
alpha = 0.4
beta = 0.6
gamma = 10
eta = 1e-2
bit = 32
# bits = [16, 32, 64, 128]
num_sample = 2000

filename = './log/result_' + datetime.datetime.now().__str__() + '_' + str(bit) + 'bits_' + str(Dataset_name) + '.pkl'
# filename = filename.replace(':', '_')  # win10√ linux×


def train_img_net(image_input, cur_f_batch, var, ph, train_x, train_L, lr, train_step_x, mean_pixel_, anchor_ind):
    F = var['F']
    batch_size = var['batch_size']
    num_train = train_x.shape[0]
    for iter in range((int)(num_train / batch_size)):
        index = np.random.permutation(num_train)
        ind = index[0: batch_size]
        sample_L = train_L[ind, :]
        image = train_x[ind, :, :, :].astype(np.float64)
        image = image - mean_pixel_.astype(np.float64)

        cur_f = cur_f_batch.eval(feed_dict={image_input: image})
        F[:, ind] = cur_f

        train_step_x.run(
            feed_dict={ph['L1']: sample_L[:, 0:num_class1], ph['L2']: sample_L[:, num_class1:num_class],
                       ph['b_batch']: (var['B'][:,anchor_ind])[:, ind],
                       ph['y1']: var['Y1'], ph['y2']: var['Y2'], image_input: image})

    return F


def train_txt_net(text_input, cur_g_batch, var, ph, train_y, train_L, lr, train_step_y, anchor_ind):
    G = var['G']
    batch_size = var['batch_size']
    num_train = train_y.shape[0]
    for iter in range((int)(num_train / batch_size)):
        index = np.random.permutation(num_train)
        ind = index[0: batch_size]
        sample_L = train_L[ind, :]
        text = train_y[ind, :].astype(np.float32)
        text = text.reshape([text.shape[0], 1, text.shape[1], 1])

        cur_g = cur_g_batch.eval(feed_dict={text_input: text})
        G[:, ind] = cur_g

        train_step_y.run(
            feed_dict={ph['L1']: sample_L[:, 0:num_class1], ph['L2']: sample_L[:, num_class1:num_class],
                       ph['b_batch']: (var['B'][:,anchor_ind])[:, ind],
                       ph['y1']: var['Y1'], ph['y2']: var['Y2'], text_input: text})
    return G


def split_data(images, tags, labels):
    X = {}
    X['query'] = images[0:QUERY_SIZE, :, :, :]
    X['train'] = images[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE, :, :, :]
    X['retrieval'] = images[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE, :, :, :]

    Y = {}
    Y['query'] = tags[0:QUERY_SIZE, :]
    Y['train'] = tags[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE, :]
    Y['retrieval'] = tags[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE, :]

    L = {}
    L['query'] = labels[0:QUERY_SIZE, :]
    L['train'] = labels[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE, :]
    L['retrieval'] = labels[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE, :]

    return X, Y, L


def calc_loss(B, F, G, bc1, bc2, Sim12, L1, L2, alpha, beta, gamma, eta, anchor_ind):
    term1 = alpha * np.sum(np.power((bit * L1 - np.matmul(np.transpose(B), bc1)), 2) + np.power(
        (bit * L1 - np.matmul(np.transpose(B), bc1)), 2))
    term2 = beta * np.sum(np.power((bit * L2 - np.matmul(np.transpose(B), bc2)), 2) + np.power(
        (bit * L2 - np.matmul(np.transpose(B), bc2)), 2))
    term3 = eta * np.sum(np.power((bit * Sim12 - np.matmul(np.transpose(bc1), bc2)), 2))
    term4 = gamma * np.sum(np.power((B[:, anchor_ind] - F), 2) + np.power((B[:, anchor_ind] - G), 2))
    loss = term1 + term2 + term3 + term4
    return loss


def generate_image_code(image_input, cur_f_batch, X, bit, mean_pixel):
    batch_size = 128
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter in range((int)(num_data / batch_size) + 1):
        if (iter * batch_size == num_data):
            break;
        ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
        mean_pixel_ = np.repeat(mean_pixel[:, :, :, np.newaxis], len(ind), axis=3)
        image = X[ind, :, :, :].astype(np.float32) - mean_pixel_.astype(np.float32).transpose(3, 0, 1, 2)

        cur_f = cur_f_batch.eval(feed_dict={image_input: image})
        B[ind, :] = cur_f.transpose()
    B = np.sign(B)
    return B


def generate_text_code(text_input, cur_g_batch, Y, bit):
    batch_size = 128
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter in range((int)(num_data / batch_size) + 1):
        if (iter * batch_size == num_data):
            break;
        ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
        text = Y[ind, :].astype(np.float32)
        text = text.reshape([text.shape[0], 1, text.shape[1], 1])
        cur_g = cur_g_batch.eval(feed_dict={text_input: text})
        B[ind, :] = cur_g.transpose()
    B = np.sign(B)
    return B



if __name__ == '__main__':
        filename = './log/SHDCH_a/NEW/param/result_' + datetime.datetime.now().__str__() + '_' + str(bit) + 'bits_' + str(
            Dataset_name) + '_alpha=' + str(alpha) + '_beta=' + str(beta) + '_gamma=' + str(gamma) + '_eta=' + str(
            eta) +  '_sample='+ str(num_sample) + '.pkl'
        filename = filename.replace(':', '_')  # win10√ linux×
        print(filename)
        print('loading...')
        images, tags, labels = loading_data(DATA_DIR)
        print(images.shape)
        print(tags.shape)
        print(labels.shape)
        S12 = np.zeros([num_class1, num_class2])
        index1 = 0
        index2 = 0
        for i in range(0, len(labels)):
            for j in range(0, len(labels[0])):
                if (labels[i][j] == 1 and j < num_class1):
                    index1 = j
                elif labels[i][j] == 1 and j >= num_class1:
                    index2 = j
            S12[index1][index2 - num_class1] = 1

        ydim = tags.shape[1]

        X, Y, L = split_data(images, tags, labels)
        print('...loading and splitting data finish')

        # tf2.0
        gpuconfig = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))

        os.environ["CUDA_VISIBLE_DEVICES"] = select_gpu
        batch_size = 128
        with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session(config=gpuconfig) as sess:
            # construct image network
            image_input = tf.compat.v1.placeholder(tf.float32, (None,) + (224, 224, 3))
            net, _meanpix = img_net_strucuture(image_input, bit)
            mean_pixel_ = np.repeat(_meanpix[:, :, :, np.newaxis], batch_size, axis=3).transpose(3, 0, 1, 2)
            cur_f_batch = tf.transpose(net['fc8'])
            # construct text network
            text_input = tf.compat.v1.placeholder(tf.float32, (None,) + (1, ydim, 1))
            cur_g_batch = txt_net_strucuture(text_input, ydim, bit)

            # training algorithm
            train_L = L['train']
            train_L1 = train_L[:, 0:num_class1]
            train_L2 = train_L[:, num_class1:num_class]
            train_x = X['train']
            train_y = Y['train']

            query_L = L['query']
            query_x = X['query']
            query_y = Y['query']

            retrieval_L = L['retrieval']
            retrieval_x = X['retrieval']
            retrieval_y = Y['retrieval']
            num_train = train_x.shape[0]


            var = {}
            lr_img = 0.0001
            lr_txt = 0.01

            var['lr_img'] = lr_img
            var['lr_txt'] = lr_txt
            var['batch_size'] = batch_size

            var['F'] = np.random.randn(bit, num_sample)
            var['G'] = np.random.randn(bit, num_sample)

            # real-value
            var['Y1'] = np.random.randn(bit, num_class1)
            var['Y2'] = np.random.randn(bit, num_class2)
            var['B'] = np.sign(np.random.randn(bit, num_train))

            ph = {}
            ph['lr_img'] = tf.compat.v1.placeholder('float32', (), name='lr_img')
            ph['lr_txt'] = tf.compat.v1.placeholder('float32', (), name='lr_txt')


            ph['L1'] = tf.compat.v1.placeholder('float32', [batch_size, num_class1], name='L1')
            ph['L2'] = tf.compat.v1.placeholder('float32', [batch_size, num_class2], name='L2')

            ph['b_batch'] = tf.compat.v1.placeholder('float32', [bit, batch_size], name='b_batch')
            ph['y1'] = tf.compat.v1.placeholder('float32', [bit, num_class1], name='y1')
            ph['y2'] = tf.compat.v1.placeholder('float32', [bit, num_class2], name='y2')

            quantization_x = tf.reduce_sum(tf.pow((ph['b_batch'] - cur_f_batch), 2))
            loss_x = tf.compat.v1.div(gamma * quantization_x, float(num_train))

            quantization_y = tf.reduce_sum(tf.pow((ph['b_batch'] - cur_g_batch), 2))
            loss_y = tf.compat.v1.div(gamma * quantization_y, float(num_train))

            optimizer_x = tf.compat.v1.train.AdamOptimizer(var['lr_img'])
            optimizer_y = tf.compat.v1.train.AdamOptimizer(var['lr_txt'])

            gradient_x = optimizer_x.compute_gradients(loss_x)
            gradient_y = optimizer_y.compute_gradients(loss_y)
            train_step_x = optimizer_x.apply_gradients(gradient_x)
            train_step_y = optimizer_y.apply_gradients(gradient_y)
            sess.run(tf.compat.v1.global_variables_initializer())

            result = {}
            result['loss'] = []
            result['imapi2t'] = []
            result['imapt2i'] = []
            result['alpha'] = alpha
            result['beta'] = beta
            result['gamma'] = gamma
            result['eta'] = eta
            result['bit'] = bit
            result['train_time'] = []

            print('...training procedure starts')
            # start_time = time()
            train_time = []
            for epoch in range(MAX_ITER):

                anchor_index = np.random.permutation(num_train)
                anchor_ind = anchor_index[0: num_sample]
                anchor_train_x = train_x[anchor_ind, :, :, :]
                anchor_train_y = train_y[anchor_ind, :]
                anchor_train_L = train_L[anchor_ind, :]

                start_time = time()
                lr_img = var['lr_img']
                lr_txt = var['lr_txt']

                # update F
                var['F'] = train_img_net(image_input, cur_f_batch, var, ph, anchor_train_x, anchor_train_L, lr_img, train_step_x, mean_pixel_, anchor_ind)

                # update G
                var['G'] = train_txt_net(text_input, cur_g_batch, var, ph, anchor_train_y, anchor_train_L, lr_txt, train_step_y,anchor_ind)

                # update B
                F_all = np.zeros((bit, num_train))
                F_all[:, anchor_ind] = var['F']
                G_all = np.zeros((bit, num_train))
                G_all[:, anchor_ind] = var['G']
                Q = alpha * bit * np.matmul(var['Y1'], np.transpose(train_L1)) + beta * bit * np.matmul(var['Y2'], np.transpose(train_L2)) + gamma * F_all + gamma * G_all
                for i in range(3):
                    B = var['B']
                    Y1 = var['Y1']
                    Y2 = var['Y2']
                    for k in range(bit):
                        sel_ind = np.setdiff1d([ii for ii in range(bit)], k)
                        Y1_ = Y1[sel_ind, :]
                        y1k = np.transpose(Y1[k, :])
                        Y2_ = Y2[sel_ind, :]
                        y2k = np.transpose(Y2[k, :])
                        B_ = B[sel_ind, :]
                        bk = np.transpose(B[k, :])


                        b = np.sign(np.transpose(Q[k,:]) - alpha * B_.transpose().dot(Y1_.dot(y1k)) - beta * B_.transpose().dot(Y2_.dot(y2k)))
                        var['B'][k, :] = np.transpose(b)
                    if np.linalg.norm(var['B']-B) < 1e-6 * np.linalg.norm(B):
                        break

                # update Y
                Q1 = bit * (alpha * (np.matmul(var['B'], train_L1)) +
                            eta * np.matmul(var['Y2'], np.transpose(S12)))
                Q2 = bit * (beta * (np.matmul(var['B'], train_L2)) +
                            eta * np.matmul(var['Y1'], S12))
                Y1 = var['Y1']
                Y2 = var['Y2']
                var['Y1'] = np.linalg.inv(alpha * var['B'] @ var['B'].transpose() + eta * var['Y2'] @ var['Y2'].transpose()) @  Q1
                var['Y2'] = np.linalg.inv(beta * var['B'] @ var['B'].transpose()  + eta * var['Y1'] @ var['Y1'].transpose()) @ Q2

                end_time = time()
                train_time.append((end_time - start_time))
                result['train_time'].append((end_time - start_time))

                # calculate loss
                loss_ = calc_loss(var['B'], var['F'], var['G'], var['Y1'], var['Y2'], S12, train_L1, train_L2, alpha,
                                  beta, gamma, eta, anchor_ind)

                print('[epoch: %3d], [loss: %3.3f]' % (epoch + 1, loss_))

                result['loss'].append(loss_)
                if (epoch+1)  == 1:
                    qBX = generate_image_code(image_input, cur_f_batch, query_x, bit, _meanpix)
                    qBY = generate_text_code(text_input, cur_g_batch, query_y, bit)
                    rBX = generate_image_code(image_input, cur_f_batch, retrieval_x, bit, _meanpix)
                    rBY = generate_text_code(text_input, cur_g_batch, retrieval_y, bit)


                    mapi2t = calc_map(qBX, rBY, query_L, retrieval_L, num_class1)
                    mapt2i = calc_map(qBY, rBX, query_L, retrieval_L, num_class1)

                    result['imapi2t'].append(mapi2t)
                    result['imapt2i'].append(mapt2i)

                    print('[time: %s], [epoch: %3d], [loss: %3.3f], [map(i->t): \033[1;32;40m%3.3f\033[0m], [map(t->i): \033[1;32;40m%3.3f\033[0m], [train_time:%3.3f]' % (datetime.datetime.now(), epoch + 1, loss_,
                        mapi2t, mapt2i,(end_time - start_time)))
                    fp = open(filename, 'wb')
                    pickle.dump(result, fp)
                if (epoch+1) % 10 == 0:
                    qBX = generate_image_code(image_input, cur_f_batch, query_x, bit, _meanpix)
                    qBY = generate_text_code(text_input, cur_g_batch, query_y, bit)
                    rBX = generate_image_code(image_input, cur_f_batch, retrieval_x, bit, _meanpix)
                    rBY = generate_text_code(text_input, cur_g_batch, retrieval_y, bit)

                    mapi2t = calc_map(qBX, rBY, query_L, retrieval_L, num_class1)
                    mapt2i = calc_map(qBY, rBX, query_L, retrieval_L, num_class1)

                    result['imapi2t'].append(mapi2t)
                    result['imapt2i'].append(mapt2i)

                    print('[time: %s], [epoch: %3d], [loss: %3.3f], [map(i->t): \033[1;32;40m%3.3f\033[0m], [map(t->i): \033[1;32;40m%3.3f\033[0m], [train_time:%3.3f]' % (datetime.datetime.now(), epoch + 1, loss_,
                        mapi2t, mapt2i,(end_time - start_time)))

                    fp = open(filename, 'wb')
                    pickle.dump(result, fp)


            run_time = np.sum(train_time)
            average_time = np.average(result['train_time'])
            result['average_time'] = average_time
            result['run_time'] = run_time

            print('...training procedure finish')
            qBX = generate_image_code(image_input, cur_f_batch, query_x, bit, _meanpix)
            qBY = generate_text_code(text_input, cur_g_batch, query_y, bit)
            rBX = generate_image_code(image_input, cur_f_batch, retrieval_x, bit, _meanpix)
            rBY = generate_text_code(text_input, cur_g_batch, retrieval_y, bit)


            mapi2t = calc_map(qBX, rBY, query_L, retrieval_L, num_class1)
            mapt2i = calc_map(qBY, rBX, query_L, retrieval_L, num_class1)

            print(
                '[time: %s], [map(i->t): \033[1;32;40m%3.3f\033[0m], [map(t->i): \033[1;32;40m%3.3f\033[0m], [total_time:%3.3f], [average_time:%3.3f]' % (
                datetime.datetime.now(),
                mapi2t, mapt2i, run_time, average_time))

            result['mapi2t'] = mapi2t
            result['mapt2i'] = mapt2i
            result['lr_img'] = lr_img
            result['lr_txt'] = lr_txt

            print('result has saved...')

            fp = open(filename, 'wb')
            pickle.dump(result, fp)

            fp.close()
