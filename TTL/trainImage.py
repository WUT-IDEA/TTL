# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
@author: Yang
@time: 17-11-14 上午1:42
'''

import utils
from PIL import Image

charset = utils.read_text('material/charset.txt')
chrstLen = len(charset) + 1
chrst = {char: i for i, char in enumerate(charset)}
chrstReversed = {i: char for i, char in enumerate(charset)}

import os

maxlen = 20
width, height, n_len, n_class = 500, 48, maxlen, chrstLen

# ------------------#
# build the structure of biRNN network
from keras.callbacks import *
from keras.layers import *
from keras.models import *

rnn_size = 100

input_tensor = Input(shape=(width, height))

lstm_1 = LSTM(units=rnn_size, input_shape=(maxlen, chrstLen), return_sequences=True, kernel_initializer='he_normal',
              name='LSTM_1')(input_tensor)
lstm_1b = LSTM(units=rnn_size, input_shape=(maxlen, chrstLen), return_sequences=True, kernel_initializer='he_normal',
               name='LSTM_1b', go_backwards=True)(input_tensor)
inner = concatenate([lstm_1, lstm_1b], axis=-1)

inner = Dense(n_class, kernel_initializer='he_normal', activation='softmax', name='fc')(inner)
output_tensor = inner
base_model = Model(input=input_tensor, output=output_tensor)


# ------------------#
# CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([output_tensor, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizers.rmsprop())

model.summary()

# ------------------#
# training
import time
import numpy as np

# if GPU and time allowed, you could choose np.float16 or np.float32 as image data type
# in here, we use np.unit8 for saving time
NP_DTYPE = np.uint8


# NP_DTYPE = np.float16



def training(traingPath, modelDir, basis, basisModel, epochs=None, testPath=None):
    trainingFileNum = len(os.listdir(traingPath)) / 2

    utils.mkdir(modelDir)

    def gen(batch_size=50):
        current_index = batch_size
        index = ['/01%04d' % i for i in xrange(batch_size)]
        while 1:
            X = np.asanyarray([utils.read_npy(traingPath + i + '.npy')
                               for i in index], dtype=NP_DTYPE)
            str2index = lambda line: [chrst[i] for i in line]
            y = np.asanyarray([str2index(line) for line in
                               [utils.read_text(traingPath + i + '.txt') for i in index]],
                              dtype=NP_DTYPE)
            if current_index + batch_size > trainingFileNum:
                index = range(current_index, trainingFileNum) + range(0, batch_size + current_index - trainingFileNum)
                index = ['/01%04d' % i for i in index]
                current_index = batch_size + current_index - trainingFileNum
            else:
                index = ['/01%04d' % i for i in range(current_index, current_index + batch_size)]
                current_index += batch_size
            yield [X, y, np.ones(batch_size) * int(width - 2), np.ones(batch_size) * n_len], np.ones(batch_size)

    def gen_for_test(start, batch_size=50):
        index = ['/01%04d' % i for i in xrange(start, start + batch_size)]
        while 1:
            X = np.asanyarray([utils.read_npy(testPath + i + '.npy')
                               for i in index], dtype=NP_DTYPE)
            y = [utils.read_text(testPath + i + '.txt') for i in index]
            yield X, y

    # ------------------#
    # some function
    drop = lambda line: [i for i in line if not i == -1]
    decoder = lambda mat: [''.join([chrstReversed[i] for i in drop(line)]) for line in mat]

    def evaluate(path=testPath):
        testFileNum = len(os.listdir(path)) / 2
        batch_size = 100
        acc_arr = [0 for i in xrange(testFileNum / batch_size)]

        for j in xrange(testFileNum / batch_size):
            start = j * batch_size
            X_test, y_test = next(gen_for_test(start, batch_size))
            y_pred = base_model.predict(X_test)
            shape = y_pred.shape
            out = K.get_value(
                K.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1], greedy=False)
                [0][0])
            out = decoder(out)
            # single
            out = [utils.string_diff(out[i], y_test[i]) for i in xrange(len(out))]
            acc_arr[j] = np.mean(out)
        return np.mean(acc_arr)

    class Evaluate(Callback):
        def __init__(self):
            self.accuracy = np.zeros(shape=(epochs))

        def on_epoch_end(self, epoch, logs):
            print '%s / %s' % (epoch + 1, epochs)
            self.accuracy[epoch] = evaluate()
            model.save(filepath='%s/%s.h5' % (modelDir, epoch + 1))

    evaluator = Evaluate()

    if basis == 0:
        pass
    else:
        model.load_weights(basisModel, by_name=True)
        print 'load %s' % (basisModel)

    batch_size = 100
    model.save(filepath='%s/%s.h5' % (modelDir, 0))
    model.fit_generator(gen(batch_size=batch_size), \
                        steps_per_epoch=trainingFileNum / batch_size, \
                        callbacks=[EarlyStopping(patience=10), evaluator],
                        epochs=epochs,
                        verbose=1)
