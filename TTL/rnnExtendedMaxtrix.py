# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
@author: Yang
@time: 17-11-13 下午11:09
'''

import utils
from PIL import Image

charset = utils.read_text('material/charset.txt')
chrstLen = len(charset) + 1
chrst = {char: i for i, char in enumerate(charset)}
chrstReversed = {i: char for i, char in enumerate(charset)}

import os

maxlen = 20
width, height, n_len, n_class = 600, 48, maxlen, chrstLen

# ------------------#
# define a matrix loss
# loss function = sum(
#       (a-b)^2     for a in matrix A, for b in matrix B
# )

from keras import backend as K


def matrix_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def matrix_loss_sum(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred))


# ---------------rmsprop---#
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

# build
model = Model(input=input_tensor, output=output_tensor)
model.compile(loss=matrix_loss_sum, optimizer=optimizers.adam())

model.summary()

# ------------------#
import time
import numpy as np


# training
def training(dataType, extendType, epochs):
    imgPath = 'dataset/image/%s/training' % (dataType)
    extendPath = 'dataset/image/%s/%s' % (dataType, extendType)
    fileNum = len(os.listdir(imgPath)) / 2

    modelDir = 'model/%s/%s/' % (dataType, extendType)
    utils.mkdir(dirr=modelDir)

    def gen(batch_size=50):
        current_index = batch_size
        index = ['/01%04d' % i for i in xrange(batch_size)]
        while 1:
            X = np.asanyarray([utils.read_npy(imgPath + i + '.npy')
                               for i in index], dtype=np.float)
            y = np.asanyarray([utils.read_npy(extendPath + i + '.npy')
                               for i in index], dtype=np.float)
            if current_index + batch_size > fileNum:
                index = range(current_index, fileNum) + range(0, batch_size + current_index - fileNum)
                index = ['/01%04d' % i for i in index]
                current_index = batch_size + current_index - fileNum
            else:
                index = ['/01%04d' % i for i in range(current_index, current_index + batch_size)]
                current_index += batch_size
            yield [X, y]

    class Evaluate(Callback):
        def __init__(self):
            pass

        def on_epoch_end(self, epoch, logs):
            print '%s / %s' % (epoch + 1, epochs)
            model.save(filepath='%s/%s.h5' % (modelDir, epoch + 1))

    evaluator = Evaluate()

    batch_size = 100
    model.save(filepath='%s/%s.h5' % (modelDir, 0))
    model.fit_generator(gen(batch_size=batch_size), \
                        steps_per_epoch=fileNum / batch_size, \
                        epochs=epochs,
                        callbacks=[EarlyStopping(patience=10), evaluator],
                        verbose=1)


training(dataType='acceleration', extendType='extend', epochs=10)
training(dataType='underfitting', extendType='extend', epochs=50)

del model
