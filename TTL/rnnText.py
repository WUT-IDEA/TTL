# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
@author: Yang
@time: 17-11-13 下午5:20
'''

import utils
import numpy as np
import random

text = utils.read_text(filename='text.txt')
charset = utils.read_text(filename='material/charset.txt')
chrstlen = len(charset) + 1
chrst = {ch: index for index, ch in enumerate(charset)}
chrst_reversed = {index: ch for index, ch in enumerate(charset)}

num = 40000
maxlen = 20
step = 10
sentences = [text[i:i + maxlen] for i in xrange(0, len(text), step)][:num]
random.shuffle(sentences)
del text

# build matrix for sentences
sentnMatrix = np.zeros(shape=(num, maxlen, chrstlen), dtype=np.float)
for i, sntc in enumerate(sentences):
    for j, char in enumerate(sntc):
        sentnMatrix[i, j, chrst[char]] = 1.0

# ------------------#
# define matrix loss
from keras import backend as K


def matrix_loss(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred))


# ------------------#
# build a rnn model to transfer text-type data into a possibility-based float-type matrix
from keras.callbacks import *
from keras.layers import *
from keras.models import *

# build model
# input
input_tensor = Input(shape=(maxlen, chrstlen))

# ------------------#
# a layer of RNN with LSTM block + a full connected network with softmax activation
inner = LSTM(units=100, input_shape=(maxlen, chrstlen), return_sequences=True,
             kernel_initializer='he_normal', name='LSTM_1')(input_tensor)
inner = Dense(units=chrstlen, activation='softmax', kernel_initializer='he_normal', name='fc')(inner)

# output
output_tensor = inner

model = Model(input=input_tensor, output=output_tensor)
model.compile(loss=matrix_loss, optimizer=optimizers.adagrad())

model.summary()

# ------------------#
modelDir = 'model/'
utils.mkdir(modelDir)
modelName = '%s/rnnTex.h5' % (modelDir)
epochs = 10
if utils.dirExit(dirr=modelName):
    model.load_weights(modelName, by_name=True)
else:

    def evaluate(num=100):
        # test rnn generator
        matrix2string = lambda matrix: [''.join([chrst_reversed[np.argmax(line)] for line in mat]) for mat in matrix]
        random.shuffle(sentnMatrix)
        predictMatrix = matrix2string(model.predict(sentnMatrix[:num]))
        acc = [utils.string_diff(string, sentences[j]) for j, string in enumerate(predictMatrix)]
        # print 'accuracy of rnn generator is %s' % (np.mean(acc))
        return np.mean(acc)


    class Evaluate(Callback):
        def __init__(self):
            self.accuracy = np.zeros(shape=(epochs))

        def on_epoch_end(self, epoch, logs):
            print '%s / %s' % (epoch + 1, epochs)
            self.accuracy[epoch] = evaluate()
            model.save(filepath='%s/%s.h5' % (modelDir, epoch + 1))


    evaluator = Evaluate()

    model.fit(x=sentnMatrix, y=sentnMatrix,
              batch_size=1000,
              epochs=epochs,
              callbacks=[EarlyStopping(patience=10), evaluator],
              verbose=1)

import os


def extend_matrix(sourceDir, targetDir, encodingMethod=1):
    utils.mkdir(targetDir)

    print 'extending matrix of %s ...' % (sourceDir)
    index = sorted([fl for fl in os.listdir(sourceDir) if fl.endswith('txt')])
    extendData = [sourceDir + '/' + fl for fl in sorted(index, key=lambda name: int(name.split('.')[0]))]

    widthDict = utils.width_dict(filename='material/width.txt')
    for i, fl in enumerate(extendData):
        tmpMatrix = np.zeros(shape=(1, maxlen, chrstlen), dtype=float)
        sntc = utils.read_text(fl)
        for j, char in enumerate(sntc):
            tmpMatrix[0, j, chrst[char]] = 1.0
        extendMatrix = model.predict(tmpMatrix)[0]
        # print ''.join([chrst_reversed[ch] for ch in [np.argmax(line) for line in extendMatrix]])
        mat = utils.extend_matrix(original=extendMatrix, length=600, string=sntc, widthDict=widthDict, chrst=chrst,
                                  encodingMethod=encodingMethod)
        utils.write_npy(filename='%s/%s.npy' % (targetDir, index[i].split('.')[0]), mat=mat)


# extending
sourceDirs = [
    'dataset/image/acceleration/training',
    'dataset/image/underfitting/training',
]
for sDir in sourceDirs:
    extend_matrix(sourceDir=sDir, targetDir=sDir.replace('training', 'extend'))

del model
