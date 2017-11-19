# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
@author: Yang
@time: 17-11-14 下午2:57
'''

'''
extractor text lines from a text file

acceleration dataset:
training data: 2000 | test data: 100

underfitting dataset:
training data: 500 | test data: 1000
'''
import Extractor

extractor = Extractor.Extractor(filename='text.txt')

extractor.load_data(trainingNum=2000, testNum=100)
extractor.save_data(target='dataset/text/acceleration/')

extractor.load_data(trainingNum=500, testNum=1000)
extractor.save_data(target='dataset/text/underfitting/')

extractor.charset()

import os

'''
imageGen.py
generate image for OCR
'''
os.system('python imageGen.py')

'''
rnnText.py
1.training text generator
2.generate extended matrix

acceleration dataset:
training data: 2000 <- extending

underfitting dataset:
training data: 500 <- extending
'''
os.system('python rnnText.py')

'''
rnnExtendedMaxtrix.py
training RNN on the collection composed of images and extended matrix

acceleration dataset:
training data (extended matrix): 2000 <- training

underfitting dataset:
training data (extended matrix): 500 <- training
'''
os.system('python rnnExtendedMaxtrix.py')

'''
trainImage.py
transfer RNN pre-trained extended matrix to normal OCR model

--------
training on
acceleration dataset:
training data: 2000 <- training

underfitting dataset:
training data: 500 <- training

--------
test on
acceleration dataset:
training data: 100 <- test

underfitting dataset:
training data: 1000 <- test

'''

from trainImage import *

# acceleration

# attention: Due to SGD, deep learning models are more likely to fall into local optimum, but our approach won't

# deep learning, 0 for no transfer learning
basis = 0
training(traingPath='dataset/image/acceleration/training/', \
         modelDir='model/acceleration/image/%s/' % (basis), \
         basis=basis, \
         basisModel='model/acceleration/extend/%s.h5' % (basis), \
         epochs=50, \
         testPath='dataset/image/acceleration/test/')

# # deep learning, 10 for transfer learning based on 10 times pre-trainings
# basis = 1, 3, 5, 10
# training(traingPath='dataset/image/acceleration/training/', \
#          modelDir='model/acceleration/image/%s/' % (basis), \
#          basis=basis, \
#          basisModel='model/acceleration/extend/%s.h5' % (basis), \
#          epochs=50, \
#          testPath='dataset/image/acceleration/test/')


# underfitting
#
# basis = 0
# training(traingPath='dataset/image/underfitting/training/', \
#          modelDir='model/underfitting/image/%s/' % (basis), \
#          basis=basis, \
#          basisModel='model/underfitting/extend/%s.h5' % (basis), \
#          epochs=150 - basis, \
#          testPath='dataset/image/underfitting/test/')
#
# basis = 50
# training(traingPath='dataset/image/underfitting/training/', \
#          modelDir='model/underfitting/image/%s/' % (basis), \
#          basis=basis, \
#          basisModel='model/underfitting/extend/%s.h5' % (basis), \
#          epochs=150 - basis, \
#          testPath='dataset/image/underfitting/test/')
