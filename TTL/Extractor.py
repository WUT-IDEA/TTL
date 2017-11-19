# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
@author: Yang
@time: 17-11-13 下午3:30
'''

import utils
import random


class Extractor:
    def __init__(self, filename):
        self.filename = filename

        self.maxlen = 20
        self.step = 5

    def load_data(self, trainingNum=None, testNum=None):
        self.trainingNum = trainingNum
        self.testNum = testNum

        text = utils.read_text(filename=self.filename)
        sentences = [text[i:i + self.maxlen]
                     for i in xrange(0, len(text), self.step)]
        sentences = [sntc for sntc in sentences if len(sntc) == self.maxlen]
        random.shuffle(sentences)
        if trainingNum == None:
            pass
        else:
            self.traingData = sentences[:self.trainingNum]

        if testNum == None:
            pass
        else:
            self.testData = sentences[-self.testNum:]

        del sentences

    def save_data(self, target):
        utils.mkdir(dirr=target)
        utils.write_text(filename=target + '/training.txt', obj='\n'.join(self.traingData))
        utils.write_text(filename=target + '/test.txt', obj='\n'.join(self.testData))

    def charset(self):
        text = utils.read_text(filename=self.filename)
        charset = sorted(list(set(text)), key=ord)
        utils.write_text(filename='material/charset.txt', obj=''.join(charset))
