# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
@author: Yang
@time: 17-11-14 下午9:18
'''

import os
import utils
import re

import matplotlib.pyplot as plt
import numpy as np

plt.figure()

# , 'underfitting'
Types = ['acceleration']
for dataType in Types:
    dirs = 'model/%s/image/' % (dataType)
    for subdir in os.listdir(dirs):
        logFile = dirs + subdir + '/log.txt'
        logLines = utils.read_text(filename=logFile).strip().split('\n')
        logLines = [re.findall(r'\d+\.?\d*', line) for line in logLines]
        logLines = np.asarray(logLines, dtype=np.float)
        plt.plot(logLines[:, -1], label=logFile)

plt.legend(loc='best')
plt.grid()
plt.show()
