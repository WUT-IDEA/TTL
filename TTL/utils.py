# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
@author: Yang
@time: 17-11-13 下午3:27
'''


def read_text(filename):
    with open(name=filename, mode='rb') as stream:
        return stream.read()


def write_text(filename, obj, mode='wb'):
    with open(name=filename, mode=mode) as stream:
        stream.write(obj)


import numpy as np


def write_npy(filename, mat):
    try:
        np.save(file=filename, arr=mat)
        return True
    except NameError, err:
        print err.message
        return False


def read_npy(filename):
    return np.load(file=filename)


import os


def mkdir(dirr):
    try:
        dirs = [unit for unit in dirr.split('/') if len(unit) > 0]
        for i in xrange(len(dirs)):
            new_dir = '/'.join(dirs[:i + 1])
            if os.path.exists(new_dir):
                pass
            else:
                os.mkdir(new_dir)
    except EnvironmentError, err:
        print err.message


# mkdir(dirr='dataset/text/acceleration/')


def dirExit(dirr):
    return os.path.exists(dirr)


def width_dict(filename):
    lines = read_text(filename=filename).split('\n')
    return {ln[0]: int(ln[2:]) for ln in lines}


def extend_matrix(original, length, string, widthDict, chrst, encodingMethod=1):
    assert original.shape[0] * 2 < length

    newMatrix = np.zeros(shape=(length, original.shape[1]), dtype=float)
    for i in xrange(length):
        newMatrix[i, -1] = 1.0

    times = [widthDict[char] for char in string]

    # 1. add at the beginning and ending
    # 2. interim between characters is 2
    # 3. the default is black
    # assuming picture is:
    #       pad + (interim + char) * 20 + dark area + pad
    pad = 16
    interim = 2

    pointer = pad + interim
    for i, tm in enumerate(times):
        for j in xrange(pointer, pointer + tm):
            newMatrix[j, :] = original[i, :]
        pointer += tm + interim
    assert newMatrix.shape[0] == length
    return newMatrix


import editdistance


def string_diff(str, main_str):
    dis = editdistance.eval(str, main_str)
    acc = min(1.0 * dis / len(main_str) * 1.0, 1.0)
    return 1 - acc
