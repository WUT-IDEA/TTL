# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
@author: Yang
@time: 17-11-13 下午4:52
'''

import utils
import codecs
import random as pyrandom
import traceback
import PIL
from PIL import Image
from PIL import ImageFont, ImageDraw
from scipy.ndimage import filters, measurements, interpolation
from pylab import *
import os

font = 'material/Times.ttf'

sizes = [i for i in xrange(40, 70 + 1)]

LENGTH = 500
# degradations = lo(default), med, hi
degradations = 'med'
if degradations == "lo":
    # sigma +/-   threshold +/-
    deglist = """
    0.5 0.0   0.5 0.0
    """
elif degradations == "med":
    deglist = """
    0.5 0.0   0.5 0.05
    1.0 0.3   0.4 0.05
    1.0 0.3   0.5 0.05
    1.0 0.3   0.6 0.05
    """
elif degradations == "hi":
    deglist = """
    0.5 0.0   0.5 0.0
    1.0 0.3   0.4 0.1
    1.0 0.3   0.5 0.1
    1.0 0.3   0.6 0.1
    1.3 0.3   0.4 0.1
    1.3 0.3   0.5 0.1
    1.3 0.3   0.6 0.1
    """

degradations = []
for deg in deglist.split("\n"):
    deg = deg.strip()
    if deg == "": continue
    deg = [float(x) for x in deg.split()]
    degradations.append(deg)


def rgeometry(image, eps=0.03, delta=0.3):
    m = array([[1 + eps * pyrandom.random(), 0.0], [eps * pyrandom.random(), 1.0 + eps * pyrandom.random()]])
    w, h = image.shape
    c = array([w / 2.0, h / 2])
    d = c - dot(m, c) + array([pyrandom.random() * delta, pyrandom.random() * delta])
    return interpolation.affine_transform(image, m, offset=d, order=1, mode='constant', cval=image[0, 0])


def rdistort(image, distort=3.0, dsigma=10.0, cval=0):
    h, w = image.shape
    hs = np.random.randn(h, w)
    ws = np.random.randn(h, w)
    hs = filters.gaussian_filter(hs, dsigma)
    ws = filters.gaussian_filter(ws, dsigma)
    hs *= distort / amax(hs)
    ws *= distort / amax(ws)

    def f(p):
        return (p[0] + hs[p[0], p[1]], p[1] + ws[p[0], p[1]])

    return interpolation.geometric_transform(image, f, output_shape=(h, w),
                                             order=1, mode='constant', cval=cval)


def bounding_box(a):
    a = array(a > 0, 'i')
    l = measurements.find_objects(a)
    ys, xs = l[0]
    return (0, 0, 0, 0)
    # y0,x0,y1,x1
    return (ys.start, xs.start, ys.stop, xs.stop)


def crop(image, pad=1):
    [[r, c]] = measurements.find_objects(array(image == 0, 'i'))
    r0 = r.start
    r1 = r.stop
    c0 = c.start
    c1 = c.stop
    image = image[r0 - pad:r1 + pad, c0 - pad:c1 + pad]
    return image


last_font = None
last_size = None
last_fontfile = None


def genline(text, fontfile=None, size=36, sigma=0.5, threshold=0.5):
    global image, draw, last_font, last_fontfile
    if last_fontfile != fontfile:
        last_font = ImageFont.truetype(fontfile, size)
        last_fontfile = fontfile
    font = last_font
    image = Image.new("L", (6000, 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 6000, 6000), fill="white")
    draw.text((250, 20), text, fill="black", font=font)

    # Image.fromarray(a).show()
    a = asarray(image, 'f')
    a = a * 1.0 / amax(a)
    # Image.fromarray(a,mode='I').show()
    if sigma > 0.0:
        a = filters.gaussian_filter(a, sigma)
    a += clip(np.random.rand(*a.shape) * 0.2, -0.25, 0.25)
    a = rgeometry(a)
    a = array(a > threshold, 'f')
    a = crop(a, pad=3)
    # FIXME add grid warping here
    # clf(); ion(); gray(); imshow(a); ginput(1,0.1)
    del draw
    del image
    return a


def clear_dir(dir):
    os.system("rm -rf " + dir)
    os.mkdir(dir)


distort = -1.0
dsigma = 20.0

rangee, smoothness, extra = (4, 1.0, 0.3)
MAX_VALUE = 255.0
newaxis = None
TARGET_HEIGHT = 48


def image_size(fname):
    return PIL.Image.open(fname).size


def open_image(mtrx, askewed=True):
    # img = PIL.Image.open(fname)
    # mtrx = np.fromstring(img.tobytes(), 'B') / MAX_VALUE
    # img.size[1], img.size[0]

    # get askewed parameter
    def measure(matrix):
        matrix = 1.0 - matrix
        h, w = matrix.shape
        smoothed = filters.gaussian_filter(matrix, (h * 0.5, h * smoothness), mode='constant')
        smoothed += 0.001 * filters.uniform_filter(smoothed, (h * 0.5, w), mode='constant')
        mtrx = np.argmax(smoothed, axis=0)
        mtrx = filters.gaussian_filter(mtrx, h * extra)
        center = np.array(mtrx, 'i')
        deltas = abs(np.arange(h)[:, newaxis] - center[newaxis, :])
        mad = np.mean(deltas[matrix != 0])
        r = int(1 + 4 * mad)
        return center, r

    def dewarp(img, center, r, cval=0, dtype=np.dtype('f')):
        h, w = img.shape
        padded = np.vstack([cval * np.ones((h, w)), img, cval * np.ones((h, w))])
        center += h
        dewarped = [padded[center[i] - r:center[i] + r, i] for i in range(w)]
        dewarped = np.array(dewarped, dtype=dtype).T
        return dewarped

    def scale_to_h(img, target_height, order=1, dtype=np.dtype('f'), cval=0):
        h, w = img.shape
        scale = target_height * 1.0 / h
        target_width = int(scale * w)
        output = interpolation.affine_transform(1.0 * img, np.eye(2) / scale, order=order,
                                                output_shape=(target_height, target_width),
                                                mode='constant', cval=cval)
        output = np.array(output, dtype=dtype)
        return output

    def normalize(img, center, r, order=1, dtype=np.dtype('f'), cval=0):
        dewarped = dewarp(img, center, r, cval=cval, dtype=dtype)
        h, w = dewarped.shape
        scaled = scale_to_h(dewarped, TARGET_HEIGHT, order=order, dtype=dtype, cval=cval)
        return scaled

    if askewed == True:
        center, r = measure(mtrx)
        mtrx = normalize(mtrx, center, r, cval=np.amax(mtrx))

    def prepare_line(mtrx):
        mtrx = 1 - mtrx.T
        pad = 16
        w = mtrx.shape[-1]
        # mtrx = np.concatenate((np.zeros((pad, w)), mtrx))
        mtrx = np.vstack([np.zeros((pad, w)), mtrx, np.zeros((pad, w))])
        return mtrx

    return prepare_line(mtrx)


def generate(index, line, path, conv_flag=False):
    while 1:
        (sigma, ssigma, threshold, sthreshold) = pyrandom.choice(degradations)
        sigma += (2 * pyrandom.random() - 1) * ssigma
        threshold += (2 * pyrandom.random() - 1) * sthreshold
        size = pyrandom.choice(sizes)
        try:
            image = genline(text=line, fontfile=font,
                            size=size, sigma=sigma, threshold=threshold)
            # print(image.shape)
        except:
            traceback.print_exc()
            return -1
        if amin(image.shape) < 10: return -1
        if amax(image) < 0.5: return -1
        if distort > 0:
            image = rdistort(image, 1.5, dsigma, cval=amax(image))
        fname = path + "/01%04d" % index
        # askewed
        image = open_image(image)

        if image.shape[0] > LENGTH:
            continue
            # break
        else:
            # Image.fromarray(image,'I').show()
            image = np.pad(image, ((0, LENGTH - image.shape[0]), (0, 0)), 'constant', constant_values=0)
            break
            # image = np.reshape(image, image.shape[:-1]).T
            # Image.fromarray(image * 255).show()

    # save
    try:
        if conv_flag:
            np.save(fname + '.npy', utils.twoD_3D(image))
        else:
            np.save(fname + '.npy', image)
        with codecs.open(fname + ".txt", "w", 'utf-8') as stream:
            assert len(line) == 20, 'length of string %s is not equal to 20' % fname
            stream.write(line)
    except:
        try:
            os.system('rm %s.npy' % fname)
            os.system('rm %s.txt' % fname)
        except:
            pass

    if (index + 1) % 100 == 0:
        print('%s' % (index + 1))
    return -1


def mp_generate(textlines, aim_path=None, threads_num=None, conv_flag=False):
    import multiprocessing as mp
    assert not aim_path == None, 'aim path is not given\n'
    utils.mkdir(aim_path)

    if threads_num == None:
        threads_num = mp.cpu_count() - 1
    else:
        pass
    mpool = mp.Pool(threads_num)
    nums = len(textlines)

    res = [mpool.apply_async(generate, (i, textlines[i], aim_path, conv_flag,))
           for i in xrange(nums)]
    res = [r.get() for r in res]
    mpool.close()


def fromText2Image(filedir='dataset'):
    assert os.path.exists(filedir), 'file %s direction does not exit.' % (filedir)
    textfiles = {
        'acceleration': '%s/text/acceleration' % filedir,
        # 'underfitting': '%s/text/underfitting' % filedir
    }
    for key, value in textfiles.items():
        # training
        aim_path = '%s/training/' % (value.replace('text', 'image'))
        utils.mkdir(aim_path)
        trainingText = utils.read_text(filename='%s/training.txt' % (value)).split('\n')
        if not len(os.listdir(aim_path)) == 2 * len(trainingText):
            print 'generating %s image' % (aim_path)
            mp_generate(textlines=trainingText, aim_path=aim_path, threads_num=8)
            del trainingText
        else:
            pass

        # test
        aim_path = '%s/test/' % (value.replace('text', 'image'))
        utils.mkdir(aim_path)
        testText = utils.read_text(filename='%s/test.txt' % (value)).split('\n')
        if not len(os.listdir(aim_path)) == 2 * len(testText):
            print 'generating %s image' % (aim_path)
            mp_generate(textlines=testText, aim_path=aim_path, threads_num=8)
            del testText
        else:
            pass


fromText2Image()
