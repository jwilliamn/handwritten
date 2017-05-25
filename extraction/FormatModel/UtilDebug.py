#!/usr/bin/env python
# coding: utf-8

"""
    Extraction.UtilFunctions
    =============================

    Classes, .......

    _copyright_ = 'Copyright (c) 2017 Vm.C.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import sys
import time
import modeling


def plotearCategoriasPosicionesImagenes(img, cat_pos, cat_img):
    if cat_pos.hasValue:
        plotear(img, cat_pos.value.position, cat_pos.value.arrayOfImages, cat_pos.value.countItems,
                cat_img.value.predictedValue)
    else:
        for st in cat_pos.subTypes:
            plotearCategoriasPosicionesImagenes(img, st, cat_img[st.name])


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plotear(img, position, arrayOfImages, countItems, arrayPredictedValues):
    # arg = sys.argv[1]
    # background = Image.open(arg)
    if arrayOfImages is None or arrayPredictedValues is None:
        return

    if countItems != len(arrayOfImages) or countItems != len(arrayPredictedValues):
        return

    if arrayOfImages is not None:
        for k in range(countItems):
            pixel_y = int(round(position[0][1]))
            pixel_x = int(round((position[0][0] * (countItems - k) + position[1][0] * k) / countItems))

            # print('ploteando la imagen ', k, ' en ', position, ' valor predicho: ', arrayPredictedValues[k])
            if arrayOfImages[k] is not None:

                # print(arrayOfImages[k].dtype)
                # print(img.dtype)
                if len(position) == 2:
                    img32x32 = arrayOfImages[k] * 255.0 + 255.0 / 2.0

                    img32x32 = (img32x32.astype(img.dtype))

                    resized = cv2.resize(img32x32, (20, 20))
                else:
                    resized = cv2.resize(arrayOfImages[k], (0, 0), fx=0.8, fy=0.8)
                    resized = cv2.bitwise_not(resized)
                # plt.subplot(2,1,1), plt.imshow(arrayOfImages[k], 'gray')
                # plt.subplot(2, 1, 2), plt.imshow(img32x32, 'gray')
                # plt.show()
                img[pixel_y:(pixel_y + resized.shape[0]), pixel_x:(pixel_x + resized.shape[1])] = resized
                cv2.rectangle(img, (pixel_x-1, pixel_y-1), (pixel_x+20, pixel_y+20), color=0, thickness=1)
                cv2.putText(img, str(arrayPredictedValues[k]), (pixel_x, pixel_y + 5),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.8, color=0, thickness=2)
                # plt.subplot(3, 1, 1), plt.imshow(img, 'gray')
                # plt.subplot(3, 1, 2), plt.imshow(img32x32, 'gray')
                #
                # # r[r > 250] = 255
                # # r[r <= 250] = 0
                # # plt.subplot(3, 1, 3), plt.imshow(r, 'gray')
                # plt.show()

                # plt.imshow(img,'gray')
                # plt.show()
                # print('pixel Y', pixel_y)
                # print('pixel X', pixel_x)
                # background.paste(arrayOfImages[k], position, arrayOfImages[k])
                # background.show()


class ProcessTimer:
    def __init__(self, name):
        print('created: ', name)
        self.name = name
        self.count = 0
        self.secs = 0
        self.start_time = time.time()
        self.end_time = time.time()

    def startTimer(self, count):
        self.start_time = time.time()
        self.count += count

    def endTimer(self, ):
        self.end_time = time.time()
        self.secs += self.end_time - self.start_time

    def __str__(self):
        secs = round((self.secs * 1000.0)) / 1000.0
        str_secs = '%.3f' % secs
        while len(str_secs) < 8:
            str_secs = ' ' + str_secs
        name = self.name
        while len(name) < 40:
            name += ' '
        str_count = str(self.count)
        while len(str_count) < 4:
            str_count = ' ' + str_count

        return name + ': ' + str_secs + ' secs for ' + str_count + ' elements'


class CategoryTimer(ProcessTimer):
    instance = None

    def __new__(cls):
        if not CategoryTimer.instance:
            CategoryTimer.instance = ProcessTimer('Extractor y predictor categorias')
        return CategoryTimer.instance


class ArrayLetterTimer(ProcessTimer):
    instance = None

    def __new__(cls):
        if not ArrayLetterTimer.instance:
            ArrayLetterTimer.instance = ProcessTimer('Extractor bloques letras')
        return ArrayLetterTimer.instance


class ArrayDigitTimer(ProcessTimer):
    instance = None

    def __new__(cls):
        if not ArrayDigitTimer.instance:
            ArrayDigitTimer.instance = ProcessTimer('Extractor bloques digitos')
        return ArrayDigitTimer.instance


class PageDetectorTimer(ProcessTimer):
    instance = None

    def __new__(cls):
        if not PageDetectorTimer.instance:
            PageDetectorTimer.instance = ProcessTimer('PageDetectorTimer')
        return PageDetectorTimer.instance


class PredictorTimer(ProcessTimer):
    instance = None

    def __new__(cls):
        if not PredictorTimer.instance:
            PredictorTimer.instance = ProcessTimer('Predictor de letras y digitos')
        return PredictorTimer.instance


class RatiosBuffer:
    instance = None

    def __new__(cls):
        if not RatiosBuffer.instance:
            RatiosBuffer.instance = []
        return RatiosBuffer.instance


if __name__ == '__main__':
    timer_A = CategoryTimer()
    timer_A.startTimer(4)
    for i in range(0, 10000000):
        pass
    timer_A.endTimer()

    timer_A = PageDetectorTimer()
    timer_A.startTimer(3)
    for i in range(0, 10000000):
        pass
    timer_A.endTimer()

    timer_A = CategoryTimer()
    timer_A.startTimer(4)
    for i in range(0, 10000000):
        pass
    timer_A.endTimer()

    timer_A = ArrayDigitTimer()
    timer_A.startTimer(2)
    for i in range(0, 10000000):
        pass
    timer_A.endTimer()

    timers = [CategoryTimer(), ArrayLetterTimer(), ArrayDigitTimer(), PageDetectorTimer(), PredictorTimer()]
    for timer in timers:
        print(str(timer))
