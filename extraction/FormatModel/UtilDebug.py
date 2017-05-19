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

import modeling


def plotearCategoriasPosicionesImagenes(img, cat_pos, cat_img):
    if cat_pos.hasValue:
        plotear(img, cat_pos.value.position, cat_pos.value.arrayOfImages, cat_pos.value.countItems,
                cat_pos.value.predictedValue)
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
    #arg = sys.argv[1]
    #background = Image.open(arg)
    if arrayOfImages is None or arrayPredictedValues is None:
        return

    if countItems != len(arrayOfImages) or countItems != len(arrayPredictedValues):
        return

    if arrayOfImages is not None:
        for k in range(countItems):
            pixel_y = int(round(position[0][1]))
            pixel_x = int(round((position[0][0]*(countItems-k) + position[1][0]*k)/ countItems))

            print('ploteando la imagen ',k, ' en ', position, ' valor predicho: ', arrayPredictedValues[k])
            if arrayOfImages[k] is not None:

                #print(arrayOfImages[k].dtype)
                #print(img.dtype)
                if len(position)==2:
                    img32x32 = arrayOfImages[k]*255.0+255.0/2.0
                    print(img32x32.dtype)
                    img32x32 = (img32x32.astype(img.dtype))

                    resized = cv2.resize(img32x32, (20,20))
                else:
                    resized = cv2.resize(arrayOfImages[k], (0,0), fx=0.8, fy=0.8)
                    resized = cv2.bitwise_not(resized)
                # plt.subplot(2,1,1), plt.imshow(arrayOfImages[k], 'gray')
                # plt.subplot(2, 1, 2), plt.imshow(img32x32, 'gray')
                # plt.show()
                #img[pixel_y:(pixel_y + resized.shape[0]), pixel_x:(pixel_x + resized.shape[1])] = resized
                #cv2.rectangle(img, (pixel_x-1, pixel_y-1), (pixel_x+20, pixel_y+20), color=0, thickness=1)
                cv2.putText(img, str(arrayPredictedValues[k]), (pixel_x, pixel_y + 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
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
            #print('pixel Y', pixel_y)
            #print('pixel X', pixel_x)
            #background.paste(arrayOfImages[k], position, arrayOfImages[k])
            #background.show()

