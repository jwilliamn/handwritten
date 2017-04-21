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
import modeling
from matplotlib import pyplot as plt


def plotearCategoriasPosicionesImagenes(img, cat_pos, cat_img):
    if cat_pos.hasValue:
        plotear(img, cat_pos.value.position, cat_pos.value.arrayOfImages, cat_pos.value.countItems,
                cat_pos.value.predictedValue)
    else:
        for st in cat_pos.subTypes:

            plotearCategoriasPosicionesImagenes(img, st, cat_img[st.name])

def plotear(img, position, arrayOfImages, countItems, arrayPredictedValues):
    for k in range(countItems):
        pixel_y = (position[0][1] + position[1][1])/2
        pixel_x = (position[0][0]*(countItems-k) + position[1][0]*k)/ countItems
        print('ploteando la imagen ',k, ' en ', position, ' valor predicho: ', arrayPredictedValues[k])
