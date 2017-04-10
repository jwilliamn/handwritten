#!/usr/bin/env python
# coding: utf-8

"""
    Modeling.modelSettings
    ======================

    List of general parameters to generate train data

    _copyright_ = 'Copyright (c) 2017 J.W.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import numpy as np
import os
import sys
import random
import time


# General settings to generate train data ####
# Image properties
image_size = 32  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

# Image resize
largeSize = 32
smallSize = 30

# Data path
data_root = '/home/williamn/Repository/data/mnistAll/'
baseFolder = data_root + 'by_class/'
pickleFolder = data_root + 'by_classPickle/'
testPickleFolder  = data_root + 'by_classPickle_test/'

# Pickle file name
pickleFile = 'MNIST_32x32.pickle'
pickleFileDigits = 'MNIST_digit_32x32.pickle'