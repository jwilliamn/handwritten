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


# 1. General settings to generate train data ####
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


# 2. Global settings of modelD1 ####
# Path to save model parameters
model_path = '/home/williamn/Repository/handwritten/modeling'

# Hyperparameters
depth = 16 # K = 6 number of filters
patch_size = 5 # F = 5 Size of filters
S = 1 # stride
P = 0 # amount of zero padding

# Learning parameters
learning_rate = 0.001  # 0.0001  # 0.001
batch_size = 16  # 8  # 16
n_epochs = 2 # 20  # 2

# Network params
input_size = 32
n_channels = 1 # greyscale
n_classes = 26  # classes (A-Z)