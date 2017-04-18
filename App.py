#!/usr/bin/env python
# coding: utf-8

"""
    App
    ============================

    Handwritten characters and digits recognition application
    Structure/
        Extraction
        Model design - Prediction
        Integration (API)

    _copyright_ = 'Copyright (c) 2017 J.W. & Vm.C.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

from extraction import FeatureExtractor
from extraction import PageDetector


# Input  settings ####
#img = cv2.imread('input/pagina3_1.png', 0)
#img = cv2.imread('input/pagina3_2.jpeg', 0)
#img = cv2.imread('input/pagina3_3.jpg', 0)
img = cv2.imread('input/pagina3_4.png', 0)
#img = cv2.imread('input/pagina3_5.png', 0)
#img = cv2.imread('input/pagina3_6.png', 0)
#img = cv2.imread('input/pagina1_1.png', 0)
#img = cv2.imread('input/pagina1_2.png', 0)
#img = cv2.imread('input/pagina1_3.png', 0)
#img = cv2.imread('input/pagina2_1.png', 0)
#img = cv2.imread('input/pagina2_2.png', 0)
#img = cv2.imread('input/pagina4_1.png', 0)


if __name__ == '__main__':
    print("App: I'll try to be helpful :) \nBut I'm still just a robot. Sorry!")    

    img = PageDetector.enderezarImagen(img)
    page = PageDetector.detectPage(img)
    if page is not None:
        plt.imshow(page[0])
        plt.title('Es la página: '+str(page[1][0]))
        plt.show()
        if page[1][1] == 0: # esta orientado de manera normal
            FeatureExtractor.extractPageData(page[0],page[1][0])
        else:
            if page[1][1] == 1: #esta al revez
                raise NotImplementedError('Aún no se hace esto :(')
            else:
                raise ValueError('Error')
