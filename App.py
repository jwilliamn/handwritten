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
import sys
import os

from wand.image import Image
from wand.color import Color
from PyPDF2 import PdfFileWriter, PdfFileReader


from extraction import FeatureExtractor
from extraction import PageDetector


# Input  settings ####
#img = cv2.imread('input/pagina3_1.png', 0) # Doesnt work either
#img = cv2.imread('input/pagina3_2.jpeg', 0)
#img = cv2.imread('input/pagina3_3.jpg', 0)
#img = cv2.imread('input/pagina3_4.png', 0)
#img = cv2.imread('input/pagina3_5.png', 0)
#img = cv2.imread('input/pagina3_6.png', 0)
#img = cv2.imread('input/pagina1_1.png', 0) # para Debugear
#img = cv2.imread('input/pagina1_2.png', 0) # Otra mas
#img = cv2.imread('input/pagina1_3.png', 0)
#img = cv2.imread('input/pagina1_4.png', 0)
#img = cv2.imread('input/pagina2_1.png', 0)
#img = cv2.imread('input/pagina2_2.png', 0)
#img = cv2.imread('input/pagina4_1.png', 0)
#img = cv2.imread('input/pagina4_2.png', 0)
def processPdf(originalPdf):
    inputPdf = PdfFileReader(open(originalPdf, 'rb'))
    
    if not os.path.exists('input/tmp/'):
        os.makedirs('input/tmp/')

    for i in range(inputPdf.getNumPages()):
        p = inputPdf.getPage(i)
        outputPdf = PdfFileWriter()
        outputPdf.addPage(p)

        with open('input/tmp/page_%1d.pdf' % (i +1), 'wb') as f:
            outputPdf.write(f)
    return inputPdf.getNumPages()


def convert_pdf_png(filepdf):
    path = filepdf
    path = path.split('.')

    try:
        with Image(filename=filepdf, resolution=300) as img:
            with Image(width=img.width, height=img.height, background=Color('white')) as bg:
                bg.composite(img, 0, 0)
                bg.save(filename=path[0] + '.png')
    except Exception as e:
        print('Unable to convert pdf file', e)
        raise
    imagePath = []
    imagePath = path[0] + '.png'
    print('imagepath in function',imagePath)
    return imagePath



#img = cv2.imread('input/pagina51.png', 0)


if __name__ == '__main__':
    print("App: I'll try to be helpful :) \nBut I'm still just a robot. Sorry!")    
    
    arg = sys.argv[1]
    print('arg', arg)
    splitArg = arg.split('.')

    if splitArg[1] == 'png' or splitArg[1] == 'jpeg':
        print("File is a picture!")
        imgPath = arg
    else:
        print('File is a pdf! (I hope)')
        numPag = processPdf(arg)
        if numPag > 1:
            print('Pdf has multiple pages, I\'ll process all of them though.')
            imgPath = convert_pdf_png(sys.argv[1])
        else:
            imgPath = convert_pdf_png(arg)

    img = cv2.imread(imgPath, 0)
    img = PageDetector.enderezarImagen(img)
    page = PageDetector.detectPage(img)

    print('So far, so good!')
    if page is not None:
        plt.imshow(page[0],'gray')
        plt.title('Es la página: '+str(page[1][0]))
        plt.show()
        if page[1][1] == 0: # esta orientado de manera normal
            FeatureExtractor.extractPageData(page[0],page[1][0],None,os.path.basename(imgPath))
            print('Still working!')
        else:
            if page[1][1] == 1: #esta al revez
                flipped = cv2.flip(page[0],0)
                flipped = cv2.flip(flipped, 1)
                FeatureExtractor.extractPageData(flipped, page[1][0],None,os.path.basename(imgPath))
            else:
                raise ValueError('Error')
