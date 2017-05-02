#!/usr/bin/env python
# coding: utf-8

"""
    mindisApp
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
import time

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

    outputPath = []
    for i in range(inputPdf.getNumPages()):
        p = inputPdf.getPage(i)
        outputPdf = PdfFileWriter()
        outputPdf.addPage(p)

        with open('input/tmp/page_%1d.pdf' % (i +1), 'wb') as f:
            outputPdf.write(f)
            outputPath.append(f.name)
    
    outputPath = np.array(outputPath)
    #print('outputPdf', type(outputPath), outputPath)
    return outputPath, inputPdf.getNumPages()


def convert_pdf_png(filePath, numPages):
    imagePath = []
    for i in range(numPages):
        path = filePath[i]
        pathName = path.split('.')

        try:
            with Image(filename=path, resolution=600) as img:
                with Image(width=img.width, height=img.height, background=Color('white')) as bg:
                    bg.composite(img, 0, 0)
                    bg.save(filename=pathName[0] + '.png')
        except Exception as e:
            print('Unable to convert pdf file', e)
            raise

        imagePath.append(pathName[0] + '.png')
        print('Converting page %d' % (i + 1))
    imagePath = np.array(imagePath)
    return imagePath




# Main function ####
if __name__ == '__main__':
    """ .........
    To run the app, execute the following in terminal:

    [terminal_prompt]$ python App.py path/to/image.pdf

    Currently the app supports images in the following formats: 
        .png
        .jpeg
        .jpg
        .pdf
    """
    print("Hi there, its mindisApp I'll try to be helpful :) \nBut I'm still just a robot. Sorry!")    
    
    arg = sys.argv[1]
    print('arg', arg)
    splitArg = arg.split('.')

    if splitArg[1] == 'png' or splitArg[1] == 'jpeg' or splitArg[1] == 'jpg':
        print("File is a picture!")
        imgPath = np.array([arg])
    else:
        if splitArg[1] == 'pdf' or splitArg[1] == 'PDF':
            print('File is a pdf!')
            pdfPath, numPag = processPdf(arg)
            imgPath = convert_pdf_png(pdfPath, numPag)

            #print('imgPath__', type(imgPath), imgPath)
        else:
            raise ValueError(splitArg[1] + ' File format cannot be processed :(!')

    #print('imgPath for cv2', type(imgPath), len(imgPath))
    for i in range(len(imgPath)):
        print('***** IMAGE ' + str(i + 1) + ' PROCESSING *****')
        start_time = time.time()
        img = cv2.imread(imgPath[i], 0)
        img = PageDetector.enderezarImagen(img)
        page = PageDetector.detectPage(img)

        if page is not None:
            # plt.imshow(page[0],'gray')
            # plt.title(' Es la página: '+str(page[1][0]))
            # plt.show()
            print(' This image can be processed')
            if page[1][1] == 0: # esta orientado de manera normal
                FeatureExtractor.extractPageData(page[0],page[1][0],None,os.path.basename(imgPath[i]))
                print('Total time: {0} seconds'.format(time.time() - start_time))
            else:
                if page[1][1] == 1: #esta al revez
                    flipped = cv2.flip(page[0],0)
                    flipped = cv2.flip(flipped, 1)
                    FeatureExtractor.extractPageData(flipped, page[1][0],None,os.path.basename(imgPath[i]))
                    print('Total time: {0} seconds'.format(time.time() - start_time))
                else:
                    raise ValueError('Image cannot be processed, Check its quality\nUse a different scanner and try again')
