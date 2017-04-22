#!/usr/bin/env python
# coding: utf-8

"""
    Extraction.FeatureExtractor
    ============================

    Core of the extraction module.
    Extract, filter, process and format images which then be used as
    input for prediction.

    _copyright_ = 'Copyright (c) 2017 Vm.C.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
import json

from extraction.FormatModel.UtilFunctionsLoadTemplates import loadCategory
from extraction.FormatModel.VariableDefinitions import *
from extraction.FormatModel.RawVariableDefinitions import *
from extraction.FormatModel.UtilDebug import * 
from modeling import GenerateTrainDataAZ

from api import engine


# Global settings ####
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]



# Function definitions ####
# Very important function ####
def extractPageData(img, pageNumber, baseL = None):
    """Detecta la orientación de la imagen y la orienta.
    Args:
        img: Original image.
        pageNumber:
        baseL: 
    Returns:
        __
    """
    if(pageNumber == 3):
        return extractPageData_number3(img, baseL)
    if(pageNumber == 1):
        return extractPageData_number1(img, baseL)
    if (pageNumber == 2):
        return extractPageData_number2(img, baseL)
    if (pageNumber == 4):
        return extractPageData_number4(img, baseL)
    raise ValueError('Only implemented for page 1, 2, 3 & 4 of FSU, not for page: '+ str(pageNumber))



def jsonDefault(object):
    return object.__dict__

def extractPageData_numberX(img_original, baseL, str_number):
    """Detecta la orientación de la imagen y la orienta.
    Args:
        image_original: Original image.
        baseL:
        str_number: 
    Returns:
        __
    """
    paginaBase = cv2.imread('resources/pag'+str_number+'_1_Template.png', 0)
    img = cv2.resize(img_original, (paginaBase.shape[1], paginaBase.shape[0]))
    ret3, If = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    SEh = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    SEv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    opHorizontal = cv2.morphologyEx(If, cv2.MORPH_OPEN, SEh)
    opVertical = cv2.morphologyEx(If, cv2.MORPH_OPEN, SEv)

    Ifp = cv2.bitwise_xor(If, cv2.bitwise_or(opHorizontal, opVertical))
    NegPaginaBase = cv2.bitwise_not(paginaBase)
    # print(img.shape, ' <-> ', NegPaginaBase.shape)
    Ifp2 = cv2.medianBlur(Ifp, 3)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    Ifp2 = cv2.morphologyEx(Ifp2, cv2.MORPH_OPEN, se)
    edgesToDebug = If.copy()
    edgesToDebug[edgesToDebug > 0] = 125

    with open('extraction/FormatModel/pagina'+str_number+'.json', 'r') as input:
        print('INPUT: ', input)
        dict_Page1 = json.load(input)
        Page = loadCategory(dict_Page1)
        print(Page)

    Page.describe(True)
    R = Page.getAllWithValue()

    for category in R:
        if category[1].value is not None:
            print(category[0])
            print(category[1].value)
            parsed = category[1].value.parse([img, Ifp2])
            print(parsed)

    Page_parsed = Page.convert2ParsedValues()
    if Page_parsed is not None or Page is not None:
        plotearCategoriasPosicionesImagenes(img, Page, Page_parsed)
    cv2.imwrite('resultImage.png', img)
    with open('output/predictedValues_pag'+str_number+'.json', 'w') as output:
        json.dump(Page_parsed, output, default=jsonDefault, indent=4)


def extractPageData_number1(img_original, baseL):
    extractPageData_numberX(img_original, baseL, '1')

def extractPageData_number2(img_original, baseL):
    extractPageData_numberX(img_original, baseL, '2')

def extractPageData_number3(img_original, baseL):
    extractPageData_numberX(img_original, baseL, '3')

def extractPageData_number4(img_original, baseL):
    extractPageData_numberX(img_original, baseL, '4')