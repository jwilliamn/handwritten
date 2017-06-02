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
from extraction.FormatModel import UtilDebug
from modeling import GenerateTrainDataAZ

from api import engine


# Global settings ####
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]



# Function definitions ####
# Very important function ####
def extractPageData(img, pageNumber, baseL = None, page_name = 'pagina'):
    """Detecta la orientación de la imagen y la orienta.
    Args:
        img: Original image.
        pageNumber:
        baseL: 
    Returns:
        __
    """
    print('Extracting for page: ', page_name)
    if(pageNumber == 3):
        return extractPageData_number3(img, baseL, page_name)
    if(pageNumber == 1):
        return extractPageData_number1(img, baseL, page_name)
    if (pageNumber == 2):
        return extractPageData_number2(img, baseL, page_name)
    if (pageNumber == 4):
        return extractPageData_number4(img, baseL, page_name)
    raise ValueError('Only implemented for page 1, 2, 3 & 4 of FSU, not for page: '+ str(pageNumber))



def jsonDefault(object):
    return object.__dict__

def extractPageData_numberX(img_original, baseL, str_number, page_name = 'image_unknow'):
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

        dict_Page1 = json.load(input)
        Page = loadCategory(dict_Page1)
        print(Page)

    Page.describe(True)

    engine.initEngines()
    R = Page.getAllWithValue()

    Page.calcCuadro(img)
    cb = UtilFunctionsExtraction.CuadroBuffer()
    cb.calc()
    print('calculated values:', cb.A_predicted, cb.B_predicted)
    for category in R:
        if category[1].value is not None:
            print('Parsing category: ',category[0])
            print('Value: ',category[1].value)
            parsed = category[1].value.parse([img, Ifp2])
            print('Parsed value: ',parsed)

    timer_predictor =UtilDebug.PredictorTimer()
    DigitPredictor = engine.UniqueEngineDigit()
    LetterPredictor = engine.UniqueEngineLetter()
    timer_predictor.startTimer(2)

    DigitPredictor.runEngine()
    LetterPredictor.runEngine()
    timer_predictor.endTimer()

    Page_parsed = Page.convert2ParsedValues()
    if Page_parsed is not None or Page is not None:
        UtilDebug.plotearCategoriasPosicionesImagenes(img, Page, Page_parsed)
    #cv2.imwrite('output/'+page_name+'_resultImage.png', img)
    cv2.imwrite('output/'+page_name, img)
    page_name = page_name.split('.')

    characterDebugger = UtilDebug.CharacterDebugger()
    characterDebugger.printOnDisk('output/testingCharacters.png')

    with open('output/'+ page_name[len(page_name) - 2] + '_' + str_number+'.json', 'w') as output:
        json.dump(Page_parsed, output, default=jsonDefault, indent=4)


def extractPageData_number1(img_original, baseL, page_name = 'pagina_1'):
    extractPageData_numberX(img_original, baseL, '1', page_name)

def extractPageData_number2(img_original, baseL, page_name = 'pagina_2'):
    extractPageData_numberX(img_original, baseL, '2', page_name)

def extractPageData_number3(img_original, baseL, page_name = 'pagina_3'):
    extractPageData_numberX(img_original, baseL, '3', page_name)

def extractPageData_number4(img_original, baseL, page_name = 'pagina_4'):
    extractPageData_numberX(img_original, baseL, '4', page_name)