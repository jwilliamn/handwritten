#!/usr/bin/env python
# coding: utf-8

"""
    Extraction.RawVariableDefinition
    =============================

    Classes, .......

    _copyright_ = 'Copyright (c) 2017 Vm.C.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import cv2

from extraction.FormatModel import UtilFunctionsExtraction
from api import engine


class RawValue:
    def __init__(self, value, countItems=1, parser=None, nameParser='None', singleParser=None,
                 nameSingleParser='None'):

        if countItems >=1:
            self.position = value
            self.position[0]=(self.position[0][0],self.position[0][1])
            self.position[1] = (self.position[1][0], self.position[1][1])
            self.countItems = countItems
            self.parser = parser
            self.singleParser = singleParser
            self.nameParser = nameParser
            self.nameSingleParser = nameSingleParser
            self.predictedValue = None
            self.arrayOfImages = None
        else:
            self.predictedValue = value
            self.nameParser = nameParser
            self.nameSingleParser = nameSingleParser
            self.countItems = -1

    def convert2ParsedValues(self):
        if self.nameParser == 'parserImage2ArrayChar':
            if self.nameSingleParser == 'letterPredictor':
                return ArrayPredictedChar(self.predictedValue)
            elif self.nameSingleParser == 'digitPredictor':
                return ArrayPredictedNumber(self.predictedValue)
        elif self.nameParser == 'parserImage2Categoric':
            return PredictedCategoric(self.predictedValue)
        print(self.nameParser)
        if self.nameSingleParser is not None:
            print(self.nameSingleParser)
        else:
            print('nameSingleParser is None')
        raise Exception('bad arguments or not implemented parser')

    def jsonDefault(object):
        return object.__dict__

    def __str__(self):
        return str(self.position)

    def getPosition(self):
        return self.position

    def parse(self, arg):
        if self.parser is None:
            raise Exception('no parser given')

        return self.parser(arg)

    def drawPosition(self, img):
        TL = self.position[0]
        BR = self.position[1]
        # print('TL ', TL)
        # TL = (TL[0],TL[1])
        # BR = (BR[0], BR[1])
        # print('BR ', BR)
        cv2.rectangle(img, TL, BR, (0, 255, 0), 2)

    def parserImage2ArrayChar(self, arg):

        img = arg[0]
        onlyUserMarks = arg[1]
        TL = self.position[0]
        BR = self.position[1]
        count = self.countItems
        arrayOfImages = UtilFunctionsExtraction.extractCharacters(img, onlyUserMarks, TL, BR, count)
        arrayResult = []
        for singleImage in arrayOfImages:
            if singleImage is None:
                predicted = ' '
            else:
                predicted = self.singleParser(singleImage)

            arrayResult.append(predicted)
        #UtilFunctionsExtraction.plotImagesWithPrediction(arrayResult,arrayOfImages)
        self.predictedValue = arrayResult
        self.arrayOfImages = arrayOfImages
        return self.predictedValue

    def getFinalValue(self, arg):
        return self.predictedValue

    def parserImage2Categoric(self,arg):
        if self.singleParser is not None:
            return self.singleParser(arg)
        else:
            self.arrayOfImages = None
            self.predictedValue = ['unknow']
            return self.predictedValue


    def parserCategoricSimpleSelection(self, arg):
        img = arg[0]
        TL = self.position[0]
        BR = self.position[1]
        labels = self.position[2]
        self.countItems = len(labels)
        self.arrayOfImages = UtilFunctionsExtraction.extractCategory_simpleImages(img, TL, BR, len(labels))
        results = UtilFunctionsExtraction.predictValuesCategory_simple(self.arrayOfImages, labels)
        self.predictedValue = []
        for r in results:
            self.predictedValue.append(r)
        return self.predictedValue
    def parserCategoricSimpleColumn(self, arg):
        img = arg[0]
        TL = self.position[0]
        BR = self.position[1]
        labels = self.position[2]
        self.countItems = len(labels)
        self.arrayOfImages = UtilFunctionsExtraction.extractCategory_singleColumn(img, TL, BR, len(labels))
        results = UtilFunctionsExtraction.predictValuesCategory_simple(self.arrayOfImages, labels, sz=5)
        self.predictedValue = []
        for r in results:
            self.predictedValue.append(r)
        return self.predictedValue

    def letterPredictor(self, img):
        #pred_label = engine.predictImage(img)
        #return chr(pred_label + ord('A'))
        return 'A'

    def digitPredictor(self, img):
        #pred_label = engine.predictImageDigit(img)
        #return chr(pred_label + ord('0'))
        return '0'

class ArrayImageNumber(RawValue):
    def __init__(self, position, count):
        super().__init__(position, count, self.parserImage2ArrayChar, 'parserImage2ArrayChar',
                         self.digitPredictor, 'digitPredictor')
    def __str__(self):
        return 'ArrayImageNumber: ' + str(self.countItems)


class ArrayImageChar(RawValue):
    def __init__(self, position, count):
        super().__init__(position, count, self.parserImage2ArrayChar, 'parserImage2ArrayChar',
                         self.letterPredictor, 'letterPredictor')
    def __str__(self):
        return 'ArrayImageChar: ' + str(self.countItems)

class ImageCategoric(RawValue):
    def __init__(self, position, count):
        if count == 1:
            super().__init__(position, 1, self.parserImage2Categoric, 'parserImage2Categoric', None, None)
        else:
            raise Exception('Categoric values always should have count = 1')
    def __str__(self):
        return 'ImageCategoric  : ' + str(self.countItems)

class ImageCategoricSimpleSelection(RawValue):
    def __init__(self, position, count):
        if count == 1:
            super().__init__(position, 1, self.parserImage2Categoric, 'parserImage2Categoric', self.parserCategoricSimpleSelection, 'parserCategoricSimpleSelection')
        else:
            raise Exception('Categoric values always should have count = 1')
    def __str__(self):
        return 'ImageCategoric  : ' + str(self.countItems)



class ImageCategoricSingleColumn(RawValue):
    def __init__(self, position, count):
        if count == 1:
            super().__init__(position, 1, self.parserImage2Categoric, 'parserImage2Categoric',
                             self.parserCategoricSimpleColumn, 'parserCategoricSimpleColumn')
        else:
            raise Exception('Categoric values always should have count = 1')

    def __str__(self):
        return 'ImageCategoric  : ' + str(self.countItems)
class ArrayPredictedNumber(RawValue):
    def __init__(self, value):
        super().__init__(value, -1, None, 'parserImage2ArrayChar',None, 'digitPredictor')

    def __str__(self):
        return 'ArrayPredictedNumber: ' + str(self.countItems)


class ArrayPredictedChar(RawValue):
    def __init__(self, value):
        super().__init__(value, -1, None, 'parserImage2ArrayChar', None, 'letterPredictor')
    def __str__(self):
        return 'ArrayPredictedChar: ' + str(self.countItems)

class PredictedCategoric(RawValue):
    def __init__(self, value):
        super().__init__(value, -1, None, 'parserImage2Categoric', None, None)

    def __str__(self):
        return 'PredictedCategoric  : ' + str(self.countItems)