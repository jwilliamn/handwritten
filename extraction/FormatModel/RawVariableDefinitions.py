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

from extraction.FormatModel import UtilFunctionsExtraction, UtilDebug
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
            if nameSingleParser is not None and nameSingleParser == 'digitPredictor':
                self.predictedValue =[]
                EngineDigit = engine.UniqueEngineDigit()
                for v in value:
                    if v is not None:
                        self.predictedValue.append(chr(EngineDigit.pred[v] + ord('0')))
                    else:
                        self.predictedValue.append(' ')
            elif nameSingleParser is not None and nameSingleParser == 'letterPredictor':
                self.predictedValue = []
                EngineLetter = engine.UniqueEngineLetter()
                for v in value:

                    if v is not None:
                        self.predictedValue.append(chr(EngineLetter.pred[v] + ord('A')))
                    else:
                        self.predictedValue.append(' ')
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
        if self.singleParser == self.letterPredictor:
            charArray_timer = UtilDebug.ArrayLetterTimer()
        else:
            charArray_timer = UtilDebug.ArrayDigitTimer()

        charArray_timer.startTimer(1) #self.count
        img = arg[0]
        onlyUserMarks = arg[1]
        TL = self.position[0]
        BR = self.position[1]
        count = self.countItems
        arrayOfImages = UtilFunctionsExtraction.extractCharacters(img, onlyUserMarks, TL, BR, count)
        arrayResult = []
        for singleImage in arrayOfImages:
            if singleImage is None:
                predicted = None
            else:
                predicted = self.singleParser(singleImage)

            arrayResult.append(predicted)
        #UtilFunctionsExtraction.plotImagesWithPrediction(arrayResult,arrayOfImages)
        self.predictedValue = arrayResult
        self.arrayOfImages = arrayOfImages
        charArray_timer.endTimer()
        return self.predictedValue

    def getFinalValue(self, arg):
        return self.predictedValue

    def parserImage2Categoric(self,arg):
        if self.singleParser is not None:
            categoric_timer = UtilDebug.CategoryTimer()
            categoric_timer.startTimer(1)
            ret = self.singleParser(arg)
            categoric_timer.endTimer()
        else:
            self.arrayOfImages = None
            self.predictedValue = ['unknow']
            return self.predictedValue


    def parserCategoricLabelsInside(self, arg):
        img = arg[0]
        TL = self.position[0]
        BR = self.position[1]
        labels = self.position[2]
        self.countItems = len(labels)
        self.arrayOfImages = UtilFunctionsExtraction.extractCategory_extractColumnLabelsInside(img, TL, BR, len(labels))
        results = UtilFunctionsExtraction.predictValuesCategory_labelsInside(self.arrayOfImages, labels)
        self.predictedValue = []
        for r in results:
            self.predictedValue.append(r)
        return self.predictedValue
    def parserCategoricLabelsLeft(self, arg):
        img = arg[0]
        TL = self.position[0]
        BR = self.position[1]
        labels = self.position[2]
        self.countItems = len(labels)
        self.arrayOfImages = UtilFunctionsExtraction.extractCategory_extractColumnLabelsLeft(img, TL, BR, len(labels))
        results = UtilFunctionsExtraction.predictValuesCategory_labelsLeft(self.arrayOfImages, labels)
        self.predictedValue = []
        for r in results:
            self.predictedValue.append(r)
        return self.predictedValue
    def parserCategoricLabelsSex(self, arg):
        img = arg[0]
        TL = self.position[0]
        BR = self.position[1]
        labels = self.position[2]
        self.countItems = len(labels)
        self.arrayOfImages = UtilFunctionsExtraction.extractCategory_extractColumnLabelsSex(img, TL, BR, len(labels))
        results = UtilFunctionsExtraction.predictValuesCategory_labelsSex(self.arrayOfImages, labels)
        self.predictedValue = []
        for r in results:
            self.predictedValue.append(r)
        return self.predictedValue

    def parserCategoricLabelsDocumento(self, arg):
        img = arg[0]
        TL = self.position[0]
        BR = self.position[1]
        labels = self.position[2]
        self.countItems = len(labels)
        self.arrayOfImages = UtilFunctionsExtraction.extractCategory_extractColumnLabelsDocumento(img, TL, BR, len(labels))
        results = UtilFunctionsExtraction.predictValuesCategory_labelsDocumento(self.arrayOfImages, labels)
        self.predictedValue = []
        for r in results:
            self.predictedValue.append(r)
        return self.predictedValue
    def parserCategoricLabelsTipoSuministro(self, arg):
        img = arg[0]
        TL = self.position[0]
        BR = self.position[1]
        labels = self.position[2]
        self.countItems = len(labels)
        self.arrayOfImages = UtilFunctionsExtraction.extractCategory_extractColumnLabelsTipoSuministro(img, TL, BR, len(labels))
        results = UtilFunctionsExtraction.predictValuesCategory_labelsSingleButtons(self.arrayOfImages, labels)
        self.predictedValue = []
        for r in results:
            self.predictedValue.append(r)
        return self.predictedValue

    def parserCategoricLabelsTipoVia(self, arg):
        img = arg[0]
        TL = self.position[0]
        BR = self.position[1]
        labels = self.position[2]
        self.countItems = len(labels)
        self.arrayOfImages = UtilFunctionsExtraction.extractCategory_extractColumnLabelsTipoVia(img, TL, BR, len(labels))
        results = UtilFunctionsExtraction.predictValuesCategory_labelsSingleButtons(self.arrayOfImages, labels)
        self.predictedValue = []
        for r in results:
            self.predictedValue.append(r)
        return self.predictedValue

    def parserCategoricLabelsSiNo(self, arg):
        img = arg[0]
        TL = self.position[0]
        BR = self.position[1]
        labels = self.position[2]
        self.countItems = len(labels)
        self.arrayOfImages = UtilFunctionsExtraction.extractCategory_extractColumnLabelsTipoSiNo(img, TL, BR, len(labels))
        results = UtilFunctionsExtraction.predictValuesCategory_labelsSingleButtons(self.arrayOfImages, labels)
        self.predictedValue = []
        for r in results:
            self.predictedValue.append(r)
        return self.predictedValue

    def letterPredictor(self, img):
        indx = engine.predictImage(img)
        return indx#chr(pred_label + ord('A'))
        #return 'A'

    def digitPredictor(self, img):
        indx = engine.predictImageDigit(img)
        return indx#chr(pred_label + ord('0'))
        #return '0'

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

class ImageCategoricLabelsInside(RawValue):
    def __init__(self, position, count):
        if count == 1:
            super().__init__(position, 1, self.parserImage2Categoric, 'parserImage2Categoric',
                             self.parserCategoricLabelsInside, 'parserCategoricLabelsInside')
        else:
            raise Exception('Categoric values always should have count = 1')
    def __str__(self):
        return 'ImageCategoric  : ' + str(self.countItems)
class ImageCategoricLabelsLeft(RawValue):
    def __init__(self, position, count):
        if count == 1:
            super().__init__(position, 1, self.parserImage2Categoric, 'parserImage2Categoric',
                             self.parserCategoricLabelsLeft, 'parserCategoricLabelsLeft')
        else:
            raise Exception('Categoric values always should have count = 1')

    def __str__(self):
        return 'ImageCategoric  : ' + str(self.countItems)

class ImageCategoricLabelsSex(RawValue):
    def __init__(self, position, count):
        if count == 1:
            super().__init__(position, 1, self.parserImage2Categoric, 'parserImage2Categoric',
                             self.parserCategoricLabelsSex, 'parserCategoricLabelsSex')
        else:
            raise Exception('Categoric values always should have count = 1')

    def __str__(self):
        return 'ImageCategoric  : ' + str(self.countItems)

class ImageCategoricLabelsSiNo(RawValue):
    def __init__(self, position, count):
        if count == 1:
            super().__init__(position, 1, self.parserImage2Categoric, 'parserImage2Categoric',
                             self.parserCategoricLabelsSiNo, 'parserCategoricLabelsSiNo')
        else:
            raise Exception('Categoric values always should have count = 1')

    def __str__(self):
        return 'ImageCategoric  : ' + str(self.countItems)

class ImageCategoricLabelsTipoVia(RawValue):
    def __init__(self, position, count):
        if count == 1:
            super().__init__(position, 1, self.parserImage2Categoric, 'parserImage2Categoric',
                             self.parserCategoricLabelsTipoVia, 'parserCategoricLabelsTipoVia')
        else:
            raise Exception('Categoric values always should have count = 1')

    def __str__(self):
        return 'ImageCategoric  : ' + str(self.countItems)

class ImageCategoricLabelsDocumento(RawValue):
    def __init__(self, position, count):
        if count == 1:
            super().__init__(position, 1, self.parserImage2Categoric, 'parserImage2Categoric',
                             self.parserCategoricLabelsDocumento, 'parserCategoricLabelsDocumento')
        else:
            raise Exception('Categoric values always should have count = 1')

    def __str__(self):
        return 'ImageCategoric  : ' + str(self.countItems)
class ImageCategoricLabelsTipoSuministro(RawValue):
    def __init__(self, position, count):
        if count == 1:
            super().__init__(position, 1, self.parserImage2Categoric, 'parserImage2Categoric',
                             self.parserCategoricLabelsTipoSuministro, 'parserCategoricLabelsTipoSuministro')
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