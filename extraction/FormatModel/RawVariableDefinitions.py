import cv2

from extraction.FormatModel import UtilFunctionsExtraction


class RawValue:
    def __init__(self, position, countItems=1, parser=None, nameParser='None', singleParser=None,
                 nameSingleParser='None'):

        self.position = position
        self.position[0]=(self.position[0][0],self.position[0][1])
        self.position[1] = (self.position[1][0], self.position[1][1])
        self.countItems = countItems
        self.parser = parser
        self.singleParser = singleParser
        self.nameParser = nameParser
        self.nameSingleParser = nameSingleParser

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
        UtilFunctionsExtraction.plotImagesWithPrediction(arrayResult,arrayOfImages)
        return arrayResult

    def parserImage2Categoric(self,arg):
        return 'yes'

    def letterPredictor(self, img):
        # pred_label = engine.predictImage(d)
        # return chr(pred_label + ord('A'))
        return 'A'

    def digitPredictor(self, img):
        # pred_label = engine.predictImageDigit(d)
        # return chr(pred_label + ord('A'))
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
