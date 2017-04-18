from extraction.FormatModel.RawVariableDefinitions import *
from extraction.FormatModel.VariableDefinitions import Category


def loadValue(dict_current):
    if dict_current is None:
        return None
    if dict_current['nameParser'] == 'parserImage2ArrayChar':
        if dict_current['nameSingleParser'] == 'letterPredictor':
            print(dict_current['position'])
            return ArrayImageChar(dict_current['position'], dict_current['countItems'])
        if dict_current['nameSingleParser'] == 'digitPredictor':
            return ArrayImageNumber(dict_current['position'], dict_current['countItems'])
    elif dict_current['nameParser'] == 'parserImage2Categoric':
        return ImageCategoric(dict_current['position'],1)
    return None

def loadCategory(dict_current):
    cat = Category(dict_current['name'], dict_current['description'])
    cat.value = loadValue(dict_current['value'])
    cat.hasValue = dict_current['hasValue']
    for sub in dict_current['subTypes']:
        print(sub)
        r = loadCategory(sub)
        cat.addSubType(r)
    return cat

