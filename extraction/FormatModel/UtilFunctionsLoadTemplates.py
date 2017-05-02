#!/usr/bin/env python
# coding: utf-8

"""
    Extraction.LoadTemplates
    ============================

    Bunch of tools for feature extraction

    _copyright_ = 'Copyright (c) 2017 Vm.C.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""


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
        if dict_current['nameSingleParser'] is None:
            return ImageCategoric(dict_current['position'],1)

        if dict_current['nameSingleParser'] == 'parserCategoricSimpleSelection':

            return ImageCategoricSimpleSelection(dict_current['position'],1)
        if dict_current['nameSingleParser'] == 'parserCategoricSimpleColumn':
            return ImageCategoricSingleColumn(dict_current['position'],1)
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

