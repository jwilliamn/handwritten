#!/usr/bin/env python
# coding: utf-8

"""
    Extraction.VariableDefinition
    =============================

    Classes: category & variable

    _copyright_ = 'Copyright (c) 2017 Vm.C.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

class Category:

    def __init__(self, name, description):
        self.subTypes = []
        self.name = name
        self.description = description
        self.hasValue = False
        self.value = None

    def convert2ParsedValues(self):
        if self.hasValue:
            return Variable(self.name,'Valor predicho', self.value.convert2ParsedValues())
        else:
            newCategory = Category(self.name,self.description)
            for st in self.subTypes:
                newCategory.addSubType(st.convert2ParsedValues())
            return newCategory

    def jsonDefault(object):
        return object.__dict__

    def addSubType(self, subCategory):
        self.subTypes.append(subCategory)

    def getValue(self):
        return None

    def isLeaf(self):
        return len(self.subTypes) == 0

    def describe(self, explicit, ident = None):
        if ident is None:
            return self.describe(explicit, 0)

        if explicit:
            print(('\t' * ident) + '['+self.description+'] hasValue: ' + str(self.hasValue))
        print(('\t'*ident) + self.name + ' : ' +(('None' if self.value is None else str(self.value)) if self.hasValue else 'noValue'))
        for subtype in self.subTypes:
            subtype.describe( explicit,ident + 1)

    def getAllWithValue(self, baseName = ''):
        withValue = []
        if self.hasValue:
            withValue = [(baseName+' {'+self.name+'}', self)]
        else:
            for subtype in self.subTypes:
                if baseName == '':
                    subtype_withvalues = subtype.getAllWithValue('['+self.name+']')
                else:
                    subtype_withvalues = subtype.getAllWithValue(baseName+' ['+self.name+']')
                withValue.extend(subtype_withvalues)
        return withValue

    def __getitem__(self, key):
        for subtype in self.subTypes:
            if subtype.name == key:
                return subtype
        return None

    def __str__(self):
        return str(self.value)

class Variable(Category):
    def __init__(self, name, description, value): # value is rawValue
        super().__init__(name, description)
        self.hasValue = True
        self.value = value
    def jsonDefault(object):
        return object.__dict__
    def getValue(self):
        return self.value
