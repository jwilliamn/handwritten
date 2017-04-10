#!/usr/bin/env python
# coding: utf-8

"""
    Modeling.GenerateTrainData09
    ============================

    A description which can be long and explain the complete
    functionality of this module even with indented code examples.
    Class/Function however should not be documented here.

    _copyright_ = 'Copyright (c) 2017 Vm.C.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import os

from GenerateTrainDataAZ import *
from modelSettings import *


if __name__ == '__main__':

    if not os.path.exists(pickleFolder):
        os.makedirs(pickleFolder)

    if not os.path.exists(testPickleFolder):
        os.makedirs(testPickleFolder)

    folderClases = [x[0] for x in os.walk(baseFolder)]

    trainFoldersTarget = []
    testFoldersTarget=[]
    for folder in folderClases:
            basename = os.path.basename(folder)
            if len(basename) == 2:
                a = hex2int(basename[0])*16+hex2int(basename[1])
                print(a, chr(a))
                trainFolder = getTrainFolder(folder)
                testFolder = getTestFolder(folder)

                #print(trainFolder)
                trainFoldersTarget.append(trainFolder)
                testFoldersTarget.append(testFolder)
                #print(getTrainFolder(folder))
    #print(trainFoldersTarget)
    [train_datasets,trainSizes, validSizes] = maybe_pickle_digit(trainFoldersTarget, pickleFolder, 10, force=True, percentToValid=20) # 1000


    [test_datasets, trainSizes_t, validSizes_t] = maybe_pickle_digit(testFoldersTarget, testPickleFolder, 10, force=True, percentToValid=0) #

    print(sum(trainSizes),sum(validSizes))
    print(sum(trainSizes_t), sum(validSizes_t))


    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets_digit(
        train_datasets, trainSizes, validSizes)
    _, _, test_dataset, test_labels = merge_datasets_digit(test_datasets, trainSizes_t,validSizes_t)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

    pickle_file = os.path.join(data_root, pickleFileDigits)

    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)
