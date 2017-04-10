#!/usr/bin/env python
# coding: utf-8

"""
    Modeling.GenerateTrainDataAZ
    ============================

    A description which can be long and explain the complete
    functionality of this module even with indented code examples.
    Class/Function however should not be documented here.

    _copyright_ = 'Copyright (c) 2017 Vm.C.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

#from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from scipy.misc.pilutil import imresize
from six.moves import cPickle as pickle
import random
import time
from scipy.misc import toimage
import cv2

from modeling.modelSettings import * 

# Function definitions ####
def findNonEmptyInterval(a):
    leftIndx=0
    rightIndx=len(a)-1
    for num in a:
        if num == 0:
            leftIndx += 1
        else:
            break

    for num in reversed(a):
        if num == 0:
            rightIndx -= 1
        else:
            break

    if leftIndx>rightIndx:
        raise Exception('Imagen vacia')
    return leftIndx, rightIndx


def extractImportantSubset(a):
    c = a < 150
    sumCol = np.sum(c, axis=0)
    sumRow = np.sum(c, axis=1)
    [l_col, r_col] = findNonEmptyInterval(sumCol)
    [l_row, r_row] = findNonEmptyInterval(sumRow)

    n1 = range(l_col, r_col + 1)
    n2 = range(l_row, r_row + 1)

    subset = a[np.ix_(n2, n1)]


    #return toimage(subset)
    return subset


def load_letter(folder, min_num_images, max_num_images = None):

    if max_num_images is None:
        max_num_images = 1000000

    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    counterImage = 2
    for image in image_files:
        counterImage -= 1
        image_file = os.path.join(folder, image)
        try:
            image_data = ndimage.imread(image_file, True)

            #image_data = myImResize(image_data, imSize=image_size)
            image_data = myImResize_forDataTraining(image_data, None)
            if counterImage >= 0:
                pass
                #plt.imshow(image_data, cmap=plt.cm.gray)
                #plt.title('creating letter like this')
                #plt.show()

            image_data = (image_data -
                          pixel_depth / 2) / pixel_depth

            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data

            num_images += 1
            if num_images == max_num_images:
                break
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def calcShapeToResize(shape,imSize):
    m = max(shape)
    factor = imSize*1.0/m
    newShape = ((int)(round(shape[0]*factor)), (int)(round(shape[1]*factor)))
    return newShape


def myImResize_20x20_32x32(image):
    largeSize = 32
    smallSize = 30

    image = np.uint8(image)

    notImage = cv2.bitwise_not(image)
    #########3
    stats = cv2.connectedComponentsWithStats(notImage, connectivity=4)
    num_labels = stats[0]
    labels = stats[1]
    labelStats = stats[2]
    centroides = stats[3]
    # We expect the conneccted compoennt of the numbers to be more or less with a constats ratio
    # So we find the medina ratio of all the comeonets because the majorty of connected compoent are numbers
    cosl = []
    edgesLength = []
    indx_biggets_area = -1
    biggets_area = -1

    for label in range(1,num_labels):
        if labelStats[label, cv2.CC_STAT_AREA] > biggets_area:
            indx_biggets_area = label
            biggets_area = labelStats[label, cv2.CC_STAT_AREA]
    if indx_biggets_area == -1:
        raise Exception('no hay ningun objeto, posiblemente imagen vacia')


    scale = smallSize*min(1.0/labelStats[indx_biggets_area, cv2.CC_STAT_HEIGHT],1.0/labelStats[indx_biggets_area, cv2.CC_STAT_WIDTH])

    mainCharacter20x20 = cv2.resize(image, None, fx=scale, fy=scale, interpolation= cv2.INTER_AREA)
    # mainCharacter20x20[mainCharacter20x20 > 50] = 255
    # mainCharacter20x20[mainCharacter20x20 <= 50] = 0

    left = int(round(labelStats[indx_biggets_area, cv2.CC_STAT_LEFT] * scale))
    top = int(round(labelStats[indx_biggets_area, cv2.CC_STAT_TOP] * scale))

    width = int(round(labelStats[indx_biggets_area, cv2.CC_STAT_WIDTH]*scale))
    height = int(round(labelStats[indx_biggets_area, cv2.CC_STAT_HEIGHT] * scale))


    centerRows = top + height//2
    centerCols = left + width//2
    RowsTop = centerRows - largeSize//2 # por cada positivo, quitar una fila, por cada negativo agregar
    RowsBot = mainCharacter20x20.shape[0] - (centerRows + largeSize//2) ## por cada positivo quitar una columna
    ColsLeft = centerCols - largeSize//2 ##
    ColsRight = mainCharacter20x20.shape[1] - (centerCols + largeSize//2)
    arrayImage = np.reshape(mainCharacter20x20, mainCharacter20x20.shape)

    # arrayImage[:, left]=0
    # arrayImage[top,:] = 0
    # arrayImage[:, left+width] = 0
    # arrayImage[top+height, :] = 0

    while RowsTop != 0:
        if RowsTop > 0:
            arrayImage = np.delete(arrayImage, 0, axis=0)
            RowsTop -= 1
        else:
            arrayImage = np.append(np.ones((1,arrayImage.shape[1]))*255,arrayImage, axis=0)
            RowsTop += 1


    while RowsBot != 0:
        if RowsBot > 0:
            arrayImage = np.delete(arrayImage, arrayImage.shape[0]-1, axis=0)
            RowsBot -= 1
        else:
            arrayImage = np.append(arrayImage,np.ones((1,arrayImage.shape[1]))*255, axis=0)
            RowsBot += 1

    while ColsLeft != 0:
        if ColsLeft > 0:
            arrayImage = np.delete(arrayImage, 0, axis=1)
            ColsLeft -= 1
        else:
            arrayImage = np.append(np.ones((arrayImage.shape[0],1))*255, arrayImage, axis=1)
            ColsLeft += 1

    while ColsRight != 0:
        if ColsRight > 0:
            arrayImage = np.delete(arrayImage, arrayImage.shape[1]-1, axis=1)
            ColsRight -= 1
        else:
            arrayImage = np.append(arrayImage, np.ones((arrayImage.shape[0],1))*255, axis=1)
            ColsRight += 1

    #arrayImage[arrayImage<150]=0
    #arrayImage[arrayImage >= 150] = 255
    #plt.subplot(1, 2, 1), plt.imshow(image,'gray'), plt.title('image Input')
    #plt.subplot(1, 2, 2), plt.imshow(arrayImage,'gray'), plt.title('32x32')
    #plt.show()

    return arrayImage

    #row = np.ones((1, imSize)) * 255
    #col = np.ones((imSize, 1)) * 255
        #opH_A[int(round(centroides[label][1])), int(round(centroides[label][0]))] = 255
        # connectedCompoentWidth = labelStats[label, cv2.CC_STAT_WIDTH]
        # connectedCompoentHeight = labelStats[label, cv2.CC_STAT_HEIGHT]

        # area = labelStats[label, cv2.CC_STAT_AREA]
        # print(area, connectedCompoentHeight*connectedCompoentHeight, connectedCompoentHeight, connectedCompoentWidth)
        # if abs(connectedCompoentHeight - connectedCompoentWidth) < 5 \
        #        and connectedCompoentWidth * connectedCompoentHeight * 0.6 < area:
        #    cosl.append((int(round(centroides[label][0])), int(round(centroides[label][1])),  # ,
        #       connectedCompoentWidth, connectedCompoentHeight))
        # min(connectedCompoentHeight , connectedCompoentWidth)])
        #    edgesLength.append(min(connectedCompoentHeight, connectedCompoentWidth) // 2)
        ###########

def myImResize_forDataTraining(image_original, imSize):
    image_original = np.uint8(image_original)

    arrayImage = np.reshape(image_original, image_original.shape)
    image = extractImportantSubset(arrayImage)
    scale = smallSize * min(1.0 / image.shape[0],
                            1.0 / image.shape[1])

    mainCharacter20x20 = cv2.resize(image, None, fx=scale, fy=scale, interpolation= cv2.INTER_CUBIC)
    centerRows = mainCharacter20x20.shape[0]//2
    centerCols = mainCharacter20x20.shape[1]//2
    RowsTop = centerRows - largeSize // 2  # por cada positivo, quitar una fila, por cada negativo agregar
    RowsBot = mainCharacter20x20.shape[0] - (centerRows + largeSize // 2)  ## por cada positivo quitar una columna
    ColsLeft = centerCols - largeSize // 2  ##
    ColsRight = mainCharacter20x20.shape[1] - (centerCols + largeSize // 2)
    arrayImage = np.reshape(mainCharacter20x20, mainCharacter20x20.shape)

    # arrayImage[:, left]=0
    # arrayImage[top,:] = 0
    # arrayImage[:, left+width] = 0
    # arrayImage[top+height, :] = 0

    while RowsTop != 0:
        if RowsTop > 0:
            arrayImage = np.delete(arrayImage, 0, axis=0)
            RowsTop -= 1
        else:
            arrayImage = np.append(np.ones((1, arrayImage.shape[1])) * 255, arrayImage, axis=0)
            RowsTop += 1

    while RowsBot != 0:
        if RowsBot > 0:
            arrayImage = np.delete(arrayImage, arrayImage.shape[0] - 1, axis=0)
            RowsBot -= 1
        else:
            arrayImage = np.append(arrayImage, np.ones((1, arrayImage.shape[1])) * 255, axis=0)
            RowsBot += 1

    while ColsLeft != 0:
        if ColsLeft > 0:
            arrayImage = np.delete(arrayImage, 0, axis=1)
            ColsLeft -= 1
        else:
            arrayImage = np.append(np.ones((arrayImage.shape[0], 1)) * 255, arrayImage, axis=1)
            ColsLeft += 1

    while ColsRight != 0:
        if ColsRight > 0:
            arrayImage = np.delete(arrayImage, arrayImage.shape[1] - 1, axis=1)
            ColsRight -= 1
        else:
            arrayImage = np.append(arrayImage, np.ones((arrayImage.shape[0], 1)) * 255, axis=1)
            ColsRight += 1

    #arrayImage[arrayImage<200]=0
    #arrayImage[arrayImage >= 200] = 255
    #plt.subplot(1, 3, 1), plt.imshow(image_original, 'gray'), plt.title('image original')
    #plt.subplot(1, 3, 2), plt.imshow(image, 'gray'), plt.title('important data')
    #plt.subplot(1, 3, 3), plt.imshow(arrayImage, 'gray'), plt.title('32x32 Data Training')
    #plt.show()
    return arrayImage


def myImResize(image, imSize):
    arrayImage = np.reshape(image, image.shape)
    image = extractImportantSubset(arrayImage)
    sh = calcShapeToResize(image.shape,imSize=imSize)

    image = imresize(image, sh)
    arrayImage = np.reshape(image, image.shape)
    sh_final = arrayImage.shape
    if(max(sh_final)!=imSize):
        print('shape resulting: ', sh_final,' expected: ',sh)
    top = False
    row = np.ones((1, imSize))*255
    col = np.ones((imSize,1))*255
    while (arrayImage.shape)[0] < imSize:
        if top:
            arrayImage = np.vstack([row, arrayImage])
        else:
            arrayImage = np.vstack([arrayImage,row])
        top = not top
    while (arrayImage.shape)[1] < imSize:
        if top:
            arrayImage = np.append(arrayImage, col, axis=1)
        else:
            arrayImage = np.append(col, arrayImage, axis=1)
        top = not top

    return arrayImage


def maybe_pickle(data_folders, pickleFolder, min_num_images_per_class, force=False, percentToValid = 20):
    file_names_set = []
    train_sizes=[]
    validation_sizes=[]
    countImages=0
    for folder in data_folders:
        basename = os.path.basename(folder)
        vch = ''
        if 'hsf' in basename:
            baseFolder= folder.replace('/'+basename, '')
            baseFolder = os.path.basename(baseFolder)
            vch = chr(hex2int(baseFolder[0]) * 16 + hex2int(baseFolder[1]))

            set_filename = os.path.join(pickleFolder, basename + '_'+baseFolder+'_32x32.pickle')
        else:
            print(basename)
            vch = chr(hex2int(basename[6]) * 16 + hex2int(basename[7]))
            set_filename = os.path.join(pickleFolder, basename + '_32x32.pickle')

        if ord('A') <= ord(vch)  <= ord('Z'):
            file_names_set.append(set_filename)

            if os.path.exists(set_filename) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename)
                V= len(os.listdir(folder))
                print('files on: ', vch, ' : ', V)
                countImages += V
            else:
                print('Pickling %s.' % set_filename)
                dataset = load_letter(folder, min_num_images_per_class,max_num_images=100000)
                V = dataset.shape[0]
                countImages += V
                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e)

            toValid = (V * percentToValid)//100
            toTrain = V - toValid
            train_sizes.append(toTrain)
            validation_sizes.append(toValid)

    return [file_names_set,train_sizes,validation_sizes]


def maybe_pickle_digit(data_folders, pickleFolder, min_num_images_per_class, force=False, percentToValid = 20):
    file_names_set = []
    train_sizes=[]
    validation_sizes=[]
    countImages=0
    for folder in data_folders:
        basename = os.path.basename(folder)
        vch = ''
        if 'hsf' in basename:
            baseFolder= folder.replace('/'+basename, '')
            baseFolder = os.path.basename(baseFolder)
            vch = chr(hex2int(baseFolder[0]) * 16 + hex2int(baseFolder[1]))

            set_filename = os.path.join(pickleFolder, basename + '_'+baseFolder+'_32x32.pickle')
        else:
            print(basename)
            vch = chr(hex2int(basename[6]) * 16 + hex2int(basename[7]))
            set_filename = os.path.join(pickleFolder, basename + '_32x32.pickle')

        if ord('0') <= ord(vch) <= ord('9'):
            file_names_set.append(set_filename)

            if os.path.exists(set_filename) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename)
                V= len(os.listdir(folder))
                print('files on: ', vch, ' : ', V)
                countImages += V
            else:
                print('Pickling %s.' % set_filename)
                dataset = load_letter(folder, min_num_images_per_class,max_num_images=100000)
                V = dataset.shape[0]
                countImages += V
                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e)

            toValid = (V * percentToValid)//100
            toTrain = V - toValid
            train_sizes.append(toTrain)
            validation_sizes.append(toValid)

    return [file_names_set,train_sizes,validation_sizes]


def hex2int(hx):
    if 'a' <= hx <= 'f':
        return ord(hx)-ord('a')+10
    if 'A' <= hx <= 'F':
        return ord(hx)-ord('A')+10
    if '0' <= hx <= '9':
        return ord(hx) - ord('0') + 0

    raise Exception('hx is not defined for'+hx)


def hex2intMayuscula(hx):
    a = hex2int(hx[0])*16+hex2int(hx[1])
    if ord('A') <= a <= ord('Z'):
        return a - ord('A')
    raise Exception('no es mayuscula: '+chr(a))


def hex2intDigit(hx):
    a = hex2int(hx[0])*16+hex2int(hx[1])
    if ord('0') <= a <= ord('9'):
        return a - ord('0')
    raise Exception('no es digito: '+chr(a))


def getTrainFolder(baseFolder):
    folderClases = [x[0] for x in os.walk(baseFolder)]
    ans = []
    for folder in folderClases:
        basename = os.path.basename(folder)
        if 'train' in basename:
            ans = folder
    return ans


def getTestFolder(baseFolder):
    folderClases = [x[0] for x in os.walk(baseFolder)]
    ans = []
    for folder in folderClases:
        basename = os.path.basename(folder)
        if 'hsf_4' in basename:
            ans = folder
    return ans


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size_array, valid_size_array):
    print('number of files', len(pickle_files))

    valid_dataset, valid_labels = make_arrays(sum(valid_size_array), image_size)
    train_dataset, train_labels = make_arrays(sum(train_size_array), image_size)


    start_v, start_t = 0, 0

    end_v, end_t = valid_size_array[0], train_size_array[0]
    for label, pickle_file in enumerate(pickle_files):
        print(label,pickle_file)
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                #print(pickle_file, letter_set.shape)
                indxDot = pickle_file.index('.')
                vch = pickle_file[indxDot-8:indxDot-6]
                character = hex2intMayuscula(vch)
                print(vch, ' -> ', character)

                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:valid_size_array[label], :, :]
                    image_data = valid_letter[1, :, :]
                    plt.imshow(image_data, cmap=plt.cm.gray)
                    plt.show()
                    print(valid_size_array[label],' =? ', end_v,' - ',start_v)
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = character
                    start_v += valid_size_array[label]
                    if label + 1 < len(pickle_files):
                        end_v += valid_size_array[label+1]

                train_letter = letter_set[valid_size_array[label]:, :, :]
                #print(train_dataset.shape,' -- ',start_t,' --', end_t,' antes ',pickle_file)
                train_dataset[start_t:end_t, :, :] = train_letter

                train_labels[start_t:end_t] = character
                start_t += train_size_array[label]
                if label + 1 < len(pickle_files):
                    end_t += train_size_array[label + 1]
                #print(train_dataset.shape, ' -- ', start_t, ' --', end_t, ' fin ', pickle_file)
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


def merge_datasets_digit(pickle_files, train_size_array, valid_size_array):
    print('number of files', len(pickle_files))

    valid_dataset, valid_labels = make_arrays(sum(valid_size_array), image_size)
    train_dataset, train_labels = make_arrays(sum(train_size_array), image_size)


    start_v, start_t = 0, 0

    end_v, end_t = valid_size_array[0], train_size_array[0]
    for label, pickle_file in enumerate(pickle_files):
        print(label,pickle_file)
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                #print(pickle_file, letter_set.shape)
                indxDot = pickle_file.index('.')
                vch = pickle_file[indxDot-8:indxDot-6]
                character = hex2intDigit(vch)
                print(vch, ' -> ', character)

                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:valid_size_array[label], :, :]
                    image_data = valid_letter[1, :, :]
                    plt.imshow(image_data, cmap=plt.cm.gray)
                    plt.show()
                    print(valid_size_array[label],' =? ', end_v,' - ',start_v)
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = character
                    start_v += valid_size_array[label]
                    if label + 1 < len(pickle_files):
                        end_v += valid_size_array[label+1]

                train_letter = letter_set[valid_size_array[label]:, :, :]
                #print(train_dataset.shape,' -- ',start_t,' --', end_t,' antes ',pickle_file)
                train_dataset[start_t:end_t, :, :] = train_letter

                train_labels[start_t:end_t] = character
                start_t += train_size_array[label]
                if label + 1 < len(pickle_files):
                    end_t += train_size_array[label + 1]
                #print(train_dataset.shape, ' -- ', start_t, ' --', end_t, ' fin ', pickle_file)
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


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
    [train_datasets,trainSizes, validSizes] = maybe_pickle(trainFoldersTarget, pickleFolder, 10, force=True, percentToValid=5) # 1000


    [test_datasets, trainSizes_t, validSizes_t] = maybe_pickle(testFoldersTarget, testPickleFolder, 10, force=True, percentToValid=0) #

    print(sum(trainSizes),sum(validSizes))
    print(sum(trainSizes_t), sum(validSizes_t))


    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, trainSizes, validSizes)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, trainSizes_t,validSizes_t)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

    pickle_file = os.path.join(data_root, pickleFile)

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
