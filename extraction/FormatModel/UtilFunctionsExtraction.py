#!/usr/bin/env python
# coding: utf-8

"""
    Extraction.UtilFunctions
    =============================

    Classes, .......

    _copyright_ = 'Copyright (c) 2017 Vm.C.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""
import statistics

import numpy as np
import cv2
import modeling
from matplotlib import pyplot as plt

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def expandOnlyIntersections(BinaryOriginal, globalMask):
    V = np.zeros(BinaryOriginal.shape, dtype=bool)
    intersectionOfMask = cv2.bitwise_and(BinaryOriginal, globalMask)
    Q = [(1, 1)] * (BinaryOriginal.shape[0] * BinaryOriginal.shape[0])
    indxHead = 0
    indxTail = 0
    for i in range(BinaryOriginal.shape[0]):
        for j in range(BinaryOriginal.shape[1]):
            if intersectionOfMask[i, j] > 0:
                Q[indxTail] = (i, j)
                indxTail += 1
                V[i, j] = True
                # intersectionOfMask[i,j] = 0
    intersectionOfMask[intersectionOfMask >= 0] = 0  # OFF , 0 es ON
    while indxHead < indxTail:
        (i, j) = Q[indxHead]
        indxHead += 1
        intersectionOfMask[i, j] = 255
        for k in range(0, 4):
            ni = i + dx[k]
            nj = j + dy[k]
            if 0 <= ni < BinaryOriginal.shape[0] and 0 <= nj < BinaryOriginal.shape[1]:
                if BinaryOriginal[ni, nj] > 0 and not V[ni, nj]:
                    Q[indxTail] = (ni, nj)
                    indxTail += 1
                    V[ni, nj] = True
    return intersectionOfMask


def getPointProportion(A, B, a, b):
    px = (A[0] * b + B[0] * a) / (a + b)
    py = (A[1] * b + B[1] * a) / (a + b)
    return (int(px), int(py))


def closestNonZero(img, p, maxSize=21):
    if img[p[0], p[1]] > 0:
        return p
    if img[p[0], p[1]] > 0:
        return p
    dx_step = [-1, 0, 1, 0]
    dy_step = [0, 1, 0, -1]
    currentK = 0
    copyP = p
    for k in range(1, maxSize):
        for twice in range(0, 2):
            for times in range(0, k):
                p = (p[0] + dx_step[currentK]), (p[1] + dy_step[currentK])
                if p[0] >= 0 and p[1] >= 0 and p[0] < img.shape[0] and p[1] < img.shape[1] and img[p[0], p[1]] > 0:
                    # print('found: ', p)
                    return p
            currentK = (currentK + 1) % 4
    return copyP


def filterSingleCharacter_new(letter_original_and_mask):
    letter_original = letter_original_and_mask[0]
    mask = letter_original_and_mask[1]
    img = letter_original.copy()
    img[img > 230] = 255
    ret3, resaltado = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows, cols = resaltado.shape

    # creating border
    borde = resaltado.copy()
    borde[borde >= 0] = 255
    gb = 2  # grosor del borde
    borde[gb:borde.shape[0] - gb, gb:borde.shape[1] - gb] = 0

    pppb = cv2.bitwise_and(resaltado, borde)  # posible_pre_printed_borders
    wob = cv2.bitwise_and(resaltado, cv2.bitwise_not(borde))  # sin borders
    dilatado = cv2.dilate(wob, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
    pdof = cv2.bitwise_and(dilatado, pppb)  # parte de la letra borrada
    result = cv2.bitwise_or(wob, pdof)

    onlyMatch = cv2.morphologyEx(result, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    # onlyMatch = expandOnlyIntersections(result, mask)
    onlyMatch = result

    stats = cv2.connectedComponentsWithStats(onlyMatch, connectivity=8)
    num_labels = stats[0]
    labels = stats[1]
    labelStats = stats[2]
    centroides = stats[3]
    # We expect the connected component of the numbers to be more or less with a constats ratio
    # So we find the medina ratio of all the comeonets because the majorty of connected compoent are numbers
    cosl = []
    edgesLength = []
    debugThisCharacter = False
    print('Letter')
    for label in range(num_labels):
        Width = labelStats[label, cv2.CC_STAT_WIDTH]
        Height = labelStats[label, cv2.CC_STAT_HEIGHT]
        area = labelStats[label, cv2.CC_STAT_AREA]
        # print(area, connectedCompoentHeight * connectedCompoentHeight, connectedCompoentHeight, connectedCompoentWidth)
        deleteGroup = False
        if centroides[label][0] < 0.1 * onlyMatch.shape[1] or \
                        centroides[label][1] < 0.1 * onlyMatch.shape[0] or \
                        centroides[label][0] > 0.9 * onlyMatch.shape[1] or \
                        centroides[label][1] > 0.9 * onlyMatch.shape[0]:
            if Width * Height * 0.5 < area:
                deleteGroup = True

        if area < 4:
            deleteGroup = True

        if deleteGroup:
            Left = labelStats[label, cv2.CC_STAT_LEFT]
            Top = labelStats[label, cv2.CC_STAT_TOP]
            Right = Left + Width
            Bottom = Top + Height
            onlyMatch[Top:Bottom, Left:Right] = 0
            debugThisCharacter = False
            print('centroide: ', centroides[label])

    img_copy = img.copy()

    img_copy[onlyMatch < 125] = 255  # lo que es negro en only match, ahora sera blanco en img_copy
    img_copy[onlyMatch >= 125] = 0  # to_do lo que no es absolutamente blanco, pasa a ser negro

    if cv2.countNonZero(onlyMatch) == 0:
        img_copy[mask >= 0] = 255

    try:

        # imgResult = GenerateData.myImResize_20x20_32x32(img_copy)
        imgResult = modeling.GenerateTrainDataAZ.myImResize_forDataTraining(img_copy, None)

        # for i in range(0,32):
        #     for j in range(0, 32):
        #         if shouldFill(imgResult, (i,j)):
        #             imgResult[i, j] = 0
        #         if shouldClear(imgResult, (i, j)):
        #             imgResult[i, j] = 255

        imgResult = (imgResult -
                     255.0 / 2) / 255.0

    except Exception as e:
        # print('error filtering: ', e)
        imgResult = None

    if debugThisCharacter:
        plt.subplot(1, 7, 1), plt.imshow(img, 'gray'), plt.title('Original')
        plt.subplot(1, 7, 2), plt.imshow(resaltado, 'gray'), plt.title('Resaltado')
        plt.subplot(1, 7, 3), plt.imshow(borde, 'gray'), plt.title('Borde')
        plt.subplot(1, 7, 4), plt.imshow(result, 'gray'), plt.title('SinBordes')
        plt.subplot(1, 7, 5), plt.imshow(mask, 'gray'), plt.title('Mask')
        plt.subplot(1, 7, 6), plt.imshow(img_copy, 'gray'), plt.title('To myImResize')
        if imgResult is not None:
            plt.subplot(1, 7, 7), plt.imshow(imgResult, 'gray'), plt.title('imgResult resized to 32x32')
        plt.show()

    return imgResult


def filterSingleCharacter(letter_original_and_mask):
    # todavia puede tener los bordes
    letter_original = letter_original_and_mask[0]
    mask = letter_original_and_mask[1]
    img = letter_original.copy()
    # se va a crear una nueva mascara, pero mask, el parametro deberia ser util

    ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows, cols = th3.shape

    # th3 tiene el character correcto, pero tambien tiene basura, solo se extendera los que hagan match
    #
    # SEh = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    # SEv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    # opHorizontal = cv2.morphologyEx(th3, cv2.MORPH_OPEN, SEh)
    # opVertical = cv2.morphologyEx(th3, cv2.MORPH_OPEN, SEv)

    # Ifp = cv2.bitwise_xor(th3, cv2.bitwise_or(opHorizontal, opVertical))

    # Dk = getDk()
    # Ifp_union = Ifp.copy()
    # Ifp_union[:,:]=0
    # for D in Dk:
    #     closeD = cv2.morphologyEx(Ifp, cv2.MORPH_CLOSE, D)
    #     Ifp_union = cv2.bitwise_or(Ifp_union, closeD)
    # L = cv2.bitwise_or(opHorizontal, opVertical)
    # for i in L.shape[0]:
    #    for j in L.shape[1]:
    #        if L[i,j]>0:
    #            Ifp[i,j] = Ifp_union[i,j]
    # Ifp[L>0]=Ifp_union[L>0]
    onlyMatch = expandOnlyIntersections(th3, mask)

    # newMask = th3.copy()
    # for i in range(0,rows):
    #    for j in range(0, cols):
    #        if shouldFill(th3, (i,j)):
    #            th3[i,j]=255
    #        if shouldClear(th3, (i, j)):
    #            th3[i, j] = 0

    # plt.subplot(1,2,1), plt.imshow(newMask,'gray'), plt.title('newMask (th original)')
    # plt.subplot(1, 2, 2), plt.imshow(th3,'gray'), plt.title('th3, modified')
    # plt.show()

    newMask = cv2.medianBlur(th3, 3)

    img_copy = img.copy()

    img_copy[onlyMatch < 125] = 255
    img_copy[img_copy < 255] = 0
    img_copy[img_copy >= 255] = 255

    if cv2.countNonZero(onlyMatch) == 0:
        img_copy[mask >= 0] = 255

    try:

        # imgResult = GenerateData.myImResize_20x20_32x32(img_copy)
        imgResult = modeling.GenerateTrainDataAZ.myImResize_20x20_32x32(img_copy)

        # for i in range(0,32):
        #     for j in range(0, 32):
        #         if shouldFill(imgResult, (i,j)):
        #             imgResult[i, j] = 0
        #         if shouldClear(imgResult, (i, j)):
        #             imgResult[i, j] = 255

        imgResult = (imgResult -
                     255.0 / 2) / 255.0
    except Exception as e:
        # print('error filtering: ', e)
        imgResult = None

    # plt.subplot(1, 5, 1), plt.imshow(img, 'gray'), plt.title('img original')
    # plt.subplot(1, 5, 2), plt.imshow(mask, 'gray'), plt.title('mask from input')
    # plt.subplot(1, 5, 3), plt.imshow(onlyMatch, 'gray'), plt.title('mask, mezclado')
    # plt.subplot(1, 5, 4), plt.imshow(img_copy, 'gray'), plt.title('to pass resize32x32')
    # if( imgResult is not None):
    #    plt.subplot(1, 5, 5), plt.imshow(imgResult, 'gray'), plt.title('imgResult resized to 32x32')
    # plt.show()

    return imgResult


def findApropiateTemplate(ratio):
    current_image = None
    bestRatio = 100
    for k in range(1, 5):
        img = cv2.imread('extraction/FormatModel/cuadro_template_' + str(k) + '.png', 0)
        current_ratio = img.shape[0] / img.shape[1]

        if abs(current_ratio - ratio) < abs(bestRatio - ratio):
            bestRatio = current_ratio
            current_image = img
            # print('best ratio: '+str(k))
    return current_image


def plotImagesWithPrediction(preditectArray, images):
    cols = len(images)
    for k in range(0, cols):
        if images[k] is not None:
            plt.subplot(1, cols, k + 1), plt.imshow(images[k], 'gray'), plt.title(preditectArray[k]), plt.axis('off')
    plt.show()


def findMaxElement(A):
    currentValue = -1.0
    currentI = 0
    currentJ = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] > currentValue:
                currentValue = A[i, j]
                currentI = i
                currentJ = j
    return (currentI, currentJ)


def countNonZerosRows(sumRows, I, j1, j2):
    if j1 <= 0:
        return sumRows[I, j2]
    else:
        return sumRows[I, j2] - sumRows[I, (j1 - 1)]


def countNonZerosCols(sumCols, J, i1, i2):
    if i1 <= 0:
        return sumCols[i2, J]
    else:
        return sumCols[i2, J] - sumCols[(i1 - 1), J]


def countNonZeros(sumRows, sumCols, pi, pf):
    top = countNonZerosRows(sumRows, pi[0], pi[1], pf[1])
    bottom = countNonZerosRows(sumRows, pf[0], pi[1], pf[1])
    left = countNonZerosCols(sumCols, pi[1], pi[0] + 1, pf[0] - 1)
    right = countNonZerosCols(sumCols, pf[1], pi[0] + 1, pf[0] - 1)
    return top + bottom + left + right


def getFirstGroupLargerThan(a, L):
    b = np.zeros(a.shape)

    for i in range(1, len(b)):
        if a[i] > 0:
            b[i] = b[i - 1] + 1
        else:
            if b[i - 1] >= L:
                # print(b,' - ', i)
                return int(round(i - b[i - 1])), int(i)
    return -1, -1


def getBestRectangle_big(region, ratio_cols_over_rows):
    B = range(max(0, region.shape[1] - 30), region.shape[1])
    copia = region.copy()

    copia[copia >= 0] = 0
    rows, cols = region.shape
    print(region.shape)
    bestValue = -1.0
    bestA = 0
    bestB = 0
    bestPos = (-1, -1)

    sumCols = np.asarray(np.sum(region, 0) / 255.0)
    diffSumCols = np.ediff1d(sumCols)
    sortedCols = np.sort(sumCols)
    indexToCutCols = (cols - 2 * 22) // 2
    toCut = 2 * sortedCols[indexToCutCols] + 2
    sumCols[sumCols < toCut] = 0
    left, right = getFirstGroupLargerThan(sumCols, 15)
    if left + right < 0:
        return (0, 0), (20, 20)
    sumCols[:left] = 0
    sumCols[(right + 1):] = 0
    print(left, right)
    importanColum = region[:, left:right]
    sumRows = np.asarray(np.sum(importanColum, 1) / 255.0)
    diffSumRows = np.ediff1d(sumRows)
    sortedRows = np.sort(sumRows)
    indexToCutRows = (rows - 7 * 16) // 2
    valueToCut = 2 * sortedRows[indexToCutRows] + 2
    sumRows[sumRows < valueToCut] = 0

    for i in range(0, 7):
        left, right = getFirstGroupLargerThan(sumRows, 10)
        if right - left > 0:
            marca = sumRows[left:right]
            sortedMarca = np.sort(marca)
            print(i, ' : ', sortedMarca[len(sortedMarca) // 2])
            sumRows[left:right] = 0

    width = 1

    #
    #
    # max_indx_row, max_value_row = max(enumerate(sumRows), key=lambda p: p[1])
    # max_indx_col, max_value_col = max(enumerate(sumCols), key=lambda p: p[1])
    #
    # for b in B:
    #     minA = int(round(b/(ratio_cols_over_rows+0.1)))
    #     maxA = int(round(b/(ratio_cols_over_rows-0.1)))
    #     for a in range(minA,maxA):
    #         cum = np.zeros((rows, cols))
    #         for i in range(rows):
    #             if i + a >= rows:
    #                 break
    #             for j in range(cols):
    #                 if j + b >= cols:
    #                     break
    #                 #copia[copia >= 0] = 0
    #                 # pi = (j, i)
    #                 # pf = (j + b, i + a)
    #                 # cv2.rectangle(copia, pi, pf, 255, thickness=1)
    #                 # copia = cv2.bitwise_and(copia, region)
    #                 # cantMatch = cv2.countNonZero(copia)
    #                 # print('cant match: ', cantMatch)
    #                 myCantMatch = countNonZeros(sumRows, sumCols, (i, j), (i+a, j+b))
    #                 # print('cant mAtch: ', myCantMatch)
    #                 cum[i, j] = myCantMatch / (2 * (a + b))
    #                 # (I, J) = findMaxElement(cum)
    #                 # print(I,J)
    #                 # print('inicio', i, j)
    #                 # print('longitudes', a, b)
    #                 # print(cum)
    #                 # plt.subplot(1, 2, 1), plt.imshow(region, 'gray'), plt.title('region')
    #                 # plt.subplot(1, 2, 2), plt.imshow(copia, 'gray'), plt.title('copia con rect de 255')
    #                 # plt.show()
    #                 # cv2.rectangle(copia, pi, pf, 0, thickness=1)
    #
    #
    #         (I, J) = findMaxElement(cum)
    #         if cum[I, J] > bestValue:
    #             bestValue = cum[I, J]
    #             bestA = a
    #             bestB = b
    #             bestPos = (I, J)
    #
    # copia[copia >= 0] = 0
    # pi = (bestPos[1], bestPos[0])
    # pf = (bestPos[1] + bestB, bestPos[0] + bestA)
    # cv2.rectangle(copia, pi, pf, 255, thickness=1)
    # section = region[pi[1]:pf[1], pi[0]:pf[0]]
    # plt.subplot(1, 3, 1), plt.imshow(region, 'gray'), plt.title('region')
    # plt.subplot(1, 3, 2), plt.imshow(copia, 'gray'), plt.title('copia con rect de 255')
    # plt.subplot(1, 3, 3), plt.imshow(section, 'gray'), plt.title('best mark')
    plt.subplot(4, 2, 1), plt.imshow(region, 'gray')
    plt.subplot(4, 2, 2), plt.imshow(importanColum, 'gray')

    plt.subplot(4, 2, 3), plt.bar(range(len(sumRows)), sumRows, width, color="blue")
    plt.subplot(4, 2, 4), plt.bar(range(len(sumCols)), sumCols, width, color="blue")
    plt.subplot(4, 2, 5), plt.bar(range(len(diffSumRows)), diffSumRows, width, color="blue")
    plt.subplot(4, 2, 6), plt.bar(range(len(diffSumCols)), diffSumCols, width, color="blue")
    plt.subplot(4, 2, 7), plt.bar(range(len(sortedRows)), sortedRows, width, color="blue")
    plt.subplot(4, 2, 8), plt.bar(range(len(sortedCols)), sortedCols, width, color="blue")

    plt.show()
    return (0, 0), (20, 20)


def getBestRectangle(region, ratio_cols_over_rows):
    B = range(max(0, region.shape[1] - 20), region.shape[1])
    copia = region.copy()

    copia[copia >= 0] = 0
    rows, cols = region.shape
    print(region.shape)
    bestValue = -1.0
    bestA = 0
    bestB = 0
    bestPos = (-1, -1)

    sumRows = np.zeros((rows, cols))
    sumCols = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if j == 0:
                sumRows[i, j] = (1 if region[i, j] > 0 else 0)
            else:
                sumRows[i, j] = (1 if region[i, j] > 0 else 0) + sumRows[i, j - 1]

            if i == 0:
                sumCols[i, j] = (1 if region[i, j] > 0 else 0)
            else:
                sumCols[i, j] = (1 if region[i, j] > 0 else 0) + sumCols[i - 1, j]
    # print(region)
    # print(sumRows)
    # print(sumCols)


    for b in B:
        minA = int(round(b / (ratio_cols_over_rows + 0.1)))
        maxA = int(round(b / (ratio_cols_over_rows - 0.1)))
        for a in range(minA, maxA):
            cum = np.zeros((rows, cols))
            for i in range(rows):
                if i + a >= rows:
                    break
                for j in range(cols):
                    if j + b >= cols:
                        break
                    # copia[copia >= 0] = 0
                    # pi = (j, i)
                    # pf = (j + b, i + a)
                    # cv2.rectangle(copia, pi, pf, 255, thickness=1)
                    # copia = cv2.bitwise_and(copia, region)
                    # cantMatch = cv2.countNonZero(copia)
                    # print('cant match: ', cantMatch)
                    myCantMatch = countNonZeros(sumRows, sumCols, (i, j), (i + a, j + b))
                    # print('cant mAtch: ', myCantMatch)
                    cum[i, j] = myCantMatch / (2 * (a + b))
                    # (I, J) = findMaxElement(cum)
                    # print(I,J)
                    # print('inicio', i, j)
                    # print('longitudes', a, b)
                    # print(cum)
                    # plt.subplot(1, 2, 1), plt.imshow(region, 'gray'), plt.title('region')
                    # plt.subplot(1, 2, 2), plt.imshow(copia, 'gray'), plt.title('copia con rect de 255')
                    # plt.show()
                    # cv2.rectangle(copia, pi, pf, 0, thickness=1)

            (I, J) = findMaxElement(cum)
            if cum[I, J] > bestValue:
                bestValue = cum[I, J]
                bestA = a
                bestB = b
                bestPos = (I, J)

    copia[copia >= 0] = 0
    pi = (bestPos[1], bestPos[0])
    pf = (bestPos[1] + bestB, bestPos[0] + bestA)
    # cv2.rectangle(copia, pi, pf, 255, thickness=1)
    # section = region[pi[1]:pf[1], pi[0]:pf[0]]
    # plt.subplot(1, 3, 1), plt.imshow(region, 'gray'), plt.title('region')
    # plt.subplot(1, 3, 2), plt.imshow(copia, 'gray'), plt.title('copia con rect de 255')
    # plt.subplot(1, 3, 3), plt.imshow(section, 'gray'), plt.title('best mark')
    # plt.show()
    return pi, pf


def predictCategoric_simple_column_squared(column, labels, sz=12):
    sumRows = np.asarray(np.sum(column, 1) // 255)
    resp = extractLabelsBySquares(column, sumRows, labels)
    return resp


def predictCategoric_simple_column(column, labels, sz=12):
    sumRows = np.asarray(np.sum(column, 1) // 255)

    rows = len(sumRows)
    sumRows = sumRows - int(min(sumRows))
    sumRows = dropMinsTo0(sumRows, 9)
    sr = sumRows.copy()
    left = 0
    right = max(sumRows)

    toCut = -1
    for i in range(left, right):
        sc = sumRows.copy()
        sc[sc < i] = 0
        print('testing with ', i)
        numBlocks = countBlocks(sc, sz)

        if numBlocks == len(labels):
            toCut = i
            break

    sumRows[sumRows < toCut] = 0
    sr_c = sumRows.copy()
    results = ''
    for i in range(0, len(labels)):
        left, right = getFirstGroupLargerThan(sumRows, sz-2)
        if right - left > 0 and results is not None:
            marca = sumRows[left:right]
            sortedMarca = np.sort(marca)
            if sortedMarca[len(sortedMarca) // 2] > 15:
                if len(results) > 0:
                    results = results + ';' + labels[i]
                else:
                    results = labels[i]

            sumRows[left:right] = 0
        else:
            results = None

    if results is None or len(results) == 0:
        results = '?'

    # print(labels)
    # print(results)
    #
    # plt.subplot(1, 3, 1), plt.imshow(column)
    # plt.subplot(1, 3, 2), plt.bar(range(len(sr)), sr, 1)
    # plt.subplot(1, 3, 3), plt.bar(range(len(sr_c)), sr_c, 1)
    # plt.show()
    print(results)
    return results
    #return results


def predictValuesCategory_simple(arrayOfImages, array_labels, sz=12):
    if len(arrayOfImages) != len(array_labels):
        print('error: sz arrayOfImages != sz arrayLabels')
        return ['None'] * len(array_labels)

    result = []
    for indx, img in enumerate(arrayOfImages):
        if img is not None:
            labels = array_labels[indx]
            if sz <= 7:
                predictions = predictCategoric_simple_column(img, labels, sz)
            else:
                predictions = predictCategoric_simple_column_squared(img, labels, sz)
            if predictions is not None and result is not None:
                result.append(predictions)
            else:
                result = None
        else:
            print('wtf is none')

    if result is None:
        result = ['Unknow'] * len(arrayOfImages)
    print('Returning:', result)
    return result


def countBlocks(a, sz):
    count = 0
    while True:
        left, right = getFirstGroupLargerThan(a, sz)
        if left + right < 0:
            return count
        count += 1
        print(left, right)
        a[left:right] = 0
        if left == right:
            a[left] = 0


def calcMeans(a, limit, iterations=1):
    n = len(a)
    medias = np.zeros(n)
    count_items = 0
    acum = []

    for i in range(n):
        acum.append(a[i])
        count_items += 1
        if count_items > limit:
            acum = acum[1:]
            count_items -= 1
        if count_items == limit:
            medias[i - limit // 2] = statistics.median(acum)
    if iterations > 1:
        return calcMeans(medias, limit, iterations - 1)
    return medias


def dropMinsTo0(a, limit):
    n = len(a)
    medias = np.zeros(n, dtype=np.int32)
    count_items = 0
    acum = []

    for i in range(n):
        acum.append(a[i])
        count_items += 1
        if count_items > limit:
            acum = acum[1:]
            count_items -= 1
        if count_items == limit:
            j = i - limit // 2
            val = a[j]
            if abs(val - min(acum)) < 2 and val * 3 < max(acum[0:2]) and val * 3 < max(acum[-2:]):
                medias[j] = 0
            else:
                medias[j] = val

    return medias
def extractLabelsBySquares(column, sumRows, labels):
    cantRows = len(labels)

    sumRows = sumRows.copy()
    originalRows = sumRows.copy()
    sumRows[sumRows < int(max(sumRows))] = 0
    center = []
    sc = sumRows.copy()
    while True:
        left, right = getFirstGroupLargerThan(sumRows, 1)

        if left + right < 0:
            break
        sumRows[left:right] = 0
        if left == right:
            sumRows[left] = 0

        center.append((left + right) // 2)

    if len(center)<2 or center[0]+150 > center[-1]:
        return '?'
    i = center[0] + 17

    results = ''
    for k in range(cantRows):
        j = i + 19*k
        marca = originalRows[(j-3):(j+3)]
        sortedMarca = np.sort(marca)
        if min(marca) > 12:
            if len(results) > 0:
                results = results + ';' + labels[k]
            else:
                results = labels[k]

    if len(results) == 0:
        results = '?'
    return results
    # print(labels)
    # print(results)
    # plt.subplot(1,2,1), plt.imshow(column,'gray')
    # plt.subplot(1,2,2), plt.bar(range(len(originalRows)),originalRows, 1)
    # plt.show()





def extractColumnsBySquares(If, cantColumns):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    If = cv2.dilate(If, kernel)
    sumCols = np.asarray(np.sum(If, 0) // 255)
    sumCols = sumCols.copy()

    sumCols = sumCols - int(min(sumCols))
    sumCols = dropMinsTo0(sumCols, 3)
    left = 0
    right = max(sumCols)
    if cantColumns == 2:
        delta = 60
    else:
        delta = 45

    sumCols[sumCols < 125] = 0
    sc = sumCols.copy()
    blocks = countBlocks(sumCols, 1)
    arrayResult = []
    if blocks == 2:


        sumCols = sc.copy()
        returnNone = False

        center = []
        for k in range(0, 2):
            left, right = getFirstGroupLargerThan(sumCols, 1)
            print('LR', left, right)
            if left + right < 0:
                print('this shouldn be happening')
                returnNone = True
                continue
            sumCols[left:right] = 0
            if left == right:
                sumCols[left] = 0

            center.append((left+right)//2)

            #importanColum = If[:, left:right]
            #arrayResult.append(importanColum)


        if returnNone or len(center) < 2:
            return [None] * cantColumns

        if cantColumns == 2:
            i  = getPointProportion((center[0],0),(center[1],0), 20,47)[0]
            j = getPointProportion((center[0], 0), (center[1], 0), 47, 20)[0]
            importanColum_I = If[:, (i-11):(i+11)]
            importanColum_J = If[:, (j - 11):(j + 11)]
            arrayResult.append(importanColum_I)
            arrayResult.append(importanColum_J)
            # plt.subplot(1, 3, 1), plt.imshow(If), plt.title('If')
            # plt.subplot(1, 3, 2), plt.bar(range(len(sumCols)), sumCols, 1),plt.title('SumCols')
            # plt.subplot(1, 3, 3), plt.imshow(importanColum_I), plt.title('FIrst Column')
            # plt.show()

        else:
            i = (center[0]+center[1])//2
            importanColum_I = If[:, (i - 11):(i + 11)]
            # plt.subplot(1, 3, 1), plt.imshow(If), plt.title('If')
            # plt.subplot(1, 3, 2), plt.imshow(importanColum_I), plt.title('Column')
            # plt.show()
            arrayResult.append(importanColum_I)

        return arrayResult

    else:
        return [None]*cantColumns





def extractCategory_simpleImages(img, TL, BR, cantColumns):
    deltaAmpliacion = 10
    ROI = img[TL[1] - deltaAmpliacion:BR[1] + deltaAmpliacion, TL[0] - deltaAmpliacion:BR[0] + deltaAmpliacion]
    # If = cv2.GaussianBlur(ROI, (3, 3), 0)
    ret, If = cv2.threshold(ROI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    If = cv2.bitwise_not(If)
    rows, cols = If.shape


    resp = extractColumnsBySquares(If, cantColumns)

    sumCols = np.asarray(np.sum(If, 0) // 255)
    sumCols = dropMinsTo0(sumCols, 11)

    left = 0
    right = max(sumCols)

    toCut = -1
    for i in range(left, right):
        sc = sumCols.copy()
        sc[sc < i] = 0
        print('testing with ', i)
        numBlocks = countBlocks(sc, 20)

        if numBlocks == cantColumns:
            toCut = i
            break
    #
    # media_21 = calcMeans(sumCols, 3,3)
    # with0 = dropMinsTo0(sumCols, 11)
    # sum_medias = np.zeros(len(media_21))
    # for i in range(len(sum_medias)):
    #     sum_medias[i] = with0[i]+media_21[i]
    # plt.subplot(2, 2, 1), plt.imshow(If, 'gray')
    # plt.subplot(2, 2, 2), plt.bar(range(len(sumCols)), sumCols, 1, color="blue")
    # plt.subplot(2, 2, 3), plt.bar(range(len(media_21)), media_21, 1, color="blue")
    # plt.subplot(2, 2, 4), plt.bar(range(len(with0)), with0, 1, color="blue")
    # plt.show()
    sumCols[sumCols < toCut] = 0

    arrayResult = []
    arResultOk = True
    for k in range(0, cantColumns):
        left, right = getFirstGroupLargerThan(sumCols, 15)
        if left + right < 0:
            arrayResult.append(None)
            arResultOk = False
            continue

        sumCols[left:right] = 0
        importanColum = If[:, left:right]
        arrayResult.append(importanColum)
    print('Returning images A')
    toPrint = False
    for I in arrayResult:
        if I is None:
            toPrint = False
    # if arResultOk == False:
    #     arrayResult = resp
    if toPrint:

        sumCols = np.asarray(np.sum(If, 0) // 255)
        plt.subplot(3, 2, 1), plt.imshow(If, 'gray')
        plt.subplot(3, 2, 2), plt.bar(range(len(sumCols)), sumCols, 1, color="blue")
        if resp is not None and len(resp) > 0 and resp[0] is not None:
            plt.subplot(3, 2, 3), plt.imshow(resp[0], 'gray')
            sumRows = np.asarray(np.sum(resp[0], 1) // 255)
            plt.subplot(3, 2, 4), plt.bar(range(len(sumRows)), sumRows, 1, color="blue")
        if resp is not None and len(resp) > 1 and resp[1] is not None:
            plt.subplot(3, 2, 5), plt.imshow(resp[1], 'gray')
            sumRows = np.asarray(np.sum(resp[1], 1) // 255)
            plt.subplot(3, 2, 6), plt.bar(range(len(sumRows)), sumRows, 1, color="blue")
        plt.show()
    return resp


def extractCategory_singleColumn(img, TL, BR, cantColumns):
    deltaAmpliacion = 10
    ROI = img[TL[1] - deltaAmpliacion:BR[1] + deltaAmpliacion, TL[0] - deltaAmpliacion:BR[0] + deltaAmpliacion]
    # If = cv2.GaussianBlur(ROI, (3, 3), 0)
    ret, If = cv2.threshold(ROI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    If = cv2.bitwise_not(If)
    arrayResult = []
    arrayResult.append(If)

    return arrayResult


def extractCategory_Test(img, TL, BR):
    deltaAmpliacion = 10
    ROI = img[TL[1] - deltaAmpliacion:BR[1] + deltaAmpliacion, TL[0] - deltaAmpliacion:BR[0] + deltaAmpliacion]
    # If = cv2.GaussianBlur(ROI, (3, 3), 0)
    ret, If = cv2.threshold(ROI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    If = cv2.bitwise_not(If)
    top_left, bottom_right = getBestRectangle_big(If, 0.4)
    bestRectangle = If[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    canny = cv2.Canny(If, 50, 240)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    plt.subplot(1, 4, 1), plt.imshow(ROI), plt.title('original')
    plt.subplot(1, 4, 2), plt.imshow(If), plt.title('If')
    plt.subplot(1, 4, 3), plt.imshow(canny), plt.title('canny on If')
    plt.subplot(1, 4, 4), plt.imshow(bestRectangle), plt.title('bestRectangle')

    plt.show()

    return None


def extractCharacters(img, onlyUserMarks, TL, BR, count):
    numRows = (BR[0] - TL[0]) / count
    numCols = BR[1] - TL[1]
    # print('finding ratio nr/nc : ' + str(numRows)+' / ' + str(numCols)+'  divided by '+ str(count))
    template = findApropiateTemplate(numRows / numCols)

    deltaAmpliacion = 5

    ROI = img[TL[1] - deltaAmpliacion:BR[1] + deltaAmpliacion, TL[0] - deltaAmpliacion:BR[0] + deltaAmpliacion]
    ROI_onlyUserMarks = onlyUserMarks[TL[1] - deltaAmpliacion:BR[1] + deltaAmpliacion,
                        TL[0] - deltaAmpliacion:BR[0] + deltaAmpliacion]

    If = cv2.GaussianBlur(ROI, (3, 3), 0)
    If = cv2.adaptiveThreshold(If, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    If = cv2.bitwise_not(If)

    leftPart = If[:, 0:(template.shape[1] + (deltaAmpliacion * 2 - 1))]
    rightPart = If[:, -(template.shape[1] + (deltaAmpliacion * 2 - 1)):]

    top_left_L, bottom_right_L = getBestRectangle(leftPart, 0.8)
    delta_L = (bottom_right_L[0] - top_left_L[0], bottom_right_L[1] - top_left_L[1])

    top_left_R, bottom_right_R = getBestRectangle(rightPart, 0.8)
    delta_R = (bottom_right_R[0] - top_left_R[0], bottom_right_R[1] - top_left_R[1])

    bestLeft = leftPart[top_left_L[1]:bottom_right_L[1], top_left_L[0]:bottom_right_L[0]]
    bestRight = rightPart[top_left_R[1]:bottom_right_R[1], top_left_R[0]:bottom_right_R[0]]

    # print('If shape: ', If.shape)
    # print('template shape: ', template.shape)
    # print('current top_left_R', top_left_R)
    top_left_R = (top_left_R[0] + If.shape[1] - (template.shape[1] + ((deltaAmpliacion * 2 - 1))), top_left_R[1])
    bottom_right_R = (top_left_R[0] + delta_R[0], top_left_R[1] + delta_R[1])
    # print('after top_left_R', top_left_R)
    possibleBestLeft = If[top_left_L[1]:bottom_right_L[1], top_left_L[0]:bottom_right_L[0]]
    possibleBestRight = If[top_left_R[1]:bottom_right_R[1], top_left_R[0]:bottom_right_R[0]]

    # plt.subplot(1,8,1), plt.imshow(If)
    # plt.subplot(1, 8, 2), plt.imshow(template)
    # plt.subplot(1, 8, 3), plt.imshow(leftPart)
    # plt.subplot(1, 8, 4), plt.imshow(rightPart)
    # plt.subplot(1, 8, 5), plt.imshow(bestLeft)
    # plt.subplot(1, 8, 6), plt.imshow(possibleBestLeft)
    # plt.subplot(1, 8, 7), plt.imshow(bestRight)
    # plt.subplot(1, 8, 8), plt.imshow(possibleBestRight)
    # plt.show()

    pointA = (top_left_L[1], top_left_L[0])
    pointY = (bottom_right_R[1], bottom_right_R[0])

    pointB = (pointY[0], pointA[1])
    pointX = (pointA[0], pointY[1])

    # print(pointA,pointB,pointX,pointY,ROI.shape)
    ROI_2 = ROI[pointA[0]:pointY[0], pointA[1]:pointY[1]]

    # plt.subplot(2,1,1), plt.imshow(ROI)
    # plt.subplot(2, 1, 2), plt.imshow(ROI_2)
    # plt.show()

    letters = []
    for k in range(0, count):
        upperLeft = getPointProportion(pointA, pointX, k, count - k)
        bottomLeft = getPointProportion(pointB, pointY, k, count - k)
        upperRight = getPointProportion(pointA, pointX, k + 1, count - (k + 1))
        bottomRight = getPointProportion(pointB, pointY, k + 1, count - (k + 1))

        minX = min(upperLeft[0], bottomLeft[0])
        maxX = max(upperRight[0], bottomRight[0])

        minY = min(bottomLeft[1], bottomRight[1])
        maxY = max(upperLeft[1], upperRight[1])

        singleCharacter = (ROI[minX:maxX, minY:maxY], ROI_onlyUserMarks[minX:maxX, minY:maxY])
        letters.append(singleCharacter)

    filteredLetters = []

    for letter in letters:
        singleLetterFiltered = filterSingleCharacter_new(letter)
        filteredLetters.append(singleLetterFiltered)

        # if singleLetterFiltered != None:
        #    plt.imshow(singleLetterFiltered)
        #    plt.show()

    return filteredLetters


def extractCharacters_old(img, onlyUserMarks, TL, BR, count):
    letters = []

    ROI = img[TL[1] - 3:BR[1] + 3, TL[0] - 3:BR[0] + 3]
    ROI_onlyUserMarks = onlyUserMarks[TL[1] - 3:BR[1] + 3, TL[0] - 3:BR[0] + 3]
    # ROI = cv2.medianBlur(ROI, 3)
    If = cv2.GaussianBlur(ROI, (3, 3), 0)
    # If = cv2.Canny(ROI,50,200)


    If = cv2.adaptiveThreshold(If, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    If = cv2.bitwise_not(If)

    dst = cv2.cornerHarris(If, 2, 3, 0.04)
    # dst = cv2.dilate(dst, None)

    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(If, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    If_copy = If.copy()
    If_copy[If_copy >= 125] = 125
    If[If >= 0] = 0

    list2 = res[:, 2]
    for indx, x in enumerate(res[:, 3]):
        y = list2[indx]
        if 0 <= x < If.shape[0] and 0 <= y < If.shape[1]:
            If[x, y] = 255
            If_copy[x, y] = 255
    # print('looking for A,B,X,Y')
    pointA = closestNonZero(If, (3, 3), 12)
    pointB = closestNonZero(If, (ROI.shape[0] - 3, 3), 12)
    pointX = closestNonZero(If, (3, ROI.shape[1] - 3), 12)
    pointY = closestNonZero(If, (ROI.shape[0] - 3, ROI.shape[1] - 3), 12)

    #
    # #ret3, If = cv2.threshold(ROI, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # #If = cv2.dilate(If, cv2.getStructuringElement(cv2.MORPH_RECT, (3,1)), iterations=1)
    # D9 = cv2.getStructuringElement(cv2.MORPH_CROSS, (15,15))
    # D9[1:,1:] = 0
    #
    # #D9[0:8, :] = 0
    # print(D9)
    # ROI_Open_D9 = cv2.morphologyEx(If, cv2.MORPH_ERODE, D9)

    # plt.subplot(3,1,1), plt.imshow(ROI), plt.title('ROI')
    # plt.subplot(3,1,2), plt.imshow(If), plt.title('If')
    # plt.subplot(3, 1, 3), plt.imshow(If_copy), plt.title('If_copy')
    # plt.show()

    for k in range(0, count):
        upperLeft = getPointProportion(pointA, pointX, k, count - k)

        # if k ==0:
        #    upperLeft = getCross(edges, 9, upperLeft)
        # else:
        #    upperLeft = getCross(edges, 13, upperLeft)

        bottomLeft = getPointProportion(pointB, pointY, k, count - k)

        # if k == 0:
        #    bottomLeft = getCross(edges, 3, bottomLeft)
        # else:
        #    bottomLeft = getCross(edges, 7, bottomLeft)

        upperRight = getPointProportion(pointA, pointX, k + 1, count - (k + 1))

        # if k == cant-1:
        #    upperRight = getCross(edges, 12, upperRight)
        # else:
        #    upperRight = getCross(edges, 13, upperRight)

        bottomRight = getPointProportion(pointB, pointY, k + 1, count - (k + 1))

        # if k == cant-1:
        #    bottomRight = getCross(edges, 6, bottomRight)
        # else:
        #    bottomRight = getCross(edges, 7, bottomRight)

        minX = min(upperLeft[0], bottomLeft[0]) + 2
        maxX = max(upperRight[0], bottomRight[0]) - 2

        minY = min(bottomLeft[1], bottomRight[1]) + 2
        maxY = max(upperLeft[1], upperRight[1]) - 2

        singleCharacter = (ROI[minX:maxX, minY:maxY], ROI_onlyUserMarks[minX:maxX, minY:maxY])
        letters.append(singleCharacter)

    filteredLetters = []

    for letter in letters:
        singleLetterFiltered = filterSingleCharacter(letter)
        filteredLetters.append(singleLetterFiltered)
    return filteredLetters
