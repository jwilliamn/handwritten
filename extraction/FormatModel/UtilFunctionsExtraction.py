#!/usr/bin/env python
# coding: utf-8

"""
    Extraction.UtilFunctions
    =============================

    Classes, .......

    _copyright_ = 'Copyright (c) 2017 Vm.C.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import numpy as np
import cv2
import modeling
from matplotlib import pyplot as plt


dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

def expandOnlyIntersections(BinaryOriginal, globalMask):
    V = np.zeros(BinaryOriginal.shape, dtype=bool)
    intersectionOfMask = cv2.bitwise_and(BinaryOriginal, globalMask)
    Q = [(1, 1)]*(BinaryOriginal.shape[0]*BinaryOriginal.shape[0])
    indxHead = 0
    indxTail = 0
    for i in range(BinaryOriginal.shape[0]):
        for j in range(BinaryOriginal.shape[1]):
            if intersectionOfMask[i,j] > 0:
                Q[indxTail] = (i,j)
                indxTail += 1
                V[i,j] = True
                #intersectionOfMask[i,j] = 0
    intersectionOfMask[intersectionOfMask>=0] = 0 #OFF , 0 es ON
    while indxHead < indxTail:
        (i,j) = Q[indxHead]
        indxHead += 1
        intersectionOfMask[i,j] = 255
        for k in range(0,4):
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

def closestNonZero(img, p, maxSize = 21):

    if img[p[0],p[1]]>0:
        return p
    if img[p[0],p[1]] > 0:
        return p
    dx_step=[-1, 0,1,0]
    dy_step=[0, 1, 0, -1]
    currentK = 0
    copyP = p
    for k in range(1,maxSize):
        for twice in range(0,2):
            for times in range(0,k):
                p = (p[0]+dx_step[currentK]), (p[1] + dy_step[currentK])
                if p[0]>=0 and p[1]>=0 and p[0]< img.shape[0] and p[1]< img.shape[1] and img[p[0], p[1]] > 0:
                    #print('found: ', p)
                    return p
            currentK = (currentK + 1) % 4
    return copyP

def filterSingleCharacter(letter_original_and_mask):
    # todavia puede tener los bordes
    letter_original = letter_original_and_mask[0]
    mask = letter_original_and_mask[1]
    img = letter_original.copy()
    #se va a crear una nueva mascara, pero mask, el parametro deberia ser util

    ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows, cols = th3.shape

    #th3 tiene el character correcto, pero tambien tiene basura, solo se extendera los que hagan match
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
    #for i in L.shape[0]:
    #    for j in L.shape[1]:
    #        if L[i,j]>0:
    #            Ifp[i,j] = Ifp_union[i,j]
    # Ifp[L>0]=Ifp_union[L>0]
    onlyMatch = expandOnlyIntersections(th3, mask)


    #newMask = th3.copy()
    #for i in range(0,rows):
    #    for j in range(0, cols):
    #        if shouldFill(th3, (i,j)):
    #            th3[i,j]=255
    #        if shouldClear(th3, (i, j)):
    #            th3[i, j] = 0
    #plt.subplot(1,2,1), plt.imshow(newMask,'gray'), plt.title('newMask (th original)')
    #plt.subplot(1, 2, 2), plt.imshow(th3,'gray'), plt.title('th3, modified')
    #plt.show()

    newMask = cv2.medianBlur(th3, 3)


    img_copy = img.copy()

    img_copy[onlyMatch < 125] = 255
    img_copy[img_copy< 255] = 0
    img_copy[img_copy >= 255] = 255

    if cv2.countNonZero(onlyMatch) == 0:
        img_copy[mask >= 0] = 255

    try:

        #imgResult = GenerateData.myImResize_20x20_32x32(img_copy)
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
        #print('error filtering: ', e)
        imgResult = None
    
    
    #plt.subplot(1, 5, 1), plt.imshow(img, 'gray'), plt.title('img original')
    #plt.subplot(1, 5, 2), plt.imshow(mask, 'gray'), plt.title('mask from input')
    #plt.subplot(1, 5, 3), plt.imshow(onlyMatch, 'gray'), plt.title('mask, mezclado')
    #plt.subplot(1, 5, 4), plt.imshow(img_copy, 'gray'), plt.title('to pass resize32x32')
    #if( imgResult is not None):
    #    plt.subplot(1, 5, 5), plt.imshow(imgResult, 'gray'), plt.title('imgResult resized to 32x32')
    #plt.show()
    
    return imgResult

def findApropiateTemplate(ratio):

    current_image = None
    bestRatio = 100
    for k in range(1,5):
        img = cv2.imread('extraction/FormatModel/cuadro_template_'+str(k)+'.png', 0)
        current_ratio = img.shape[0]/img.shape[1]

        if abs(current_ratio-ratio) < abs(bestRatio-ratio):
            bestRatio = current_ratio
            current_image = img
            # print('best ratio: '+str(k))
    return current_image

def plotImagesWithPrediction(preditectArray, images):
    cols = len(images)
    for k in range(0,cols):
        if images[k] is not None:
            plt.subplot(1,cols,k+1), plt.imshow(images[k],'gray'), plt.title(preditectArray[k]), plt.axis('off')
    plt.show()
def findMaxElement(A):
    currentValue = -1.0
    currentI = 0
    currentJ = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] > currentValue:
                currentValue = A[i,j]
                currentI = i
                currentJ = j
    return (currentI, currentJ)

def getBestRectangle(region):
    A = range(region.shape[0] - 10, region.shape[0])
    B = range(region.shape[1] - 10, region.shape[1])
    copia = region.copy()

    copia[copia >= 0] = 0
    rows,cols = region.shape
    print(region.shape)
    bestValue = -1.0
    bestA = 0
    bestB = 0
    bestPos = (-1,-1)
    for a in A:
        for b in B:
            cum = np.zeros((cols,rows))
            for i in range(rows):
                if i+a >= rows:
                    break

                for j in range(cols):
                    if j+b >= cols:
                        break
                    copia[copia >= 0] = 0
                    pi = (j, i)
                    pf = (j+b, i+a)
                    cv2.rectangle(copia, pi, pf, 255, thickness=1)
                    copia = cv2.bitwise_and(copia, region)
                    cantMatch = cv2.countNonZero(copia)
                    cum[i,j]=cantMatch/(2*(a+b))
                    # (I,J) = findMaxElement(cum)
                    # print(I,J)
                    # print('inicio', i, j)
                    # print('longitudes', a, b)
                    # print(cum)
                    # plt.subplot(1, 2, 1), plt.imshow(region, 'gray'), plt.title('region')
                    # plt.subplot(1, 2, 2), plt.imshow(copia, 'gray'), plt.title('copia con rect de 255')
                    # plt.show()
            (I, J) = findMaxElement(cum)
            if cum[I,J] > bestValue:
                bestValue = cum[I,J]
                bestA = a
                bestB = b
                bestPos = (I,J)

    # copia[copia >= 0] = 0
    # pi = (bestPos[1], bestPos[0])
    # pf = (bestPos[1] + bestB, bestPos[0] + bestA)
    # cv2.rectangle(copia, pi, pf, 255, thickness=1)
    # plt.subplot(1, 2, 1), plt.imshow(region, 'gray'), plt.title('region')
    # plt.subplot(1, 2, 2), plt.imshow(copia, 'gray'), plt.title('copia con rect de 255')
    # plt.show()
    return bestPos,(bestPos[0]+bestB, bestPos[1]+bestA)




def extractCharacters(img, onlyUserMarks, TL, BR, count):
    numRows = (BR[0]-TL[0]) / count
    numCols = BR[1] - TL[1]
    # print('finding ratio nr/nc : ' + str(numRows)+' / ' + str(numCols)+'  divided by '+ str(count))
    template = findApropiateTemplate(numRows/numCols)
    ROI = img[TL[1] - 3:BR[1] + 3, TL[0] - 3:BR[0] + 3]
    ROI_onlyUserMarks = onlyUserMarks[TL[1] - 3:BR[1] + 3, TL[0] - 3:BR[0] + 3]

    If = cv2.GaussianBlur(ROI, (3, 3), 0)
    If = cv2.adaptiveThreshold(If, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    If = cv2.bitwise_not(If)



    leftPart = If[:, 0:(template.shape[1]+5)]
    rightPart = If[:, -(template.shape[1] + 5):]

    top_left_L,bottom_right_L = getBestRectangle(leftPart)
    delta_L = (bottom_right_L[0]-top_left_L[0], bottom_right_L[1]-top_left_L[1])
    # res = cv2.matchTemplate(leftPart,template,cv2.TM_CCORR)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left_L = min_loc
    # bottom_right_L = (top_left_L[0] + template.shape[1], top_left_L[1] + template.shape[0])

    top_left_R, bottom_right_R = getBestRectangle(leftPart)
    delta_R = (bottom_right_R[0] - top_left_R[0], bottom_right_R[1] - top_left_R[1])
    # res = cv2.matchTemplate(rightPart, template, cv2.TM_CCORR)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left_R = min_loc
    # bottom_right_R = (top_left_R[0] + template.shape[1], top_left_R[1] + template.shape[0])
    #
    #
    # print('bottom_right_R', bottom_right_R)

    bestLeft = leftPart[top_left_L[1]:bottom_right_L[1], top_left_L[0]:bottom_right_L[0]]
    bestRight = rightPart[top_left_R[1]:bottom_right_R[1], top_left_R[0]:bottom_right_R[0]]

    # print('If shape: ', If.shape)
    # print('template shape: ', template.shape)
    # print('current top_left_R', top_left_R)
    top_left_R = (top_left_R[0]+If.shape[1]-(template.shape[1]+5), top_left_R[1])
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
    pointA = (top_left_L[1],top_left_L[0])
    pointY = (bottom_right_R[1],bottom_right_R[0])

    pointB = (pointY[0],pointA[1])
    pointX = (pointA[0],pointY[1])


    # print(pointA,pointB,pointX,pointY,ROI.shape)
    ROI_2 = ROI[pointA[0]:pointY[0], pointA[1]:pointY[1]]
    # plt.subplot(2,1,1), plt.imshow(ROI)
    # plt.subplot(2, 1, 2), plt.imshow(ROI_2)
    # plt.show()
    letters = []
    for k in range(0,count):
        upperLeft = getPointProportion(pointA, pointX, k, count - k)
        bottomLeft = getPointProportion(pointB, pointY, k, count - k )
        upperRight = getPointProportion(pointA, pointX,  k + 1, count - (k + 1))
        bottomRight = getPointProportion(pointB, pointY, k + 1, count - (k + 1))

        minX = min(upperLeft[0],bottomLeft[0])+2
        maxX = max(upperRight[0], bottomRight[0])-2

        minY = min(bottomLeft[1], bottomRight[1])+2
        maxY = max(upperLeft[1], upperRight[1])-2

        singleCharacter = (ROI[minX:maxX, minY:maxY], ROI_onlyUserMarks[minX:maxX, minY:maxY])
        letters.append(singleCharacter)


    filteredLetters = []

    for letter in letters:
        singleLetterFiltered = filterSingleCharacter(letter)
        filteredLetters.append(singleLetterFiltered)
    return filteredLetters


def extractCharacters_old(img, onlyUserMarks, TL, BR, count):
    letters = []

    ROI = img[TL[1]-3:BR[1]+3,TL[0]-3:BR[0]+3]
    ROI_onlyUserMarks = onlyUserMarks[TL[1]-3:BR[1]+3,TL[0]-3:BR[0]+3]
    #ROI = cv2.medianBlur(ROI, 3)
    If = cv2.GaussianBlur(ROI, (3, 3), 0)
    #If = cv2.Canny(ROI,50,200)


    If = cv2.adaptiveThreshold(If, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    If = cv2.bitwise_not(If)

    dst = cv2.cornerHarris(If, 2, 3, 0.04)
    #dst = cv2.dilate(dst, None)

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
    If_copy=If.copy()
    If_copy[If_copy>=125]=125
    If[If>=0] = 0

    list2 = res[:, 2]
    for indx, x  in enumerate(res[:,3]):
        y = list2[indx]
        if 0 <= x < If.shape[0] and 0<= y < If.shape[1]:
            If[x, y] = 255
            If_copy[x, y] = 255
    #print('looking for A,B,X,Y')
    pointA = closestNonZero(If, (3,3), 12)
    pointB = closestNonZero(If, (ROI.shape[0]-3,3), 12)
    pointX = closestNonZero(If, (3, ROI.shape[1]-3), 12)
    pointY = closestNonZero(If, (ROI.shape[0]-3, ROI.shape[1]-3), 12)

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

    for k in range(0,count):
        upperLeft = getPointProportion(pointA, pointX, k, count - k)

        #if k ==0:
        #    upperLeft = getCross(edges, 9, upperLeft)
        #else:
        #    upperLeft = getCross(edges, 13, upperLeft)

        bottomLeft = getPointProportion(pointB, pointY, k, count - k )

        #if k == 0:
        #    bottomLeft = getCross(edges, 3, bottomLeft)
        #else:
        #    bottomLeft = getCross(edges, 7, bottomLeft)

        upperRight = getPointProportion(pointA, pointX,  k + 1, count - (k + 1))

        #if k == cant-1:
        #    upperRight = getCross(edges, 12, upperRight)
        #else:
        #    upperRight = getCross(edges, 13, upperRight)

        bottomRight = getPointProportion(pointB, pointY, k + 1, count - (k + 1))

        #if k == cant-1:
        #    bottomRight = getCross(edges, 6, bottomRight)
        #else:
        #    bottomRight = getCross(edges, 7, bottomRight)

        minX = min(upperLeft[0],bottomLeft[0])+2
        maxX = max(upperRight[0], bottomRight[0])-2

        minY = min(bottomLeft[1], bottomRight[1])+2
        maxY = max(upperLeft[1], upperRight[1])-2

        singleCharacter = (ROI[minX:maxX, minY:maxY], ROI_onlyUserMarks[minX:maxX, minY:maxY])
        letters.append(singleCharacter)


    filteredLetters = []

    for letter in letters:
        singleLetterFiltered = filterSingleCharacter(letter)
        filteredLetters.append(singleLetterFiltered)
    return filteredLetters
