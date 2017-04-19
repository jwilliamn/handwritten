#!/usr/bin/env python
# coding: utf-8

"""
    Extraction.PageDetector
    ============================
    Detect page number and oritentation to extract features
    properly.

    _copyright_ = 'Copyright (c) 2017 Vm.C.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

from extraction import FeatureExtractor


# Global settings ####
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


# Function definition ####
def sortSquareCenters(L):
    if len(L) != 4:
        print('wtf, there are not 4 centers: ', len(L))
    L = sorted(L, key=lambda x: x[0] + x[1])
    return L


def getSquares(image_original):
    """Detecta la orientación de la imagen y la orienta.
    Args:
        image_original: Original image.
    Returns:
        cosl: ....
    """
    ret3, th3 = cv2.threshold(image_original, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    thIMor = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, se)

    maxShape = max(image_original.shape)
    toleranciaCuadrado = int(round(20*maxShape/1750))

    print('ToleranciaCuadrado: ',toleranciaCuadrado)
    k_left = int(round(5*maxShape/1750)) # componentes >= 4
    k_right = int(round(80*maxShape/1750)) # componentes < 4

    while k_left + +1 < k_right:
        k = (k_left+k_right)//2
        onlySquares = cv2.erode(thIMor, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=1)
        onlySquares = cv2.dilate(onlySquares, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=1)
        # Connected compoent labling
        stats = cv2.connectedComponentsWithStats(onlySquares, connectivity=8)
        print('with ', k, ' we got: ', stats[0])
        if stats[0] >= 5:
            k_left = k
        else:
            k_right = k
        #plt.imshow(onlySquares), plt.title(str(k)+'::'+str(stats[0]))
        #plt.show()

    print('k_left final: ', k_left)
    thIMor = cv2.erode(thIMor, cv2.getStructuringElement(cv2.MORPH_RECT, (k_left, k_left)), iterations=1)
    thIMor = cv2.dilate(thIMor, cv2.getStructuringElement(cv2.MORPH_RECT, (k_left, k_left)), iterations=1)
    # Connected component labling
    stats = cv2.connectedComponentsWithStats(thIMor, connectivity=8)
    num_labels = stats[0]
    print('num labels:', num_labels)
    labels = stats[1]
    labelStats = stats[2]
    centroides = stats[3]
    # We expect the connected component of the numbers to be more or less with a constats ratio
    # So we find the medina ratio of all the comeonets because the majorty of connected compoent are numbers
    cosl = []
    edgesLength = []
    for label in range(num_labels):
        connectedCompoentWidth = labelStats[label, cv2.CC_STAT_WIDTH]
        connectedCompoentHeight = labelStats[label, cv2.CC_STAT_HEIGHT]
        area = labelStats[label,cv2.CC_STAT_AREA]
        print(area, connectedCompoentHeight*connectedCompoentHeight, connectedCompoentHeight, connectedCompoentWidth)
        if abs(connectedCompoentHeight-connectedCompoentWidth) < toleranciaCuadrado \
                and connectedCompoentWidth*connectedCompoentHeight * 0.6 < area:
            cosl.append((int(round(centroides[label][0])), int(round(centroides[label][1])),  #,
                         connectedCompoentWidth, connectedCompoentHeight))
                                         # min(connectedCompoentHeight , connectedCompoentWidth)])
            edgesLength.append(min(connectedCompoentHeight,connectedCompoentWidth)//2)

    delta = sum(edgesLength)//4
    cosl = sortSquareCenters(cosl)
    cosl[0] = (cosl[0][0] - cosl[0][2]//2, cosl[0][1] - cosl[0][3]//2)
    if cosl[1][0] < cosl[1][1]:
        cosl[1] = (cosl[1][0] - cosl[1][2]//2, cosl[1][1] + cosl[1][3]//2)
        cosl[2] = (cosl[2][0] + cosl[2][2]//2, cosl[2][1] - cosl[2][3]//2)
    else:
        cosl[1] = (cosl[1][0] + cosl[1][2]//2, cosl[1][1] - cosl[1][3]//2)
        cosl[2] = (cosl[2][0] - cosl[2][2]//2, cosl[2][1] + cosl[2][3]//2)

    cosl[3] = (cosl[3][0] + cosl[3][2]//2, cosl[3][1] + cosl[3][3]//2)
    print('cosl',cosl)

    #plt.subplot(1, 3, 1), plt.imshow(image_original,'gray')
    #plt.subplot(1, 3, 2), plt.imshow(th3, 'gray')
    #plt.subplot(1, 3, 3), plt.imshow(thIMor, 'gray')

    #plt.show()
    return cosl


def enderezarImagen(image):
    """Detecta la orientación de la imagen y la orienta.
    Args:
        image: Original image.
    Returns:
        rotatedImg: Rotated image.
    """
    squaresCenters = getSquares(image)
    if len(squaresCenters) != 4:
        raise Exception('There is no 4 centers')
    print('First squares: ', squaresCenters)
    dI = squaresCenters[2][0] - squaresCenters[0][0]
    dJ = squaresCenters[2][1] - squaresCenters[0][1]
    theta = np.arctan2(dJ, dI)
    thetaDegrees = theta*180/np.pi
    (oldY, oldX) = image.shape
    M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=thetaDegrees, scale=1)  # rotate about center of image.
    newX, newY = oldX * 1, oldY * 1
    r = np.deg2rad(thetaDegrees)
    newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # Find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty

    rotatedImg = cv2.warpAffine(image, M, dsize=(int(newX), int(newY)))
    return rotatedImg


def getCenterZone(img, center, delta):
    """ .........
    Args:
        img: Original image.
        center: Center of ....
        delta: .....
    Returns:
        centerZone: .........
    """
    K = img[center[1] - delta:center[1] + delta, center[0] - delta:center[0] + delta].copy()
    # K = img[squaresCenters[0][1]:squaresCenters[3][1],squaresCenters[0][0]:squaresCenters[3][0]].copy()
    #ret3, th3 = cv2.threshold(K, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret3, th3 = cv2.threshold(K, 240, 255, cv2.THRESH_BINARY_INV)
    K = cv2.dilate(th3, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
    centerZone = cv2.resize(K,(750,750))
    centerZone[centerZone>125] = 255
    centerZone[centerZone <= 125] = 0
    return centerZone

def getPercentMatched(baseImage,testImage):
    if baseImage.shape != testImage.shape:
        print('shapeBase ', baseImage.shape)
        print('testBase ', testImage.shape)
        return 0
    onlyMatched = cv2.bitwise_and(baseImage, testImage)
    nonZeroDB = cv2.countNonZero(baseImage)
    nonZeroAND = cv2.countNonZero(onlyMatched)
    return nonZeroAND / nonZeroDB


def percentPage1Normal(centerZone):
    centerZone_Base = cv2.imread('resources/centerZone_page1.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)

def percentPage2Normal(centerZone):
    centerZone_Base = cv2.imread('resources/centerZone_page2.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)

def percentPage3Normal(centerZone):
    centerZone_Base = cv2.imread('resources/centerZone_page3.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)

def percentPage4Normal(centerZone):
    centerZone_Base = cv2.imread('resources/centerZone_page4.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)


def percentPage1Inversa(centerZone):
    centerZone_Base = cv2.imread('resources/centerZone_page1inv.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)

def percentPage2Inversa(centerZone):
    centerZone_Base = cv2.imread('resources/centerZone_page2inv.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)

def percentPage3Inversa(centerZone):
    centerZone_Base = cv2.imread('resources/centerZone_page3inv.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)

def percentPage4Inversa(centerZone):
    centerZone_Base = cv2.imread('resources/centerZone_page4inv.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)


def detectPage(img):
    """Detecta el número de la página.
    Args:
        img: Imagen orientada.
    Returns:
        _: Rotated image.
    """
    squaresCenters = getSquares(img)
    print('Second squares: ', squaresCenters)
    if len(squaresCenters) != 4:
        raise Exception('There is no 4 centers')

    #supuestamente esta horizontal
    distCols = squaresCenters[3][1]-squaresCenters[0][1]
    distRows = squaresCenters[3][0]-squaresCenters[0][0]



    print('Ratio: ', (distRows/distCols))
    sumX = 0
    sumY = 0
    for sqCenters in squaresCenters:
        sumX += sqCenters[0]
        sumY += sqCenters[1]

    delta = distRows//4
    center = (sumX//4, sumY//4)

    centerZone = getCenterZone(img, center, delta)

    detectors = []
    percent = []
    page = []
    detectors.append(percentPage1Normal)
    page.append((1, 0))
    detectors.append(percentPage2Normal)
    page.append((2, 0))
    detectors.append(percentPage3Normal)
    page.append((3, 0))
    detectors.append(percentPage4Normal)
    page.append((4, 0))

    detectors.append(percentPage1Inversa)
    page.append((1, 1))
    detectors.append(percentPage2Inversa)
    page.append((2, 1))
    detectors.append(percentPage3Inversa)
    page.append((3, 1))
    detectors.append(percentPage4Inversa)
    page.append((4, 1))

    for detector in detectors:
        percent.append(detector(centerZone))
    print('Percents: ',percent)
    max_indx, max_value = max(enumerate(percent), key=lambda p: p[1])
    newImage = img[squaresCenters[0][1]:squaresCenters[3][1], squaresCenters[0][0]:squaresCenters[3][0]]
    return (newImage,page[max_indx])
