import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

from extraction import FeatureExtractor

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def getPageNumber(image):
    image = enderezarImagen(image)
    return (image, 3)


def completeWithOnlyMeans(M, i, j):
    ans = (i, j, 1)
    M[i, j] = False
    for k in range(0, 4):
        ni = i + dx[k]
        nj = j + dy[k]
        if 0 <= ni < M.shape[0] and 0 <= nj < M.shape[1] and M[ni, nj]:
            b = completeWithOnlyMeans(M, ni, nj)
            ans = (ans[0] + b[0], ans[1] + b[1], ans[2] + b[2])
    return ans


def distPoints(A, B):
    deltaX = (A[0] - B[0])
    deltaY = (A[1] - B[1])
    return math.sqrt(deltaX * deltaX + deltaY * deltaY)


def isSquare(L, current=None):
    d = []

    if current is None:
        for i in range(0, 4):
            for j in range(i + 1, 4):
                d.append(distPoints(L[i], L[j]))
    else:
        if len(current) != 4:
            return False
        # en verdad calcula si es un rectangul
        for i in range(0, 4):
            for j in range(i + 1, 4):
                d.append(distPoints(L[current[i]], L[current[j]]))
    d.sort()
    for i in range(0, 6, 2):
        if abs(d[i] - d[i + 1]) > 5:
            return -1
    if (d[0] * 4 < d[2]):
        return -1

    return d[2]


def findDistancesOfSquares(L):
    distances = []
    for i in range(0, len(L)):
        for j in range(i + 1, len(L)):
            for k in range(j + 1, len(L)):
                for m in range(k + 1, len(L)):
                    vertices = [L[t] for t in [i, j, k, m]]
                    if isSquare(vertices) > 0:
                        distances.append(isSquare(vertices))
    return distances


def findSquares(L, lengthSquare):
    V = []
    for i in range(0, len(L)):
        for j in range(i + 1, len(L)):
            for k in range(j + 1, len(L)):
                for m in range(k + 1, len(L)):
                    vertices = [L[t] for t in [i, j, k, m]]
                    if isSquare(vertices) > 0:
                        currentLengthSquare = isSquare(vertices)
                        if lengthSquare - 3 < currentLengthSquare < lengthSquare + 3:
                            V.append(vertices)
    return V


def dfs(L, V, i, d):
    V[i] = True
    K = [L[i]]
    for j in range(0, len(L)):
        if (not V[j]) and distPoints(L[j], L[i]) < d:
            K.extend(dfs(L, V, j, d))
    return K


def get4CenterOfSquare(L, image, d=None):
    if d is None:
        left = 1
        right = 800
        V = [True] * len(L)
        while left + 1 < right:
            m = (left + right) // 2
            for indx_v in range(0, len(V)):
                V[indx_v] = False

            counter = 0
            for i in range(0, len(L)):
                if not V[i]:
                    counter += 1
                    dfs(L, V, i, m)
            print('counter: ', counter, m)
            if counter > 4:
                left = m
            else:
                right = m

        print('dist: ', right)
        for indx_v in range(0, len(V)):
            V[indx_v] = False
        groups = []
        for i in range(0, len(L)):
            if not V[i]:
                groups.append(dfs(L, V, i, right))
        if len(groups) != 4:
            raise Exception('cant get 4 corners  COD: 301')
        print(len(groups))
        square_distances = []
        for L in groups:
            print(len(L))
            minI = min([a[0] for a in L])
            maxI = max([a[0] for a in L])
            minJ = min([a[1] for a in L])
            maxJ = max([a[1] for a in L])

            # space = image[minI:maxI, minJ:maxJ]
            # cv2.imshow('group', space)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if len(L) == 4:
                distanceSquare = isSquare(L)
                if distanceSquare > 0:
                    square_distances.append([distanceSquare])
                else:
                    raise Exception('One group doesnt have a square 303')

            else:
                distanceSquare = findDistancesOfSquares(L)
                if len(distanceSquare) == 0:
                    raise Exception('One group doesnt have a square 304')
                else:
                    square_distances.append(distanceSquare)

        print(square_distances)

        squareLength = []
        for distanceGroup in square_distances:
            for d in distanceGroup:
                counter = 0
                for t in square_distances:
                    for q in t:
                        if q - 5 < d < q + 5:
                            counter += 1
                            break
                if counter == 4:
                    squareLength.append(d)
        print(squareLength)
        SquareCenters = []
        if max(squareLength) - min(squareLength) > 5:
            raise Exception('there may be many squares')

        meanDistance = (max(squareLength) + min(squareLength)) / 2
        print(' length of square ', meanDistance)
        for L in groups:

            if len(L) == 4:
                sumI = 0
                sumJ = 0
                for (I, J) in L:
                    sumI += I
                    sumJ += J
                SquareCenters.append((int(round(sumI / 4)), int(round(sumJ / 4))))

            else:
                multipleSquares = findSquares(L, meanDistance)
                print('multiple squares: ', multipleSquares)
                centersOfSquareOfCorner = []
                for singleSquare in multipleSquares:
                    #print('single square', singleSquare)
                    sumI = 0
                    sumJ = 0
                    for (I, J) in singleSquare:
                        sumI += I
                        sumJ += J
                    centersOfSquareOfCorner.append((int(round(sumI / 4)), int(round(sumJ / 4))))

                if len(centersOfSquareOfCorner) == 1:
                    SquareCenters.extend(centersOfSquareOfCorner)
                else:
                    ordenada_of_centers = [center[0] for center in centersOfSquareOfCorner]
                    absis_of_centers = [center[1] for center in centersOfSquareOfCorner]
                    print('centers for this group: ', centersOfSquareOfCorner)
                    if max(ordenada_of_centers) - min(ordenada_of_centers) < 8 and max(absis_of_centers) - min(
                            absis_of_centers) < 8:
                        sumX = sum(ordenada_of_centers)
                        sumY = sum(absis_of_centers)
                        count_centers = len(ordenada_of_centers)
                        SquareCenters.append((sumX // count_centers, sumY // count_centers))

                    else:
                        raise Exception('many centers for this group 304')

        return SquareCenters

    else:
        return None


def sortSquareCenters(L):
    if len(L) != 4:
        print('wtf, there are not 4 centers: ', len(L))
    L = sorted(L, key=lambda x: x[0] + x[1])
    return L


def getSquares(image_original):
    # plt.hist(image.ravel(), 256, [0, 256]);
    # plt.show()

    #blur = cv2.GaussianBlur(image_original, (5, 5), 0)
    ret3, th3 = cv2.threshold(image_original, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    thIMor = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, se)

    k_left = 5 # componentes >=4
    k_right = 80 # componentes < 4

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

    print('k_left final: ', k_left)
    thIMor = cv2.erode(thIMor, cv2.getStructuringElement(cv2.MORPH_RECT, (k_left, k_left)), iterations=1)
    thIMor = cv2.dilate(thIMor, cv2.getStructuringElement(cv2.MORPH_RECT, (k_left, k_left)), iterations=1)
    # Connected compoent labling
    stats = cv2.connectedComponentsWithStats(thIMor, connectivity=8)
    num_labels = stats[0]
    print('num labels:', num_labels)
    labels = stats[1]
    labelStats = stats[2]
    centroides = stats[3]
    # We expect the conneccted compoennt of the numbers to be more or less with a constats ratio
    # So we find the medina ratio of all the comeonets because the majorty of connected compoent are numbers
    cosl = []
    edgesLength = []
    for label in range(num_labels):
        connectedCompoentWidth = labelStats[label, cv2.CC_STAT_WIDTH]
        connectedCompoentHeight = labelStats[label, cv2.CC_STAT_HEIGHT]
        area = labelStats[label,cv2.CC_STAT_AREA]
        #print(area, connectedCompoentHeight*connectedCompoentHeight, connectedCompoentHeight, connectedCompoentWidth)
        if abs(connectedCompoentHeight-connectedCompoentWidth) < 5 \
                and connectedCompoentWidth*connectedCompoentHeight*0.6 < area:
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

def getSquares_2(image_original):
    # plt.hist(image.ravel(), 256, [0, 256]);
    # plt.show()

    img = image_original < 220
    image = image_original.copy()
    image[img] = 255
    image[img == False] = 0
    # TODO 30 must be proportional to image.shape
    print('kernel could be: ', (30 / min(image.shape)))


    kernel = np.ones((20, 20), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    dilate = cv2.dilate(erosion, kernel, iterations=1)

    #cv2.imshow('dilate', dilate)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    possibleSquares = dilate > 125
    image[possibleSquares] = 254
    image[possibleSquares == False] = 0
    dst = cv2.cornerHarris(image, 2, 3, 0.04)

    possibleCorners = np.zeros(image.shape, dtype=bool)
    possibleCorners[dst > 0.05 * dst.max()] = True

    image[possibleCorners] = 254
    image[possibleCorners == False] = 0

    #cv2.imshow('image2', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    height, width = possibleCorners.shape
    K = []
    for i in range(0, height):  # looping at python speed...
        for j in range(0, width):  # ...
            if possibleCorners[i, j]:
                (I, J, counter) = completeWithOnlyMeans(possibleCorners, i, j)
                K.append((int(round(I / counter)), int(round(J / counter))))
    print(K)
    squareCenters = get4CenterOfSquare(K, image)
    squareCenters = sortSquareCenters(squareCenters)
    distI = squareCenters[3][0] - squareCenters[0][0]
    distJ = squareCenters[3][1] - squareCenters[0][1]

    print(distI, distJ)
    newImage = image_original.copy()
    if distI > distJ:
        newImage = cv2.transpose(newImage)
        newImage = cv2.flip(newImage, 0)
        for i in range(0, len(squareCenters)):
            squareCenters[i] = (newImage.shape[0] - squareCenters[i][1], squareCenters[i][0])
    squareCenters = sortSquareCenters(squareCenters)
    return squareCenters


def enderezarImagen(image):
    squaresCenters = getSquares(image)
    if len(squaresCenters) != 4:
        raise Exception('no 4 centers')
    print('first squares: ', squaresCenters)
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

    # So I will find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty

    rotatedImg = cv2.warpAffine(image, M, dsize=(int(newX), int(newY)))
    #rows, cols = image.shape
    #pointRotation = min (rows,cols)
    #M = cv2.getRotationMatrix2D((squaresCenters[3]), thetaDegrees, 1)
    #dst = cv2.warpAffine(image, M, (rows, cols))
    #plt.subplot(121), plt.imshow(image), plt.title('Input')
    #plt.subplot(122), plt.imshow(rotatedImg), plt.title('Output')
    #plt.show()

    #    squaresCenters = getSquares(dst)
    #    print('second squares: ', squaresCenters)
    #    if len(squaresCenters) != 4:
    #        raise Exception('no 4 centers')
    return rotatedImg


def isSquaredFull(franja, center, sideLength, val):
    rows, cols = franja.shape
    counter = 0
    for i in range(center[0] - sideLength // 2, center[0] + sideLength // 2):
        for j in range(center[1] - sideLength // 2, center[1] + sideLength // 2):
            if 0 <= i < rows and 0 <= j < cols:
                if (franja[i, j] > 125) == val:
                    counter += 1

    #minI = max(center[0]-sideLength//2, 0)
    #maxI = min(center[0] + sideLength // 2, rows-1)

    #minJ = max(center[1] - sideLength // 2, 0)
    #maxJ = min(center[1] + sideLength // 2, cols - 1)
    #print(val, sideLength, counter)
    #cv2.imshow('val =',franja[minI:maxI, minJ:maxJ])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return sideLength * sideLength * 0.6 < counter


def fill(franja, center, sideLength, val):
    rows, cols = franja.shape
    counter = 0
    for i in range(center[0] - sideLength // 2, center[0] + sideLength // 2):
        for j in range(center[1] - sideLength // 2, center[1] + sideLength // 2):
            if 0 <= i < rows and 0 <= j < cols:
                if val:
                    franja[i, j] = 255
                else:
                    franja[i, j] = 0


logoTipo_A = ['--X-',
              '--X-',
              '--X-',
              '----',
              '-XX-',
              '-X-X',
              'X-XX',
              '--XX']


def logoStartsHere(franja, logo, ini, L):
    rows, cols = franja.shape
    minI = max(ini[0]-L//2, 0)
    maxI = min(ini[0] + L // 2 + L * 8, rows-1)

    minJ = max(ini[1] - L // 2, 0)
    maxJ = min(ini[1] + L // 2 + 4*L, cols - 1)
    # print(val, sideLength, counter)
    #cv2.imshow('val =', franja[minI:maxI, minJ:maxJ])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    for k in range(0, 8):
        for p in range(0, 4):
            centerI = ini[0] + k * L
            centerJ = ini[1] + p * L
            if not isSquaredFull(franja, (centerI, centerJ), L, logo[k][p] == 'X'):
                return False
    return True


def fillWithCorrectValue(franja, logo, ini, L):
    for k in range(0, 8):
        for p in range(0, 4):
            centerI = ini[0] + k * L
            centerJ = ini[1] + p * L
            fill(franja, (centerI, centerJ), L, logo[k][p] == 'X')


def getLengthm(franja, row, minD, maxD):
    counter = 0
    continous = []
    for j in range(0, franja.shape[1]):
        if franja[row, j] > 125:  # blanco
            counter += 1
        else:
            if counter > 0:
                continous.append(counter)
            counter = 0
    if counter > 0:
        continous.append(counter)
    continous = sorted(continous)
    if len(continous) == 0:
        return None
    if len(continous) == 1:
        if minD <= continous[0] <= maxD:
            return continous[0]
        else:
            return None
    a = continous[-1]
    b = continous[-2]
    # a >= b
    div = (a * 1.0) / b
    dif = abs(div - round(div))
    k = int(round(div))
    if dif < 0.2 and minD <= b <= maxD and (k == 1 or k == 2):
        return b
    return None


def areClose(a, b):
    return abs(a - b) <= abs((a+b)/2.0)*2.0/13.0

def countTrueValues(franja_B, center, direction):
    counter = 0
    base = (center[0],center[1])
    while base[direction] >= 0 and franja_B[base] > 125:
        counter += 1
        if direction == 0:
            base = (base[0] - 1, base[1])
        else:
            base = (base[0], base[1] -1)
    lower = base
    base = (center[0], center[1])

    while base[direction] < franja_B.shape[direction] and franja_B[base] > 125:
        counter += 1
        if direction == 0:
            base = (base[0] + 1, base[1])
        else:
            base = (base[0], base[1] + 1)
    upper = base
    return (counter, lower, upper)

def detectFranja(franja_original):  # supuestamente vertical
    franja = franja_original.copy()
    img = franja < 200
    franja[img] = 255
    franja[img == False] = 0
    print('shape of franja original: ', franja_original.shape)
    #cv2.imshow('qwr', franja_original)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # TODO 30 must be proportional to image.shape
    # print('kernel could be: ', (20 / min(image.shape)))
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(franja, kernel, iterations=1)
    franja = cv2.dilate(erosion, kernel, iterations=1)

    rows, cols = franja.shape

    if cols >= rows:
        raise Exception('la franja debe ser vertical')

    minL = cols // 8
    maxL = cols // 4
    print(minL, maxL)

    maxJ = min(rows, 20 * maxL)
    L_m = []
    for i in range(0, maxJ):
        lengthM = getLengthm(franja, i, minL, maxL)
        if lengthM is not None:
            L_m.append(lengthM)
    L_m = sorted(L_m)
    print(L_m)
    if len(L_m) == 0:
        return None
    L = L_m[len(L_m) // 2]
    rectangleFound = False
    center = []
    for i in range(0, maxJ):
        iniJ = -1
        counter = 0
        for j in range( 0, cols):
            if franja[i,j] > 125: # blanco
                if iniJ == -1:
                    iniJ = j
                counter += 1
            else:
                if areClose(counter, L):
                    lastJ = j-1
                    centerJ = (iniJ+lastJ)//2
                    counterOverDirection0 = countTrueValues(franja, (i, centerJ), 0)

                    counterI = counterOverDirection0[0]

                    centerI = (counterOverDirection0[1][0]+counterOverDirection0[2][0])//2
                    if areClose(counterI, 3* L):
                        counterOverDirection1 = countTrueValues(franja, (centerI, centerJ), 1)
                        if areClose(counterOverDirection1[0], L):
                            print('Found ', centerI, centerJ)
                            rectangleFound = True
                            franja[centerI, centerJ] = 0
                            #cv2.imshow('found', franja)
                            #cv2.waitKey(0)
                            #cv2.destroyAllWindows()
                    center = (centerI,centerJ)

            if rectangleFound:
                break
                iniJ = -1
                counter = 0

        if rectangleFound:
            break
    if rectangleFound:
        if logoStartsHere(franja,logoTipo_A,(center[0]-L,center[1]-2*L),L):
            print('LOGO FOUND')
            return (1,1,L)


    return None


    # for L in range(minL, maxL):


#
#    for i in range(0,  L * 10):
#
#      for j in range(0, 10 * L):
#
#          if logoStartsHere(franja, logoTipo_A, (i, j), L):
#            print('Found')
#            newFranja = franja.copy()
#            newFranja[:]=0
#           fillWithCorrectValue(newFranja, logoTipo_A, (i,j), L)

def getCenterZone(img, center, delta):
    K = img[center[1] - delta:center[1] + delta, center[0] - delta:center[0] + delta].copy()
    # K = img[squaresCenters[0][1]:squaresCenters[3][1],squaresCenters[0][0]:squaresCenters[3][0]].copy()
    #ret3, th3 = cv2.threshold(K, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret3, th3 = cv2.threshold(K, 240, 255, cv2.THRESH_BINARY_INV)
    K = cv2.dilate(th3, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
    centerZone = cv2.resize(K,(750,750))
    centerZone[centerZone>125] = 255
    centerZone[centerZone <= 125] = 0

    #cv2.imwrite('centerZone_page1.png', centerZone)
    #print(delta)
    #print(center)
    #print(K.shape)
    #cv2.imshow('centro', K)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return centerZone

def getPercentMatched(baseImage,testImage):
    if baseImage.shape != testImage.shape:
        print('shapeBase ', baseImage.shape)
        print('testBase ', testImage.shape)
        return 0
    onlyMatched = cv2.bitwise_and(baseImage, testImage)
    nonZeroDB = cv2.countNonZero(baseImage)
    nonZeroAND = cv2.countNonZero(onlyMatched)

    #print('Comparing NoN Zero DB', nonZeroDB, ' afterAND', nonZeroAND)
    #plt.subplot(131), plt.imshow(testImage, 'gray'), plt.title('input')
    #plt.subplot(132), plt.imshow(baseImage,'gray'), plt.title('fromDatBase')
    #plt.subplot(133), plt.imshow(onlyMatched, 'gray'), plt.title('and')
    #plt.show()
    return nonZeroAND / nonZeroDB


def percentPage1Normal(centerZone):
    centerZone_Base = cv2.imread('extraction/centerZone_page1.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)

def percentPage2Normal(centerZone):
    centerZone_Base = cv2.imread('extraction/centerZone_page2.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)

def percentPage3Normal(centerZone):
    centerZone_Base = cv2.imread('extraction/centerZone_page3.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)

def percentPage4Normal(centerZone):
    centerZone_Base = cv2.imread('extraction/centerZone_page4.png', 0)
    return getPercentMatched(centerZone_Base, centerZone)

def detectPage(img):
    squaresCenters = getSquares(img)
    print('second squares: ', squaresCenters)
    if len(squaresCenters) != 4:
        raise Exception('no 4 centers')

    #supuestamente esta horizontal
    distCols = squaresCenters[3][1]-squaresCenters[0][1]
    distRows = squaresCenters[3][0]-squaresCenters[0][0]



    print('ratio: ', (distRows/distCols))
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

    for detector in detectors:
        percent.append(detector(centerZone))
    print('percents: ',percent)
    max_indx, max_value = max(enumerate(percent), key=lambda p: p[1])
    #print(max_value,  max_indx)
    newImage = img[squaresCenters[0][1]:squaresCenters[3][1], squaresCenters[0][0]:squaresCenters[3][0]]
    #plt.imshow(newImage,'gray')
    #plt.show()
    return (newImage,page[max_indx])

def detectPage_2(img):
    squaresCenters = getSquares(img)
    print('second squares: ', squaresCenters)
    if len(squaresCenters) != 4:
        raise Exception('no 4 centers')

    rows, cols = img.shape
    minJ_franja1 = max(0, squaresCenters[0][1] - 40)
    maxJ_franja1 = min(cols - 1, squaresCenters[0][1] + 40)

    minI_franja1 = max(0, squaresCenters[0][0] - 40)
    maxI_franja1 = min(rows - 1, squaresCenters[1][0] + 40)

    franja1 = img[minI_franja1:maxI_franja1, minJ_franja1:maxJ_franja1]
    resFranja1 = detectFranja(franja1)

    minJ_franja2 = max(0, squaresCenters[3][1] - 40)
    maxJ_franja2 = min(cols - 1, squaresCenters[3][1] + 40)

    minI_franja2 = max(0, squaresCenters[2][0] - 40)
    maxI_franja2 = min(rows - 1, squaresCenters[3][0] + 40)

    franja2 = img[minI_franja2:maxI_franja2, minJ_franja2:maxJ_franja2]
    resFranja2 = detectFranja(franja2)

    print(resFranja1, resFranja2)
    if resFranja1 is None:
        if resFranja2 is None:
            #TODO rotar apropiadamente para el caso 2
            return (2, 0)
        else:
            if resFranja2[0] == 1: #LOGO TIPO A
                return (3,resFranja2[1],resFranja2[2])
    return (0,0)

