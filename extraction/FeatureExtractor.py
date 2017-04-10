import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

#import GenerateData
#import PageDetector

from modeling import GenerateTrainDataAZ
from extraction import PageDetector
from api import engine

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def getFeature(numberedPage, indx_feature):
    if (numberedPage[1] == 3):
        if indx_feature == 5:
            return getFeaturePage3Feature5(numberedPage[0])


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x, y)


# Create a black image, a window and bind the function to window




def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (h, w))

    return rotated


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


def fillWithMeans(M, L):
    for (I, J) in L:
        M[I, J] = True
        for p in range(-3, 3):
            for q in range(-3, 3):
                ni = I + p
                nj = J + q
                if 0 <= ni < M.shape[0] and 0 <= nj < M.shape[1]:
                    M[ni, nj] = True

def filterByMaxCountDistances(L):
    D={}
    for (I, J, a, d) in L:
        if d in D:
            D[d] += 1
        else:
            D[d] = 1
    D = D.items()
    maxElementCounter = 0
    maxElement = -1
    for (element, count) in D:
        if count > maxElementCounter:
            maxElementCounter = count
            maxElement = element

    newL = []
    for (I, J, a, d) in L:
        if d == maxElement:
            newL.append((I, J, a, d))
    return newL

def getFeaturePage3Feature5(image):
    edges = cv2.Canny(image, 80, 120, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    kernel[0][0] = 0
    kernel[0][2] = 0
    kernel[2][0] = 0
    kernel[2][2] = 0
    edges = cv2.dilate(edges, kernel, iterations=1)
    #edges = cv2.erode(edges, kernel, iterations=1)
    dst = cv2.cornerHarris(image, 2, 3, 0.04)
    # for a in np.linspace(-20/1600,20/1600,100):
    #    counter = 0

    #
    #minLineLength = 100
    #maxLineGap = 1
   #
   # lines = cv2.HoughLinesP(edges, 1, np.pi / 2, 2, None, minLineLength, maxLineGap)
   # print(len(lines))
#
 #   for L in lines:
  #      for line in L:
   #         pt1 = (line[0],line[1])
    #        pt2 = (line[2],line[3])
    #        cv2.line(image, pt1, pt2, (0), 3)
    cv2.imshow('image2', edges)
    cv2.imshow('image3', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getFeaturePage3Feature5_points(image):
    print(image.shape)
    image[image < 50] = 0
    dst = cv2.cornerHarris(image, 2, 3, 0.04)
    # for a in np.linspace(-20/1600,20/1600,100):
    #    counter = 0
    possibleCorners = np.zeros(image.shape, dtype=bool)
    possibleCorners[dst > 0.01 * dst.max()] = True

    height, width = possibleCorners.shape
    K = []
    for i in range(0, height):  # looping at python speed...
        for j in range(0, width):  # ...
            if possibleCorners[i, j]:
                (I, J, counter) = completeWithOnlyMeans(possibleCorners, i, j)
                K.append((int(round(I / counter)), int(round(J / counter))))
    print(K)
    fillWithMeans(possibleCorners, K)
    image[possibleCorners] = 254
    image[possibleCorners == False] = 0

    pointsAndLines = []
    for a in np.arange(-3, 3, 1):
        for d in np.arange(150, 250, 1):

            for (I, J) in K:
                if I < 100:
                    counter = 0
                    for r in range(0, 6):
                        ni = I + d*r
                        nj = J + a*r
                        if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1] and image[ni, nj]>125:
                            counter += 1

                        else:
                            break
                    if counter == 6:
                        pointsAndLines.append((I,J, a, d))

    pointsAndLines = filterByMaxCountDistances(pointsAndLines)
    print(pointsAndLines)

    #image[possibleCorners] = 0
    #image[possibleCorners == False] = 0
    for (I, J, a, d) in pointsAndLines:
        #TODO a is not always 0
        image[ :, J ] = 255
        for r in range(0,6):
            newI = I + r*d
            image[ newI, :] = 255

    cv2.imshow('image2', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

    # cv2.namedWindow('image')
    # cv2.setMouseCallback('image', draw_circle)

    # while True:
    #    cv2.imshow('image', image)
    #    if cv2.waitKey(20) & 0xFF == 27:
    #        break
    # cv2.destroyAllWindows()

def extractPageData(img, pageNumber, baseL = None):
    if(pageNumber == 3):
        return extractPageData_number3(img, baseL)
    raise ValueError('Only implemented for page 3, not for page: '+str(pageNumber))

def addPoint(vertical, P, minX = True):

    x = vertical[0]
    if abs(P[0]-x) <= 10:

        minY = min(vertical[1], P[1])
        maxY = max(vertical[2], P[1])
        if (minX):
            return (min(x,P[0]), minY, maxY)
        else:
            return (max(x, P[0]), minY, maxY)
    else:
        if  P[0] < x:
            if minX:
                return (P[0],P[1],P[1])
            else:
                return vertical
        else:
            if minX:
                return vertical
            else:
                return (P[0],P[1],P[1])

def BFS(V, i ,j):
    Q = [(i,j)]
    V[i,j] = False
    indxIni = 0
    indxFin = 1
    rows, cols = V.shape
    leftVertical = (2000,0,0)
    rightVertical = (-2000, 0, 0)
    while indxIni < indxFin:
        (i,j) = Q[indxIni]
        leftVertical = addPoint(leftVertical, (i,j), True)
        rightVertical = addPoint(rightVertical, (i, j), False)
        indxIni += 1
        for k in range(0,4):
            newI = i + dx[k]
            newJ = j + dy[k]

            if 0 <= newI < rows and 0 <= newJ < cols and V[newI,newJ]:
                Q.append((newI,newJ))
                V[newI,newJ]  = False
                indxFin +=1
    minX = leftVertical[0]
    maxX = rightVertical[0]
    minY = max(leftVertical[1], rightVertical[1])
    maxY = min(leftVertical[2], rightVertical[2])

    #print([minXMinY, minXMaxY, maxXMinY, maxXMaxY])
    return [minX, minY, maxX, maxY]

def detectRectangles(edges):

    V = np.zeros(edges.shape, dtype=bool)
    V[edges>125] = True
    Rectangles = []
    for i in range(0,edges.shape[0]):
        for j in range(0, edges.shape[1]):
            if(V[i,j]):
                Rectangles.append(BFS(V, i, j))
    Rectangles = sorted(Rectangles, key=lambda x: -x[2] + x[0])
    R = Rectangles[0]
    print(R)
    edges[edges>=50] = 125

    edges[R[0], R[1]: R[3]] = 255
    edges[R[0]+1, R[1]: R[3]] = 255
    edges[R[0] + 2, R[1]: R[3]] = 255

    edges[R[2], R[1]: R[3]] = 255
    edges[R[2]-1, R[1]: R[3]] = 255
    edges[R[2] - 2, R[1]: R[3]] = 255

    edges[R[0]: R[2], R[1]] = 255
    edges[R[0]: R[2], R[1]+1] = 255
    edges[R[0]: R[2], R[1] + 2] = 255

    edges[R[0]: R[2], R[3]] = 255
    edges[R[0]: R[2], R[3]-1] = 255
    edges[R[0]: R[2], R[3] - 2] = 255
    print(R)
def fillAround (edges, i , j):
    for p in range(i-2, i+2):
        for q in range(j-2, j+2):
            if 0<=p<edges.shape[0] and 0<=q< edges.shape[1]:
                edges[p,q] = 255

def getPointProportion(A, B, a, b):
    px = (A[0] * b + B[0] * a) / (a + b)
    py = (A[1] * b + B[1] * a) / (a + b)
    return (int(px), int(py))

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
def getDk():
    D1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    D2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    D3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    D3[0, 0] = 0
    D3[0, 1] = 0
    D3[1, 0] = 0

    D3[2, 2] = 0
    D3[2, 1] = 0
    D3[1, 2] = 0
    D4 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    D4[0, 1] = 0
    D4[0, 2] = 0
    D4[1, 2] = 0

    D4[1, 0] = 0
    D4[2, 0] = 0
    D4[2, 1] = 0
    return [D1, D2, D3, D4]
def filterLetter(letter_original_and_mask):
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
        imgResult = GenerateTrainDataAZ.myImResize_20x20_32x32(img_copy)

        # for i in range(0,32):
        #     for j in range(0, 32):
        #         if shouldFill(imgResult, (i,j)):
        #             imgResult[i, j] = 0
        #         if shouldClear(imgResult, (i, j)):
        #             imgResult[i, j] = 255

        imgResult = (imgResult -
                      255.0 / 2) / 255.0
    except Exception as e:
        print('error filtering: ', e)
        imgResult = None
    #
    # if imgResult is not None:
    #     plt.subplot(1, 4, 1), plt.imshow(img, 'gray'), plt.title('img original')
    #     plt.subplot(1, 4, 2), plt.imshow(onlyMatch, 'gray'), plt.title('mask')
    #     plt.subplot(1, 4, 3), plt.imshow(img_copy, 'gray'), plt.title('to pass resize32x32')
    #     if( imgResult is not None):
    #         plt.subplot(1, 4, 4), plt.imshow(imgResult, 'gray'), plt.title('imgResult resized to 32x32')
    #     plt.show()

    return imgResult

def extractLetters(img, pointA,pointB, pointX,pointY, cant, soloUsuarioLapicero, edgesToDebug = None):
    letters = []
    counter = random.randint(0,1000000)
    ROI = img[pointA[0]-3:pointY[0]+3,pointA[1]-3:pointY[1]+3]
    ROI_soloLapicero = soloUsuarioLapicero[pointA[0]-3:pointY[0]+3,pointA[1]-3:pointY[1]+3]
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
    print('looking for A,B,X,Y')
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

    for k in range(0,cant):
        upperLeft = getPointProportion(pointA, pointX, k, cant - k)

        #if k ==0:
        #    upperLeft = getCross(edges, 9, upperLeft)
        #else:
        #    upperLeft = getCross(edges, 13, upperLeft)

        bottomLeft = getPointProportion(pointB, pointY, k, cant - k )

        #if k == 0:
        #    bottomLeft = getCross(edges, 3, bottomLeft)
        #else:
        #    bottomLeft = getCross(edges, 7, bottomLeft)

        upperRight = getPointProportion(pointA, pointX,  k + 1, cant - (k + 1))

        #if k == cant-1:
        #    upperRight = getCross(edges, 12, upperRight)
        #else:
        #    upperRight = getCross(edges, 13, upperRight)

        bottomRight = getPointProportion(pointB, pointY, k + 1, cant - (k + 1))

        #if k == cant-1:
        #    bottomRight = getCross(edges, 6, bottomRight)
        #else:
        #    bottomRight = getCross(edges, 7, bottomRight)

        minX = min(upperLeft[0],bottomLeft[0])+2
        maxX = max(upperRight[0], bottomRight[0])-2

        minY = min(bottomLeft[1], bottomRight[1])+2
        maxY = max(upperLeft[1], upperRight[1])-2

        singleCharacter = (ROI[minX:maxX, minY:maxY], ROI_soloLapicero[minX:maxX, minY:maxY])
        letters.append(singleCharacter)
        if edgesToDebug is not None:
            fillAround(edgesToDebug, upperLeft[0], upperLeft[1])
            fillAround(edgesToDebug, bottomLeft[0], bottomLeft[1])
            fillAround(edgesToDebug, upperRight[0], upperRight[1])
            fillAround(edgesToDebug, bottomRight[0], bottomRight[1])

        counter += 1

    filteredLetters = []

    for letter in letters:
        singleLetterFiltered = filterLetter(letter)
        filteredLetters.append(singleLetterFiltered)
    return filteredLetters

def extractPageData_number3_feature5_Row(img, pointA,pointB, pointX, pointY, edges, edgesToDebug = None):

    if edgesToDebug is not None:
        fillAround(edgesToDebug, pointA[0], pointA[1])
        fillAround(edgesToDebug, pointB[0], pointB[1])
        fillAround(edgesToDebug, pointX[0], pointX[1])
        fillAround(edgesToDebug, pointY[0], pointY[1])

    pointAX = getPointProportion(pointA, pointX, 1, 55)
    pointBY = getPointProportion(pointB, pointY, 1, 55)

    pointUpperLeftLastName = getPointProportion(pointAX,pointBY, 1, 25)
    #pointUpperLeftLastName = getCross(edges, 9, pointUpperLeftLastName)
    pointLowerLeftLastName = getPointProportion(pointAX, pointBY, 2, 6)
    #pointLowerLeftLastName = getCross(edges, 3, pointLowerLeftLastName)

    pointXA = getPointProportion(pointX, pointA, 1, 55)
    pointYB = getPointProportion(pointY, pointB, 1, 55)

    pointUpperRightLastName = getPointProportion(pointXA, pointYB, 1, 25)
    #pointUpperRightLastName = getCross(edges, 12, pointUpperRightLastName)
    pointLowerRightLastName = getPointProportion(pointXA, pointYB, 2, 6)
    #pointLowerRightLastName = getCross(edges, 6, pointLowerRightLastName)



    #28 es la cantidad de letras
    lastName = extractLetters(img, pointUpperLeftLastName,
                              pointLowerLeftLastName,
                              pointUpperRightLastName,
                              pointLowerRightLastName, 28, edges, edgesToDebug)

    pointUpperLeftLastName_M = getPointProportion(pointAX, pointBY, 43, 120)
    pointLowerLeftLastName_M = getPointProportion(pointAX, pointBY, 75, 88)
    pointUpperRightLastName_M = getPointProportion(pointXA, pointYB, 43, 120)
    pointLowerRightLastName_M = getPointProportion(pointXA, pointYB, 75, 88)

    lastName_M = extractLetters(img, pointUpperLeftLastName_M,
                              pointLowerLeftLastName_M,
                              pointUpperRightLastName_M,
                              pointLowerRightLastName_M, 28, edges, edgesToDebug)

    pointUpperLeftFirstName = getPointProportion(pointAX, pointBY, 79, 84)
    pointLowerLeftFirstName = getPointProportion(pointAX, pointBY, 111, 52)
    pointUpperRightFirstName = getPointProportion(pointXA, pointYB, 79, 84)
    pointLowerRightFirstName = getPointProportion(pointXA, pointYB, 111, 52)

    firstName = extractLetters(img, pointUpperLeftFirstName,
                                pointLowerLeftFirstName,
                                pointUpperRightFirstName,
                                pointLowerRightFirstName, 28, edges, edgesToDebug)
    pointUpperLeftFecha = getPointProportion(pointAX, pointBY, 125, 38)
    pointLowerLeftFecha = getPointProportion(pointAX, pointBY, 157, 6)
    pointUpperRightFecha_Temp = getPointProportion(pointXA, pointYB, 125, 38)
    pointLowerRightFecha_Temp = getPointProportion(pointXA, pointYB, 157, 6)
    pointUpperRightFecha = getPointProportion(pointUpperLeftFecha, pointUpperRightFecha_Temp, 8, 20)
    pointLowerRightFecha = getPointProportion(pointLowerLeftFecha, pointLowerRightFecha_Temp, 8, 20)
    #secondLastName = (img, ...)
    #firstName = extractLetters(img, ...)
    fecha = extractLetters(img, pointUpperLeftFecha,
                           pointLowerLeftFecha,
                           pointUpperRightFecha,
                           pointLowerRightFecha, 8, edges, edgesToDebug)

    pointUL_edad = getPointProportion(pointUpperLeftFecha, pointUpperRightFecha_Temp, 305-84,812-305)
    pointUR_edad = getPointProportion(pointUpperLeftFecha, pointUpperRightFecha_Temp, 359 - 84, 812 - 359)

    pointLL_edad = getPointProportion(pointLowerLeftFecha, pointLowerRightFecha_Temp, 305 - 84, 812 - 305)
    pointLR_edad = getPointProportion(pointLowerLeftFecha, pointLowerRightFecha_Temp, 359 - 84, 812 - 359)

    edad = extractLetters(img, pointUL_edad,
                          pointUR_edad,
                          pointLL_edad,
                          pointLR_edad, 2, edges, edgesToDebug)

    pointUL_meses = getPointProportion(pointUpperLeftFecha, pointUpperRightFecha_Temp, 370 - 84, 812 - 370)
    pointUR_meses = getPointProportion(pointUpperLeftFecha, pointUpperRightFecha_Temp, 422 - 84, 812 - 422)

    pointLL_meses = getPointProportion(pointLowerLeftFecha, pointLowerRightFecha_Temp, 370 - 84, 812 - 370)
    pointLR_meses = getPointProportion(pointLowerLeftFecha, pointLowerRightFecha_Temp, 422 - 84, 812 - 422)

    meses = extractLetters(img, pointUL_meses,
                           pointUR_meses,
                           pointLL_meses,
                           pointLR_meses, 2, edges, edgesToDebug)

    pointUL_dni = getPointProportion(pointUpperLeftFecha, pointUpperRightFecha_Temp, 20, 8)
    pointUR_dni = getPointProportion(pointUpperLeftFecha, pointUpperRightFecha_Temp, 20, 8)

    pointLL_dni = pointLowerRightFecha_Temp
    pointLR_dni = pointLowerRightFecha_Temp

    dni = extractLetters(img, pointUL_dni,
                           pointUR_dni,
                           pointLL_dni,
                           pointLR_dni, 8, edges, edgesToDebug)

    return [lastName, lastName_M, firstName, fecha, edad, meses,dni]

def extractPageData_number3_feature5(img, pointA,pointB, pointX,pointY, edges, edgesToDebug = None):
    data = []
    for k in range(0,5):
        upperLeft = getPointProportion(pointA,pointB, k, 5-k)
        bottomLeft = getPointProportion(pointA,pointB, k+1, 5-(k+1))

        upperRight = getPointProportion(pointX, pointY, k, 5 - k)
        bottomRight = getPointProportion(pointX, pointY, k + 1, 5 - (k + 1))

        data.append(extractPageData_number3_feature5_Row(img, upperLeft, bottomLeft, upperRight, bottomRight, edges, edgesToDebug))
        if(edgesToDebug is not None):
            fillAround(edgesToDebug, upperLeft[0], upperLeft[1])
            fillAround(edgesToDebug, bottomLeft[0], bottomLeft[1])

    return data

        #cv2.waitKey()
        #cv2.destroyAllWindows()

def calcPercent(countTrues, A, B = None):

    if B is None:
        if A[0] < 0 or A[1] < 0:
            return 0
        return countTrues[A[0], A[1]]
    else:
        sum1 = calcPercent(countTrues, B)
        sum2 = calcPercent(countTrues, (A[0]-1, A[1]-1))
        sum3 = calcPercent(countTrues, (A[0]-1, B[1]))
        sum4 = calcPercent(countTrues, (B[0], A[1]-1))
        maxPossible = (abs(B[0]-A[0])+1)*(abs(B[1]-A[1])+1)*1.0
        #print(sum1+sum2-sum4-sum3, maxPossible,A, B)
        return abs(sum1+sum2-sum4-sum3)/maxPossible

def getPercent(direction, countTrues, center, L):
    rows, cols = countTrues.shape
    A = (0,0)
    B = (0,0)
    if direction == 0:
        A = ((center[0]-L),(center[1]-L))
        B = ((center[0] + L), cols - 1)

    else:
        if direction == 1:
            A = (0, center[1] - L)
            B = (center[0] + L, center[1] + L)
        else:
            if direction == 2:
                A = (center[0]-L, 0)
                B = (center[0]+L, center[1]+L)
            else:
                if direction == 3:
                    A = (center[0] - L, center[1] - L)
                    B = (rows -1, center[1]+L)
                else:
                    raise Exception('direction '+ str(direction))
    return calcPercent(countTrues, A, B)
def getMaxL(type, countTrues, center):
    rows, cols = countTrues.shape

    numDirections = bin(type).count("1")


    currentPercent = 0

    if numDirections > 0:
        for i in range(0, 4):
            if (type & (1 << i)) > 0:
                currentPercent += getPercent(i, countTrues, center, 0) / numDirections

    left = 0, currentPercent
    right = min(min(center[0], rows - center[0]), min(center[1], cols - center[1])) - 1, 0.5

    while left[0] + 1 < right[0]:
        m = (left[0] + right[0]) //2
        currentPercent = 0

        if numDirections > 0:
            for i in range(0, 4):
                if (type & (1 << i)) > 0:
                    currentPercent += getPercent(i, countTrues, center, m)/numDirections

        if currentPercent > 0.9:
            left = m, currentPercent
        else:
            right = m, currentPercent

    return left


def getCross(img, type, centerROI):
    rows, cols = (10,10)
    centerAns = (rows//2, cols//2)
    countTrues= np.zeros((rows,cols), np.uint32)
    weightCenter = 0
    offsetI = centerROI[0] - rows//2
    offsetJ = centerROI[1] - cols // 2
    onlyTrues = np.zeros((rows, cols), np.uint32)
    for i in range(0, rows):
        for j in range(0, cols):
            onlyTrues[i,j] = 1 if img[i + offsetI, j + offsetJ] > 125 else 0
            if i == 0:
                if j == 0:
                    countTrues[i, j] = 1 if img[i + offsetI, j + offsetJ] > 125 else 0
                else:
                    countTrues[i, j] =(1 if img[i + offsetI, j + offsetJ] > 125 else 0) + countTrues[i, j - 1]
            else:
                if j == 0:
                    countTrues[i, j] = (1 if img[i + offsetI, j + offsetJ] > 125 else 0) + countTrues[i - 1, j]
                else:
                    countTrues[i, j] = (1 if img[i + offsetI, j + offsetJ] > 125 else 0) + countTrues[i - 1, j] + \
                                                                        countTrues[i, j - 1] - \
                                                                        countTrues[i-1, j-1]


    currentCenter = (1, 1)
    currentL = getMaxL(type, countTrues, currentCenter)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            tempL = getMaxL(type, countTrues, (i,j))
            #print(tempL, (i,j))
            if tempL[0] > currentL[0] or (tempL[0]==currentL[0] and tempL[1] > currentL[1]):
                currentL = tempL
                currentCenter = (i,j)
    return currentCenter[0] + offsetI, currentCenter[1] + offsetJ
patternFill = [['?1?',
               '101',
               '?1?'],
              ['?1?',
               '101',
               '011']]
patternClear = [['00?',
                '011',
                '001'],
               ['000',
                '011',
                '00?']]

def matchAnyRotation(img, p, pattern):
    for k in range(0,4):
        if match(img, p, pattern):
            return True
        pattern = rotatePattern(pattern)
    return False

def rotatePattern(pattern):
    topRow    = pattern[2][0] + pattern[1][0] + pattern[0][0]
    centerRow = pattern[2][1] + pattern[1][1] + pattern[0][1]
    bottomRow = pattern[2][2] + pattern[1][2] + pattern[0][2]
    return [topRow, centerRow, bottomRow]
def match(img, p, pattern):

    for k in range(-1, 2):
        for t in range(-1, 2):
            ni = p[0] + k
            nj = p[1] + t
            if 0 <= ni < img.shape[0] and 0 <= nj < img.shape[1]:

                if pattern[k + 1][t + 1] == '1':
                    if img[ni][nj] > 0:
                        return False

                if pattern[k + 1][t + 1] == '0':
                    if img[ni][nj] == 0:
                        return False
            else:
                return False
    return True

def shouldFill(img, p, pattern = None):
    if pattern is None:
        if shouldFill(img, p, patternFill[0]):
            return True
        if shouldFill(img, p, patternFill[1]):
            return True
        return False
    else:
        return matchAnyRotation(img, p, pattern)

def shouldClear(img, p, pattern = None):
    if pattern is None:
        if shouldClear(img, p, patternClear[0]):
            return True
        if shouldClear(img, p, patternClear[1]):
            return True
        return False
    else:
        return matchAnyRotation(img, p, pattern)

def enhance(img):
    for i in range(0,img.shape[0]):
        for j in range(0, img.shape[1]):
            if shouldFill(img, (i,j)):
                img[i,j] = 255
            if shouldClear(img, (i, j)):
                img[i, j] = 0
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
                    print('found: ', p)
                    return p
            currentK = (currentK + 1) % 4

    return copyP

def getCornersOfNamesAndLastNames(Ioriginal, I_all_base):
    SEh = cv2.getStructuringElement(cv2.MORPH_CROSS, (31, 31))
    #SEh[0:15, :] =0
    #SEh[:, 0:12] = 0
    #print(SEh)
    #SEv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    #increaseVerticalLength = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    #Ioriginal = cv2.morphologyEx(Ioriginal, cv2.MORPH_DILATE, increaseVerticalLength)
    opH_A = cv2.morphologyEx(Ioriginal, cv2.MORPH_OPEN, SEh)

    #########3
    stats = cv2.connectedComponentsWithStats(opH_A, connectivity=4)
    num_labels = stats[0]
    print('num of Crossing:', num_labels)
    labels = stats[1]
    labelStats = stats[2]
    centroides = stats[3]
    # We expect the conneccted compoennt of the numbers to be more or less with a constats ratio
    # So we find the medina ratio of all the comeonets because the majorty of connected compoent are numbers
    cosl = []
    edgesLength = []
    opH_A[opH_A>=0] = 0
    for label in range(num_labels):
        opH_A[int(round(centroides[label][1])), int(round(centroides[label][0]))] = 255
        #connectedCompoentWidth = labelStats[label, cv2.CC_STAT_WIDTH]
        #connectedCompoentHeight = labelStats[label, cv2.CC_STAT_HEIGHT]

        #area = labelStats[label, cv2.CC_STAT_AREA]
        # print(area, connectedCompoentHeight*connectedCompoentHeight, connectedCompoentHeight, connectedCompoentWidth)
        #if abs(connectedCompoentHeight - connectedCompoentWidth) < 5 \
        #        and connectedCompoentWidth * connectedCompoentHeight * 0.6 < area:
        #    cosl.append((int(round(centroides[label][0])), int(round(centroides[label][1])),  # ,
                  #       connectedCompoentWidth, connectedCompoentHeight))
            # min(connectedCompoentHeight , connectedCompoentWidth)])
        #    edgesLength.append(min(connectedCompoentHeight, connectedCompoentWidth) // 2)
    ###########
    #opV_A = cv2.morphologyEx(Ioriginal, cv2.MORPH_OPEN, SEv)
    #print(SEh)

    #crossingA = cv2.bitwise_and(opH_A,opV_A)

    TL = closestNonZero(opH_A, (129, 67))
    TR = closestNonZero(opH_A, (130, 826))
    BR = closestNonZero(opH_A, (781, 826))
    BL = closestNonZero(opH_A, (781, 67))
    print('TL ',TL)
    print('BR ', BR)
    print('TR ', TR)
    print('BL ', BL)

    #actualizar BR and BL
    BL = ((BL[0]*5-TL[0])//4,(BL[1]*5-TL[1])//4)
    BR = ((BR[0] * 5 - TR[0]) // 4, (BR[1] * 5 - TR[1]) // 4)

    #plt.subplot(2,1,1), plt.imshow(opH_A, 'gray'), plt.title('crossing Input')
    #plt.subplot(2, 1, 2), plt.imshow(Ioriginal, 'gray'), plt.title('Input image')
    #plt.show()

    return (TL, BL, TR, BR)

def extractPageData_number3(img_original, baseL):
    paginaBase = cv2.imread('extraction/pag3_1_Template.png', 0)
    img = cv2.resize(img_original, (paginaBase.shape[1],paginaBase.shape[0]))
    ret3, If = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernelDiate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    If_dilated_3 = cv2.dilate(If, kernelDiate, iterations=1)
    cornersNamesLastNames = getCornersOfNamesAndLastNames(If_dilated_3, paginaBase)


    SEh = cv2.getStructuringElement(cv2.MORPH_RECT, (30,1))
    #SEh[0,:] = 0
    #SEh[4, :] = 0
    #print(SEh)
    SEv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    #SEv[:, 0] = 0
    #SEv[:, 4] = 0
    opHorizontal = cv2.morphologyEx(If, cv2.MORPH_OPEN, SEh)
    opVertical = cv2.morphologyEx(If, cv2.MORPH_OPEN, SEv)

    Ifp = cv2.bitwise_xor(If, cv2.bitwise_or(opHorizontal,opVertical))
    NegPaginaBase = cv2.bitwise_not(paginaBase)
    # print(img.shape, ' <-> ', NegPaginaBase.shape)
    Ifp2 = cv2.medianBlur(Ifp, 3)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    Ifp2 = cv2.morphologyEx(Ifp2, cv2.MORPH_OPEN, se)
    edgesToDebug = If.copy()
    edgesToDebug[edgesToDebug>0]=125

    dataFeature5 = extractPageData_number3_feature5(img, cornersNamesLastNames[0],cornersNamesLastNames[1],
                                            cornersNamesLastNames[2],cornersNamesLastNames[3], Ifp2, edgesToDebug)
    # #enhance(Ifp)
    # #Ifp2 = cv2.bitwise_and(Ifp, NegPaginaBase)
    # plt.subplot(3,1,1), plt.imshow(img,'gray'), plt.title('Imagen Original escalada')
    # plt.subplot(3, 1, 2), plt.imshow(Ifp2, 'gray'), plt.title('Imagen binaria, tratando de que solo sean las letras')
    # plt.subplot(3, 1, 3), plt.imshow(edgesToDebug,'gray'), plt.title('Imagen Binaria, seleccionando esquinas importantes')
    # plt.show()
    #
    # squares = PageDetector.getSquares(img_original)
    #
    # img= img_original.copy()
    # #img = cv2.bilateralFilter(img, 5,75,75)
    #
    #
    # #img[img>210] = 255
    # #cv2.imshow('after median blur', img)
    # #cv2.waitKey(0)
    # #cv2.destroyAllWindows()
    # edges = cv2.Canny(img, 80, 120, apertureSize=3)
    #
    # kernel = np.ones((3, 3), np.uint8)
    # kernel[0][0] = 0
    # kernel[0][2] = 0
    # kernel[2][0] = 0
    # kernel[2][2] = 0
    # edges = cv2.dilate(edges, kernel, iterations=1)
    # #edges = cv2.erode(edges, kernel, iterations=1)
    #
    #
    # distX = squares[3][0] - squares[0][0]
    # distY = squares[3][1] - squares[0][1]
    #
    #
    # #directamente relacionado con A y B, 0, 1
    # #TODO escale 1 pixel to original
    # factorY = (squares[1][1]-squares[0][1])/distX
    # pointA = (int(squares[0][0] + baseL * 10), int(squares[0][1] + baseL * 5))
    # bestA = getCross(edges, 15, pointA)
    # pointB = (int(squares[0][0] + baseL * 85), int(squares[0][1] + baseL * (5 + factorY * 75)))
    # bestB = getCross(edges, 7, pointB)
    #
    # pointX = (int(squares[0][0] + baseL * 10), int(squares[0][1] + baseL * 74))
    # bestX = getCross(edges, 15, pointX)
    # pointY = (int(squares[0][0] + baseL * 85), int(squares[0][1] + baseL * (74 + factorY * 75)))
    # bestY = getCross(edges, 7, pointY)
    #
    # edgesToDebug = edges.copy()
    # edgesToDebug[edges>=50] = 125
    # #for sq in squares:
    # #    fillAround(edges, sq[0], sq[1])
    # #    print(sq)
    # fillAround(edgesToDebug, pointA[0], pointA[1])
    # fillAround(edgesToDebug, bestA[0], bestA[1])
    # fillAround(edgesToDebug, pointB[0], pointB[1])
    # fillAround(edgesToDebug, bestB[0], bestB[1])
    # fillAround(edgesToDebug, pointX[0], pointX[1])
    # fillAround(edgesToDebug, bestX[0], bestX[1])
    # fillAround(edgesToDebug, pointY[0], pointY[1])
    # fillAround(edgesToDebug, bestY[0], bestY[1])

    #

    #pointX = (int(squares[0][0] + baseL * 10), int(squares[0][1] + baseL * 74))
    #pointY = (int(squares[0][0] + baseL * 85.5), int(squares[0][1] + baseL * (74 + factorY * 75)))
    #dataFeature5 = extractPageData_number3_feature5(img, bestA,bestB, bestX,bestY, edges, edgesToDebug)

    for dat in dataFeature5:
        print('mostrando datos de una persona:')
        fila = 1
        for feature in dat:
            k = 1
            for d in feature:

                if d is not None:
                    pred_label = engine.predictImage(d)
                    plt.subplot(len(dat),28,k+ (fila-1)*28)
                    plt.imshow(d, cmap=plt.cm.gray)
                    plt.title(chr(pred_label+ord('A')))
                    plt.axis('off')

                k += 1
            fila += 1
        plt.show()




    #detectRectangles(edges)
    plt.subplot(121), plt.imshow(Ifp2), plt.title('Just User marks')
    plt.subplot(122), plt.imshow(edgesToDebug), plt.title('Debug')
    plt.show()
    #cv2.imshow('image2', edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()