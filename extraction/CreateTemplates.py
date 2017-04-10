import numpy as np
import cv2

import FeatureExtractor
import PageDetector

#img = cv2.imread('pagina3_2.jpeg', 0)
#img = cv2.imread('pagina3_1.png', 0)
pagina = 'pag1'
img = cv2.imread(pagina+'.jpg', 0)
#print(img)

img = PageDetector.enderezarImagen(img)

#cv2.imshow('output ', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#img = PageDetector.enderezarImagen(img)
#cv2.imshow('output 2', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
page = PageDetector.detectPage(img)
if page is not None:
    enmarcado = page[0]
    ret3, th3 = cv2.threshold(enmarcado, 240, 255, cv2.THRESH_BINARY_INV )
    for ksize in range(1,4):
        K = cv2.dilate(th3, cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize)), iterations=1)
    #centerZone = cv2.resize(K, (750, 750))
    #centerZone[centerZone > 125] = 255
    #centerZone[centerZone <= 125] = 0

        cv2.imwrite(pagina+'_'+str(ksize)+'_Template.png', K)

    #cv2.imshow('output ', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
#numberedImage = PageDetector.getPageNumber(img)
#feature1 = FeatureExtractor.getFeature(numberedImage, 5)
