import numpy as np
import cv2

from matplotlib import pyplot as plt
from extraction import FeatureExtractor
from extraction import PageDetector

img = cv2.imread('api/pagina3_1.png', 0)
#img = cv2.imread('api/pagina3_2.jpeg', 0)
#img = cv2.imread('api/pagina3_3.jpg', 0)
#img = cv2.imread('api/pagina3_4.png', 0)
#img = cv2.imread('api/pagina3_5.png', 0)
#img = cv2.imread('api/pagina3_6.png', 0)
#img = cv2.imread('api/pagina1_1.png', 0)
#img = cv2.imread('api/pagina1_2.png', 0)
#img = cv2.imread('api/pagina1_3.png', 0)
#img = cv2.imread('api/pagina2_1.png', 0)
#img = cv2.imread('api/pagina2_2.png', 0)
#img = cv2.imread('api/pagina4_1.png', 0)

print("everything is ok")

img = PageDetector.enderezarImagen(img)
page = PageDetector.detectPage(img)
if page is not None:
    plt.imshow(page[0])
    plt.title('es una pagina: '+str(page[1][0]))
    plt.show()
    if page[1][1] == 0: # esta orientado de manera normal
        FeatureExtractor.extractPageData(page[0],page[1][0])
    else:
        if page[1][1] == 1: #esta al revez
            raise NotImplementedError('aun no se hace hecho esto')
        else:
            raise ValueError('Error')
