from pip._vendor.distlib.compat import raw_input

from extraction.FormatModel import CreatePage1Variable
from extraction.FormatModel.RawVariableDefinitions import *
from extraction.FormatModel.UtilFunctionsLoadTemplates import loadCategory
import pickle
import cv2
import json
image = cv2.imread('../../resources/pag1_1_Template.png')

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

if __name__ == '__main__':

    if(image is None):
        print('image not foound')
    img_clone = image.copy()
    with open('pagina1.json', 'r') as input:
        print(input)
        dict_Page1 = json.load(input)
        Page1 = loadCategory(dict_Page1)
        print(Page1)

    Page1.describe(True)


    # for variable in R:
    #     if variable[1] is None:
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed

    while True:

        R = Page1.getAllWithValue()
        image = img_clone.copy()
        for label, category in enumerate(R):
            if category[1].value is not None:
                category[1].value.drawPosition(image)
                cv2.putText(image, str(label), category[1].value.position[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 255), thickness=2)
            print(label, '][', category[0])
        cv2.imshow("image", image)

        cv2.waitKey(400)
        x = int(raw_input('>=0 para continuar, -1 para terminar'))
        if x < 0:
            break

        category = R[x]
        print('Editando: ', category[0])
        x = int(raw_input(' 1 para numeros 0 para letras, 2 para categorico general'))

        if x == 1:
            y = int(raw_input(' cantidad de cifras'))
        else:
            if x == 2:
                y = int(raw_input(' deberias escribir 1'))
            else:
                y = int(raw_input(' cantidad de letras'))

        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                if len(refPt) == 2:
                    print('hast 2 elements')
                    if x == 1:
                        val = ArrayImageNumber(refPt, y)
                    else:
                        if x == 2:
                            val = ImageCategoric(refPt, y)
                        else:
                            val = ArrayImageChar(refPt, y)

                    category[1].value = val
                    break
                    #image = clone.copy()

            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

    with open('pagina1.json', 'w') as output:
        json.dump(Page1, output, default=CreatePage1Variable.jsonDefault, indent=4)

