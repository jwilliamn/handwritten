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
    # keep looping until the 'q' key is pressed
    R = Page1.getAllWithValue()
    while True:

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

        cat = R[x]
        print('Editando: ', cat[0])


        while True:
            # display the image and wait for a keypress
            image = img_clone.copy()
            position = cat[1].value.position
            ROI = image[position[0][1]:position[1][1], position[0][0]:position[1][0]]
            cv2.imshow("ROI", ROI)

            for label, category in enumerate(R):
                if category[1].value is not None:
                    category[1].value.drawPosition(image)
                    cv2.putText(image, str(label), category[1].value.position[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.imshow("image", image)

            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("a"):
                cat[1].value.position[0] = (cat[1].value.position[0][0] + 1, cat[1].value.position[0][1])
            elif key == ord("d"):
                cat[1].value.position[0] = (cat[1].value.position[0][0] - 1, cat[1].value.position[0][1])
            elif key == ord("w"):
                cat[1].value.position[0] = (cat[1].value.position[0][0], cat[1].value.position[0][1]+1)
            elif key == ord("s"):
                cat[1].value.position[0] = (cat[1].value.position[0][0], cat[1].value.position[0][1]-1)
            # if the 'c' key is pressed, break from the loop
            if key == ord("h"):
                cat[1].value.position[1] = (cat[1].value.position[1][0] - 1, cat[1].value.position[1][1])
            elif key == ord("k"):
                cat[1].value.position[1] = (cat[1].value.position[1][0] + 1, cat[1].value.position[1][1])
            elif key == ord("u"):
                cat[1].value.position[1] = (cat[1].value.position[1][0], cat[1].value.position[1][1]-1)
            elif key == ord("j"):
                cat[1].value.position[1] = (cat[1].value.position[1][0], cat[1].value.position[1][1]+1)
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

    with open('pagina1.json', 'w') as output:
        json.dump(Page1, output, default=CreatePage1Variable.jsonDefault, indent=4)

