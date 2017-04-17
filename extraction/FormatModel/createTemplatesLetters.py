import cv2

image = []
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
    paginaBase_1 = cv2.imread('../../resources/pag1_1_Template.png', 0)
    paginaBase_2 = cv2.imread('../../resources/pag2_1_Template.png', 0)
    paginaBase_3 = cv2.imread('../../resources/pag3_1_Template.png', 0)
    paginaBase_4 = cv2.imread('../../resources/pag4_1_Template.png', 0)
    paginas = [paginaBase_1, paginaBase_2, paginaBase_3, paginaBase_4]

    positions = [[(66, 103), (95, 138)],[(61, 305), (93, 345)],[(82, 135), (108, 167)],[(76, 125), (102, 158)]]
    #cv2.namedWindow("image")
    #cv2.setMouseCallback("image", click_and_crop)

    for label,pagina in enumerate(paginas):
        img_clone = pagina
        position = positions[label]
        # while True:
        #     # display the image and wait for a keypress
        #     cv2.imshow("image", image)
        #     key = cv2.waitKey(1) & 0xFF
        #
        #     # if the 'r' key is pressed, reset the cropping region
        #     if key == ord("r"):
        #         if len(refPt) == 2:
        #             print(refPt)
        #             break
        #             # image = clone.copy()
        #
        #     # if the 'c' key is pressed, break from the loop
        #     elif key == ord("c"):
        #         break

        #
        # while True:
        #     image = img_clone.copy()
        #
        #     ROI = image[position[0][1]:position[1][1], position[0][0]:position[1][0]]
        #     cv2.imshow("ROI", ROI)
        #     #cv2.imshow("image", image)
        #
        #     key = cv2.waitKey(1) & 0xFF
        #
        #     # if the 'r' key is pressed, reset the cropping region
        #     if key == ord("a"):
        #         position[0] = (position[0][0] + 1, position[0][1])
        #     elif key == ord("d"):
        #         position[0] = (position[0][0] - 1, position[0][1])
        #     elif key == ord("w"):
        #         position[0] = (position[0][0], position[0][1] + 1)
        #     elif key == ord("s"):
        #         position[0] = (position[0][0], position[0][1] - 1)
        #     # if the 'c' key is pressed, break from the loop
        #     if key == ord("h"):
        #         position[1] = (position[1][0] - 1, position[1][1])
        #     elif key == ord("k"):
        #         position[1] = (position[1][0] + 1, position[1][1])
        #     elif key == ord("u"):
        #         position[1] = (position[1][0], position[1][1] - 1)
        #     elif key == ord("j"):
        #         position[1] = (position[1][0], position[1][1] + 1)
        #     # if the 'c' key is pressed, break from the loop
        #     elif key == ord("c"):
        #         print(position)
        #         break

        image = img_clone.copy()
        ROI = image[position[0][1]:position[1][1], position[0][0]:position[1][0]]
        cv2.imwrite('cuadro_template_'+str(label+1)+'.png',ROI)