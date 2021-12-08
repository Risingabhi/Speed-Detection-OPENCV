# Modified by Yulun Nie 11/25/2021 for version 0.1

#makes straight line
import numpy as np
import cv2

def get_points(img):
    # Set up points to return
    data = {}
    data['img'] = img.copy()
    data['lines'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # aggregate lines data
    ys = data['lines']
    assert len(ys) == 2
    _, width, _ = img.shape
    line1 = ((0, ys[0]), (width, ys[0]))
    line2 = ((0, ys[1]), (width, ys[1]))

    # line = (start_point, end_point), point = (x, y)
    return (line1, line2)

def mouse_handler(event, x, y, flags, data):
    # if mouse left is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        img = data['img'].copy()

        # modify lines list
        data['lines'].append(y)
        if len(data['lines']) > 2:
            data['lines'].pop(0)

        # display current lines
        _, width, _ = img.shape
        for i in data['lines']:
            cv2.line(img, (0, i), (width, i), (0,0,255), 2)
        cv2.imshow("Image", img)

# # Running the code
# img = cv2.imread('test.png', 1)

# # get lines y's
# line1, line2 = get_points(img)
# print(line1, line2)

# # draw lines
# final_img = cv2.line(img, line1[0], line1[1], (0,0,0), 1)
# final_img = cv2.line(final_img, line2[0], line2[1], (0,0,0), 1)

# cv2.imshow('Image', final_img)
# cv2.waitKey(0)