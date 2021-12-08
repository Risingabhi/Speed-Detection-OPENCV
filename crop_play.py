import cv2
import numpy as np
import pickle

# output function
def draw(frame, coordinates, output):
    mask = np.zeros(frame.shape[0:2], dtype=np.uint8)
    frame =cv2.drawContours(mask, [np.array(coordinates)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res = cv2.bitwise_and(frame,frame,mask = mask)
    cv2.imwrite(output, res)

def make_mask(frame, road_cropped="regions.p"):
    #read regions.p and get data
    with open(road_cropped,'rb')as f:
        mask_list =pickle.load(f)
        print(mask_list[0])
        print(mask_list[1])

    # make output
    draw(frame, mask_list[0], "m1.jpeg")
    draw(frame, mask_list[1], "m2.jpeg")