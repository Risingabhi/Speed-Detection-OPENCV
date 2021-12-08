import numpy as np
import cv2
import timeit
import datetime
import os
import time
import math
from line import get_points

#for SERVER SIDE
import json                    
import requests
#convert img to JSON object
import base64
import pickle


#API endpoint
api = 'https://tdispeeddetection.free.beeceptor.com/success'

speed_limit = int(input('Enter The Speed Limit: '))
distance =int(input('Enter distance between 2 lines in Meters(for better results use 10 Meters): '))



global start_time,start_time1,later,later1



def show_angle(speed_limit):
    if speed_limit !=0:
        show_direction = cv2.imread("PromptAngleinfo.JPG")
        cv2.imshow("Angle Help",show_direction)
        k = cv2.waitKey(1) & 0xff
        cv2.waitKey(100)
        Angle = int(input("Enter apporximate Angle with road :")) 

        return Angle
#Prompts user with demo image for choosing right angle.
Angle = show_angle(speed_limit) #get Angle input 



# Initialize the video & get FPS
cap = cv2.VideoCapture('night1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

lane_1_1 = []
lane_1_2 = []

#collect mask 

road_cropped = "regions.p"
with open(road_cropped,'rb')as f:
    mask_list =pickle.load(f)
    print(mask_list[0])
    print(mask_list[1])

#getting mask
mask1 = cv2.imread('m1.jpeg')
mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
ret1, thresh_MASK_1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY_INV)
mask2 = cv2.imread('m2.jpeg')
mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
ret2, thresh_MASK_2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY_INV)



# Create the background subtraction object
method = 1

if method == 0:
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
elif method == 1:
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
else:
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG()

# Create the kernel that will be used to remove the noise in the foreground mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_di = np.ones((5, 1), np.uint8)

# define variables
cnt = 0
cnt1 = 0
flag = True
flag1 = True
#distance = 0.003
#distance = 3



#Prompt user to draw 2 lines
_, img = cap.read()

line1, line2 = get_points(img)


#for line 1
l1_x1,l1_x2,l1_y1,l1_y2 = line1[0][0],line1[1][0],line1[0][1],line1[1][1]

#for line2
l2_x1,l2_x2,l2_y1,l2_y2 = line2[0][0],line2[1][0],line2[0][1],line2[1][1]

#last check point for reference of centroid tracking
'''find dist between frst 2 lines '''
starttrack = l1_y1
midtrack = l2_y1
lasttrack = int((midtrack-starttrack))
if lasttrack < 100 :
    lasttrack = (int(midtrack-starttrack)*3)+l2_y1
else:
    lasttrack = (int(midtrack-starttrack)*2)+l2_y1

print("start",starttrack)
print("last",lasttrack)
print("mid",midtrack)


## Function to Auto Calculate the detection range
'''takes input from users - speed_limit,gets FPS,distance,//pt2 and pt1 from line.py 
auto calibrates the last reference line on frame, in order to get min of 2 images for detection, code calibrates for ANY SPEED RANGE and any 
actual distance marked in Meters --- physically on ground, if distance is less say 2 Meters and you want to detect high speed of 120 KMph,code auto
calculates the new reference line, provided it doesnt fall outside the frame height'''

def max_images(speed_limit,fps,distance,midtrack,starttrack): #midtrack is last line Y pt and starttrack is first line Y pt. First line from TOP.
  
  time2coverdistance = (distance/(speed_limit*0.277))
  max_img = (time2coverdistance*fps)
  if max_img <2.0:
    #cal distance which will ensure we get atleast 2 images of vehicle d = s*t
    max_dstnc = (speed_limit*0.277)*1/fps*2
    delta = (max_dstnc-distance)
    pxl_mtr = ((midtrack-starttrack)/distance)
    pt3 = delta*pxl_mtr
    pt3_pxl = round(midtrack+pt3)
    print("max_dstnc",max_dstnc)
    print("distance",distance)
    print("delta",delta)
    print("pxl_mtr",pxl_mtr)
    print("pt3",pt3)
    print("pt3_pxl",pt3_pxl)
  else:
    pt3_pxl = midtrack+100
    print("pt3",pt3_pxl)
    return pt3_pxl

pt3_pxl = max_images(speed_limit,fps,distance,midtrack,starttrack)




area_s=[]
#defining time variables: WARNING : DONT CHANGE THESE AT ALL>>>>>>>
start_time= datetime.datetime.now()
start_time1= datetime.datetime.now()
later= datetime.datetime.now()
later1= datetime.datetime.now()

# Play until the user decides to stop
while True:
    start = timeit.default_timer()
    ret, frame = cap.read()
    # score = np.average(norm(frame, axis=2)) / np.sqrt(3)
    # if score > 60:
    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    print(framespersecond)
    #frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
    #frame =cv2.edgePreservingFilter(frame, flags=1, sigma_s=64, sigma_r=0.2) #suitable for high speed GPU -Nighttime. reduces FPS drastically
    frame_og = frame
    l, a, b = cv2.split(frame)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(1, 1)) #for improving the brightness and illumination 
    frame = clahe.apply(l)
    cv2.line(frame_og, (l1_x1, l1_y1), (l1_x2, l1_y2), (0, 255, 0), 1)
    cv2.line(frame_og, (l2_x1, l2_y1), (l2_x2, l2_y2), (0, 0, 255), 1)
    cv2.line(frame_og, (l1_x1, int((lasttrack))), (l1_x2, int((lasttrack))),(0, 0, 0), 1)
    cv2.line(frame_og,(l1_x1,pt3_pxl),(l1_x2,pt3_pxl),(200,0,127),3)

    if ret == True:
        foregroundMask = bgSubtractor.apply(frame)
        foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, kernel)
        foregroundMask = cv2.erode(foregroundMask, kernel, iterations=3)
        foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_CLOSE, kernel,iterations=6)
        foregroundMask = cv2.dilate(foregroundMask, kernel_di, iterations=7)
        foregroundMask = cv2.medianBlur(foregroundMask,5)
        thresh = cv2.threshold(foregroundMask, 25, 255, cv2.THRESH_BINARY)[1]
        thresh1 = np.bitwise_and(thresh, thresh_MASK_1)
        thresh2 = np.bitwise_and(thresh, thresh_MASK_2)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []

        for contour, hier in zip(contours, hierarchy):
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)

            cnt = contours[max_index]
            (x, y, w, h) = cv2.boundingRect(cnt)
            area_=(w*h)
            area_s.append(area_) 
            
            cx = int((w / 2) + x)
            cy = int((h / 2) + y)
            if w > 10 and h > 10:
                cv2.rectangle(frame_og, (x - 10, y - 10), (x + w, y + h), (0, 255, 0), 2)
                
                
                cv2.circle(frame_og, (cx, cy), 10, (0, 0, 255), -1)
        distA =None
        if cy > starttrack and w > 10 and h > 10:

            if flag is True and cy <midtrack:
                print("cy",cy)
                start_time = datetime.datetime.now()
                
                flag = False
            if cy > midtrack and cy < pt3_pxl:
                later = datetime.datetime.now()
                seconds = (later - start_time).total_seconds()
                frame_crossed1 = seconds*framespersecond
                speed_insta = (distance/frame_crossed1)*framespersecond*3.6
                Angle = math.radians(Angle)
                Angle = math.cos(Angle)
                speed = speed_insta*Angle
                

                print("SPEED",speed)
                print("frame_crossed1",frame_crossed1)
                print("Time taken",seconds)

                if seconds <= 0.2:
                    print("diff 0")
                else:
                    #print("seconds : " + str(seconds))
                    if flag is False:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame_og, str(int(speed)), (x, y), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
                        cv2.putText(frame, str(int(speed)), (x, y), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
                        # if not os.path.exists(path):
                        #     os.makedirs(path)
                        if int(speed) > speed_limit and cy <= lasttrack and w > 70 and h > 100:
                            roi = frame[y-50:y + h, x:x + w]
                            cv2.imshow("Lane_1", roi)
                            lane_1_1.append(roi)
                            # write_name = 'corners_found' + str(cnt1) + '.jpg'
                            # cv2.imwrite(write_name, roi)
                            # cv2.imwrite(os.path.join(path, 'carimage_l2_' + str(cnt1)) + '.jpg', roi)
                            cnt += 1
                    flag = True
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(int(speed)), (x, y), font, 2, (255, 255, 255), 8, cv2.LINE_AA)

        contours1, hierarchy1= cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            hierarchy1 = hierarchy1[0]
        except:
            hierarchy1 = []

        for contour1, hier1 in zip(contours1, hierarchy1):
            areas1 = [cv2.contourArea(c) for c in contours1]
            max_index1 = np.argmax(areas1)
            cnt1 = contours1[max_index1]
            (x1, y1, w1, h1) = cv2.boundingRect(cnt1)
            cx1 = int((w1 / 2) + x1)
            cy1 = int((h1 / 2) + y1)
            if w1 > 10 and h1 > 10:
                cv2.rectangle(frame_og, (x1 - 10, y1 - 10), (x1 + w1, y1 + h1), (255, 255, 0), 2)
                cv2.circle(frame_og, (cx1, cy1), 5, (0, 255, 0), -1)

        if cy1 > starttrack and w1 > 10 and h1 > 10:
            if flag1 is True and cy1 < midtrack:
                start_time1 = datetime.datetime.now()
                flag1 = False
            if cy1> midtrack and cy1 < pt3_pxl:
                later1 = datetime.datetime.now()
                seconds1 = (later1 - start_time1).total_seconds()
                frame_crossed2 = seconds1*framespersecond
                speed1 = (distance/frame_crossed2)*framespersecond*3.6
                Angle = math.radians(Angle)
                Angle = math.cos(Angle)
                speed1 = speed1*Angle               #COSINE CORRECTION 
                print("SPEED1",speed1)

                if seconds1 <= 0.2:
                    print("diff1 0")
                else:
                    #print("seconds1 : " + str(seconds1))
                    if flag1 is False:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame_og, str(int(speed1)), (x1, y1), font, 2, (255, 255, 255), 8, cv2.LINE_AA)
                        cv2.putText(frame, str(int(speed1)), (x1, y1), font, 2, (255, 255, 255), 8, cv2.LINE_AA)
                        # if not os.path.exists(path):
                        #     os.makedirs(path)
                        if int(speed1) > speed_limit and cy1 <= pt3_pxl and w1 > 70 and h1 > 100:
                            roi = frame[y1-50:y1 + h1, x1:x1 + w1]
                            cv2.imshow("Lane_2", roi)
                            lane_1_2.append(roi)
                            #cv2.imwrite(os.path.join('Offenders/', 'carimage_l2_' + str(cnt1)) + '.jpg', roi)
                            cnt1 += 1
                    flag1 = True
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame_og, str(int(speed1)), (x1, y1), font, 2, (255, 255, 255), 8, cv2.LINE_AA)
        #cv2.imshow('background subtraction', foregroundMask)
        #cv2.imshow('Sub',thresh)
        #cv2.imshow('Sub', thresh1)
        #cv2.imshow('Sub', frame)
        cv2.imshow('backgroundsubtraction', frame_og)
        stop = timeit.default_timer()
        time = stop-start
        print('One_frame = ',time)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
    else:
        break
# v = 0
# u = 0
# print('Saving.....')
# for la in lane_1_1:
#     cv2.imwrite('Offenders/lane1/'+'Lane'+str(v)+'.jpeg',la)
#     v+=1
# for li in lane_1_2:
#     cv2.imwrite('Offenders/lane2/'+str(v)+'.jpeg',li)
#     u+=1

# if score < 60:
#     if score <60:
#    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
#   print(framespersecond)
#   if ret == True:
#     h,w,_=(frame.shape)
#     y=0
#     x=0
    

#     roi = frame[int(y+h/2):int(y+h),x: int(x+w)]
    
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
#     gray = cv2.GaussianBlur(gray, (41, 41), 0)
   
#     (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
   

#     #to find speed 
#     cx,cy = maxLoc
#     print("cy",cy)
#     if cy > starttrack:
#       if flag is True and cy < midtrack:
#         starttime= datetime.datetime.now()
#         print("start",start)
#         flag = False
#       if cy> midtrack and cy< lasttrack:
#         endtime = datetime.datetime.now()
#         print("start",start)
#         timedelta = (endtime - starttime).total_seconds()
#         print(timedelta)
#         speed = (distance/timedelta)
#         print("speed",speed)
#     cv2.circle(roi, maxLoc, 10, (255, 0, 255), -1)
#     cv2.line(frame, (l1_x1, l1_y1), (l1_x2, l1_y2), (0, 255, 0), 1)
#     cv2.line(frame, (l2_x1, l2_y1), (l2_x2, l2_y2), (0, 0, 255), 1)
#     cv2.line(frame, (l1_x1, int((lasttrack))), (l1_x2, int((lasttrack))),(0, 0, 0), 1)
#     cv2.imshow("Robust", frame)
cap.release()
cv2.destroyAllWindows()

