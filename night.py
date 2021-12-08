
import numpy as np
import timeit
import cv2
import datetime
import time 
import math
import pickle
from numpy.linalg import norm  #for calculating brightness

from line import get_points




global start,endtime
distance = 3.0
Angle = 30

start = datetime.datetime.now()
endtime = datetime.datetime.now()

cap =cv2.VideoCapture("night1.mp4")


flag = True
# #Prompt user to draw 2 lines
_, img = cap.read()


line1,line2 = get_points(img)


# #for line 1
l1_x1,l1_x2,l1_y1,l1_y2 = line1[0][0],line1[1][0],line1[0][1],line1[1][1]

# #for line2
l2_x1,l2_x2,l2_y1,l2_y2 = line2[0][0],line2[1][0],line2[0][1],line2[1][1]

# #last check point for reference of centroid tracking
# '''find dist between frst 2 lines '''
starttrack = l1_y1
midtrack = l2_y1
lasttrack = int((midtrack-starttrack))
if lasttrack < 100 :
    lasttrack = (int(midtrack-starttrack)*3)+l2_y1
else:
    lasttrack = (int(midtrack-starttrack)*2)+l2_y1

print(starttrack)
print(lasttrack)
          
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
    bgSubtractor = cv2.createBackgroundSubtractorMOG2(history = 150,detectShadows= True)
else:
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG()

# Create the kernel that will be used to remove the noise in the foreground mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_di = np.ones((5, 1), np.uint8)

#collect contours
contours_info=[] 
frameID = 0

if cap.isOpened() == False:
  print("Video Not available")

while(cap.isOpened()):
  start = timeit.default_timer()
  ret, frame = cap.read()
  score = np.average(norm(frame, axis=2)) / np.sqrt(3)  #cal day/night  <60 night time. lower light
  original_frame = frame.copy()
  fgmask = bgSubtractor.apply(frame)
  #==================================================================
  # filter kernel for denoising:
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
  # Fill any small holes
  closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
  # Remove noise
  opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
  # Dilate to merge adjacent blobs
  dilation = cv2.dilate(opening, kernel, iterations = 2)
  # threshold (remove grey shadows)
  dilation[dilation < 240] = 0
#=========================== contours ======================
  contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # extract every contour and its information:
  for cID, contour in enumerate(contours):
    M = cv2.moments(contour)
  # neglect small contours:
    if M['m00'] < 400:
      continue
  # centroid
    c_centroid = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
  # area
    c_area = M['m00']
    # perimeter
    try:
        c_perimeter = cv2.arcLength(contour, True)
    except:
        c_perimeter = cv2.arcLength(contour, False)
    # convexity
        c_convexity = cv2.isContourConvex(contour)
    # boundingRect
        (x, y, w, h) = cv2.boundingRect(contour)
    # br centroid
        br_centroid = (x + int(w/2), y + int(h/2)) 
    # draw rect for each contour: 
        cv2.rectangle(original_frame,(x,y),(x+w,y+h),(0,255,0),2)
    # draw id:
        cv2.putText(original_frame, str(cID), (x+w,y+h), cv2.FONT_HERSHEY_PLAIN, 3, (127, 255, 255), 1)
    # save contour info
        contours_info.append([cID,frameID,c_centroid,br_centroid,c_area,c_perimeter,c_convexity,w,h])
    if score <60:
      framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
      print(framespersecond)
    if ret == True:
      h,w,_=(frame.shape)
      y=0
      x=0
          

    #roi = frame[int(y+h/2):int(y+h),x: int(x+w)]
    
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
      gray = cv2.GaussianBlur(gray, (41, 41), 0)
    
      (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
   

      #to find speed 
      cx,cy = maxLoc
      print("cy",cy)
      if cy > starttrack:
        if flag is True and cy < midtrack:
          print("detection start",cy)
          starttime= datetime.datetime.now()
          flag = False
        if cy> midtrack and cy< lasttrack:
          print("detection ends",cy)
          endtime = datetime.datetime.now()
          timedelta = (endtime - starttime).total_seconds()
          print("timedelta",timedelta)
          frame_crossed2 = timedelta*framespersecond
          print("frm",frame_crossed2)
          speed1 = (distance/frame_crossed2)*framespersecond*3.6
          Angle = math.radians(Angle)
          Angle = math.cos(Angle)
          speed1 = speed1*Angle               #COSINE CORRECTION 
          print("SPEED1",speed1)
          
      cv2.circle(frame, maxLoc, 10, (255, 0, 255), -1)
      cv2.line(frame, (l1_x1, l1_y1), (l1_x2, l1_y2), (0, 255, 0), 1)
      cv2.line(frame, (l2_x1, l2_y1), (l2_x2, l2_y2), (0, 0, 255), 1)
      cv2.line(frame, (l1_x1, int((lasttrack))), (l1_x2, int((lasttrack))),(0, 0, 0), 1)
      cv2.imshow("Robust", original_frame)
      
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break
end = timeit.default_timer()
cap.release()

# Closes all the frames
cv2.destroyAllWindows()