import cv2
import numpy as np

img1 = cv2.imread('picuture.jpg', 0)
img1 = cv2.resize(img1, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)

cap = cv2.VideoCapture('video2.mp4',0)


surf = cv2.xfeatures2d.SURF_create(5000)
kp1, des1 = surf.detectAndCompute(img1, None)


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        kp2, des2 = surf.detectAndCompute(frame, None)

        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des1, des2)

        res = cv2.drawMatches(img1, kp1, frame, kp2, matches, None, \
                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Frame',res)
    

    # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    else: 
        break

cap.release()
cv2.destroyAllWindows()