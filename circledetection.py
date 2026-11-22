import cv2
import numpy as np

eyes=cv2.imread("Lesson 5/images/Image20261122111339.jpg")

eyesgray=cv2.imread("Lesson 5/images/Image20261122111339.jpg",0)

grayblurred=cv2.blur(eyesgray,(3,3))

detectedcircles=cv2.HoughCircles(grayblurred,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=1,maxRadius=40)

if detectedcircles is not None:
    detectedcircles=np.uint16(np.around(detectedcircles))
    for i in detectedcircles[0,:]:
        x,y,r=i[0],i[1],i[2]
        cv2.circle(eyes,(x,y),r,(0,10,0),2)
        cv2.imshow("eyes",eyes)
        cv2.waitKey(0)