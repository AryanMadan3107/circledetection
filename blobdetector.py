import cv2
import numpy as np

shapes=cv2.imread("Lesson 5/images/Image20261122113117.jpg",0)

params=cv2.SimpleBlobDetector_Params()

params.filterByArea=True
params.minArea=100

params.filterByCircularity=True
params.minCircularity=0.9

params.filterByConvexity=True
params.minConvexity=0.2

params.filterByInertia=True
params.minInertiaRatio=0.001

detector=cv2.SimpleBlobDetector_create(params)

keypoints=detector.detect(shapes)

print(keypoints)

blank=np.zeros((1,1))
blob=cv2.drawKeypoints(shapes,keypoints,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
blob=cv2.putText(blob,"No. images = "+ str(len(keypoints)),(20,20),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),1)

cv2.imshow("circles",blob)

cv2.waitKey(0)