import cv2
import numpy as np

# 3, 72, 88
h = 5
s = 255 * 0.77
v = 255 * 0.87
rang_hsv = 20
im_path = 'data/sample_imgs/IM_R_1.jpg'
image = cv2.imread(im_path)
image = cv2.resize(image, (720,720))
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
min_range = (0, s-rang_hsv, v-rang_hsv)
max_range = (h+15, s+rang_hsv, v+rang_hsv)
binary_img = cv2.inRange(hsv_img, min_range, max_range)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('hsv applied', binary_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 100

# Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.9

# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.2

# Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector.create()
keypoints = detector.detect(binary_img)
print(f'number of blobs: {len(keypoints)}')
im_with_keypoints = cv2.drawKeypoints(binary_img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Show keypoints
cv2.imshow('keypoints', im_with_keypoints)
cv2.waitKey(0)