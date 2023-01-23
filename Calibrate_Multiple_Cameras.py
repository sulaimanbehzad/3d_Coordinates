import numpy as np
import cv2 as cv
import glob

# check if opencv is installed
print(cv.__version__)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# Current Grid 11 * 7
rows = 7
columns = 11
objp = np.zeros((rows*columns,3), np.float32)
objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# Images directory for loading
img_path = 'data/imgs/leftcamera'
images = glob.glob(img_path + '/*.png')
print(len(images), "images found")
for fname in images:
    print(fname)
    img = cv.imread(fname)
    # cv.imshow('image', img)
    # cv.waitKey(1500)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv.findChessboardCorners(gray, (columns,rows), chessboard_flags)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print('found')
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (columns,rows), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1500)
cv.destroyAllWindows()