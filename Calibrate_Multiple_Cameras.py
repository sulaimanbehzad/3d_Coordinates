import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
# %matplotlib inline
from mpl_toolkits.axes_grid1 import ImageGrid

# check if opencv is installed
print(cv2.__version__)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# Current Grid 11 * 7
BOARD_SIZE = (11,7)

#square size
SQUARE_SIZE = 30

# Images directory for loading
LEFT_PATH = 'data/imgs/leftcamera'
RIGHT_PATH = 'data/imgs/leftcamera'

print('We have {} Images from the left camera'.format(len(os.listdir(LEFT_PATH))))
print('and {} Images from the right camera.'.format(len(os.listdir(RIGHT_PATH))))

# sort the image names after their number
# save the image names with the whole path in a list

print('Before: {}, {}, {}, ...'.format(os.listdir(LEFT_PATH)[0], os.listdir(LEFT_PATH)[1], os.listdir(LEFT_PATH)[2]))


def SortImageNames(path):
    imagelist = sorted(os.listdir(path))
    lengths = []
    for name in imagelist:
        lengths.append(len(name))
    lengths = sorted(list(set(lengths)))
    ImageNames, ImageNamesRaw = [], []
    for l in lengths:
        for name in imagelist:
            if len(name) == l:
                ImageNames.append(os.path.join(path, name))
                ImageNamesRaw.append(name)
    return ImageNames


Left_Paths = SortImageNames(LEFT_PATH)
Right_Paths = SortImageNames(RIGHT_PATH)

print('After: {}, {}, {}, ...'.format(os.path.basename(Left_Paths[0]), os.path.basename(Left_Paths[1]),
                                      os.path.basename(Left_Paths[2])))


# we have to create the objectpoints
# that are the local 2D-points on the pattern, corresponding
# to the local coordinate system on the top left corner.

objpoints = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
objpoints[:,:2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1,2)
objpoints *= SQUARE_SIZE


# now we have to find the imagepoints
# these are the same points like the objectpoints but depending
# on the camera coordination system in 3D
# the imagepoints are not the same for each image/camera

def GenerateImagepoints(paths):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgpoints = []
    for name in paths:
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners1 = cv2.findChessboardCorners(img, BOARD_SIZE)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners1, (4,4), (-1,-1), criteria)
            imgpoints.append(corners2)
    return imgpoints

Left_imgpoints = GenerateImagepoints(Left_Paths)
Right_imgpoints = GenerateImagepoints(Right_Paths)


# we also can display the imagepoints on the example pictures.

def DisplayImagePoints(path, imgpoints):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.drawChessboardCorners(img, BOARD_SIZE, imgpoints, True)
    return img


example_image_left = DisplayImagePoints(Left_Paths[15], Left_imgpoints[15])
example_image_right = DisplayImagePoints(Right_Paths[15], Right_imgpoints[15])

fig = plt.figure(figsize=(20, 20))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)

for ax, im in zip(grid, [example_image_left, example_image_right]):
    ax.imshow(im)
    ax.axis('off')

# in this picture we now see the local coordinate system of the chessboard
# the origin is at the top left corner
# the orientation is like: long side = X

def PlotLocalCoordinates(img, points):
    points = np.int32(points)
    cv2.arrowedLine(img, tuple(points[0,0]), tuple(points[4,0]), (255,0,0), 3, tipLength=0.05)
    cv2.arrowedLine(img, tuple(points[0,0]), tuple(points[BOARD_SIZE[0]*4,0]), (255,0,0), 3, tipLength=0.05)
    cv2.circle(img, tuple(points[0,0]), 8, (0,255,0), 3)
    cv2.putText(img, '0,0', (points[0,0,0]-35, points[0,0,1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, 'X', (points[4,0,0]-25, points[4,0,1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, 'Y', (points[BOARD_SIZE[0]*4,0,0]-25, points[BOARD_SIZE[0]*4,0,1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return img

n = 15
img = cv2.imread(Left_Paths[n])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = PlotLocalCoordinates(img, Left_imgpoints[n])

fig = plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.show()



#
#
# images = glob.glob(img_path + '/*.png')
#
# for fname in images:
#     print(fname)
#     img = cv.imread(fname)
#     # cv.imshow('image', img)
#     # cv.waitKey(1500)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
#     ret, corners = cv.findChessboardCorners(gray, (columns,rows), chessboard_flags)
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         print('found')
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners2)
#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (columns,rows), corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(1500)
#
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#
#
# cv.destroyAllWindows()