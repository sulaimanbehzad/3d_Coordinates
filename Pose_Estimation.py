import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


MODE = "MPI"

if MODE == "COCO":
    protoFile = "data/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "data/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE == "MPI":
    protoFile = "data/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "data/pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Specify the input image dimensions
inWidth = 368
inHeight = 368
threshold = 0.1

image = cv2.imread("data/sample_imgs/2.jpeg")
image = cv2.resize(image, (720,720))

# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

# Set the prepared object as the input blob of the network
net.setInput(inpBlob)

output = net.forward()

H = output.shape[2]
W = output.shape[3]
# Empty list to store the detected keypoints
points = []
for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original image
    x = (image.shape[1] * point[0]) / W
    y = (image.shape[0] * point[1]) / H

    if prob > threshold:
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                    lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

cv2.imshow("Output-Keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


