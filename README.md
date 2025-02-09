# Stereo Camera Calibration and 3D Reconstruction

## Overview
This project performs stereo camera calibration using chessboard images captured from two cameras. It extracts image points, computes intrinsic and extrinsic parameters, and performs stereo calibration to estimate the transformation between the two cameras. The calibration results are saved for further 3D reconstruction tasks.

## Features
- Detects chessboard corners for calibration
- Computes intrinsic and extrinsic camera parameters
- Performs stereo camera calibration
- Saves calibration parameters for future use

## Dependencies
Ensure the following libraries are installed before running the scripts:
```sh
pip install numpy opencv-python matplotlib
```

## Dataset
- Chessboard images should be placed in the following directories:
  - `data/imgs/leftcamera/` (Left camera images)
  - `data/imgs/rightcamera/` (Right camera images)
- You can generate a chessboard pattern using: [Chessboard Pattern Generator](https://calib.io/pages/camera-calibration-pattern-generator)

## Usage
1. **Check OpenCV Installation**
   ```python
   import cv2
   print(cv2.__version__)
   ```
2. **Prepare Image Paths**
   Images should be sorted to ensure correct pairing.
3. **Detect Chessboard Corners**
   The script extracts chessboard corners for both cameras.
4. **Calibrate Individual Cameras**
   Each camera's intrinsic and extrinsic parameters are estimated.
5. **Stereo Calibration**
   Computes the transformation matrix between the two cameras.
6. **Save Parameters**
   Calibration parameters are stored in `data/out/parameters.npz`.

## Outputs
- Intrinsic Camera Matrix
- Distortion Coefficients
- Rotation and Translation Matrices
- Reprojection Errors
- Saved calibration parameters

## Example Execution
Run the script to perform calibration:
```sh
python stereo_calibration.py
```

## Results
- The transformation matrix and fundamental matrix are printed.
- Mean reprojection error provides calibration accuracy.
- The parameters are saved for future 3D reconstruction.

## Notes
- Ensure chessboard images are well-lit and have minimal distortion.
- Use at least 20 well-captured images for better calibration accuracy.

## License
This project is open-source and available under the MIT License.
