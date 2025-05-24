# Camera Calibration

This is a simple self implementation of Camera Calibration with C++ using Zhang's Method. 

## Requirements

### Dependencies
- **Eigen**: Linear algebra library for matrix operations
- **C++11 or later**: Modern C++ standard support
- **OpenCV**: OpenCV with C++ for image i/o 

### System Requirements
- Linux/Unix-based system (recommended)
- GCC or Clang compiler with C++11 support

### Validation Testing
To compare results with OpenCV's implementation:
```bash
./test.sh
```

**Note**: Ensure the Eigen and OpenCV libraries are properly installed and accessible before running the validation script.

## Dataset Used

The Dataset used is [Chessboard Images for StereoCamera Calibration from Kaggle](https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures) provided by Daniel. The dataset contains 20 images of chessboard with 11 x 7 effective corners for each camera (left and right). 
The size of the squares is provided to be 30 mm. Since we are only working on Monocular Camera Calibration, we have taken only the 20 left camera images for this project.   

## Technical Reference

For a detailed explanation of the underlying mathematics and implementation details, refer to:
- **Technical Article** written by me: [Camera Calibration with Zhang's Method](https://flashblog.hashnode.dev/camera-calibration-with-zhangs-method)
- **Original Paper**: Zhang, Z. (2000). ["A flexible new technique for camera calibration"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)

