#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Chessboard Dimensions
const int BOARD_HEIGHT = 11;
const int BOARD_WIDTH = 7;
const float SQUARE_SIZE = 0.03f;

int main() {
    vector<Point3f> objp;

    // 3D points in the real world (Assuming z = 0 for a flat chessboard)
    for (int i = 0; i < BOARD_HEIGHT; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            objp.push_back(Point3f(j * SQUARE_SIZE, i * SQUARE_SIZE, 0));
        }
    }

    vector<vector<Point3f>> objpoints; // 3D points in real world space
    vector<vector<Point2f>> imgpoints; // 2D points in image plane

    vector<String> images;
    glob("../data/imgs/*.png", images);

    if (images.empty()) {
        cerr << "No images found in the specified directory!" << endl;
        return -1;
    }

    Size imageSize;
    for (size_t i = 0; i < images.size(); i++) {
        Mat img = imread(images[i], IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Failed to load image: " << images[i] << endl;
            continue;
        }

        // Set image size once (assuming all images are the same size)
        if (i == 0) {
            imageSize = img.size();
        }

        vector<Point2f> corners;
        bool found = findChessboardCorners(img, Size(BOARD_WIDTH, BOARD_HEIGHT), corners);

        if (found) {
            // Refine corner locations
            cornerSubPix(img, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));
            objpoints.push_back(objp);
            imgpoints.push_back(corners);
        } else {
            cerr << "Chessboard corners not found in image: " << images[i] << endl;
        }
    }

    if (objpoints.empty()) {
        cerr << "No valid images with chessboard corners found!" << endl;
        return -1;
    }

    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;

    double rms = calibrateCamera(objpoints, imgpoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

    if (rms < 0) {
        cerr << "Camera calibration failed!" << endl;
        return -1;
    }

    cout << "Camera Matrix:\n" << cameraMatrix << endl;
    cout << "Distortion Coefficients:\n" << distCoeffs << endl;
    cout << "RMS Error: " << rms << endl;

    destroyAllWindows(); // Close all OpenCV windows
    return 0;
}
