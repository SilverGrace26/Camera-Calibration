cmake_minimum_required(VERSION 3.10)
project(OpenCVProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen 
find_package(Eigen3 REQUIRED)

# Find Ceres
find_package(Ceres REQUIRED)

# Include directories for OpenCV, Eigen, and Ceres
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

#For in built implementation

# add_executable(Camera InBuilt.cpp)
# target_link_libraries(Camera
#     ${OpenCV_LIBS}
#     ${CERES_LIBRARIES}
# )


# For my implementation
add_executable(CameraImp CameraImp.cpp)

# Link libraries
target_link_libraries(CameraImp
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)

# Set CUDA compiler and enable CUDA language
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
enable_language(CUDA)