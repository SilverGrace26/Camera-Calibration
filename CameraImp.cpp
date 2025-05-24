#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

// --- Helper Functions ---

Eigen::Matrix3d rodriguesToRotation(const Eigen::Vector3d& rvec) {
    double angle = rvec.norm();
    if (angle < std::numeric_limits<double>::epsilon()) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d axis = rvec.normalized();
    Eigen::AngleAxisd aa(angle, axis);
    return aa.toRotationMatrix();
}

Eigen::Vector3d rotationToRodrigues(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
}

Eigen::Vector2d projectPoints(const Eigen::Vector3d& world_point,
                              const Eigen::Matrix3d& K,
                              const Eigen::Matrix3d& R,
                              const Eigen::Vector3d& t,
                              const Eigen::VectorXd& dist_coeffs) {
    // Transform to camera coordinates
    Eigen::Vector3d cam_coords = R * world_point + t;

    // Check for points behind camera
    if (cam_coords.z() <= std::numeric_limits<double>::epsilon()) {
        return Eigen::Vector2d(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    }

    // Normalize to image plane
    double x = cam_coords.x() / cam_coords.z();
    double y = cam_coords.y() / cam_coords.z();
    double r2 = x * x + y * y;

    // Apply distortion
    double k1 = dist_coeffs(0);
    double k2 = dist_coeffs(1);
    double k3 = dist_coeffs(2);
    double p1 = dist_coeffs(3);
    double p2 = dist_coeffs(4);

    // Radial distortion
    double radial_factor = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    double x_radial = x * radial_factor;
    double y_radial = y * radial_factor;

    // Tangential distortion
    double x_dist = x_radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    double y_dist = y_radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    // Convert to pixel coordinates
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);
    double skew = K(0, 1);

    return Eigen::Vector2d(fx * x_dist + skew * y_dist + cx, fy * y_dist + cy);
}

// Functor for Levenberg-Marquardt optimization
struct CameraCalibrationFunctor {
    typedef double Scalar;
    typedef Eigen::VectorXd InputType;
    typedef Eigen::VectorXd ValueType;
    typedef Eigen::MatrixXd JacobianType;
    enum {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
    };

    const std::vector<std::vector<Eigen::Vector3d>>& world_points_all;
    const std::vector<std::vector<Eigen::Vector2d>>& image_points_all;
    const int num_images;
    const int num_intrinsics_params = 5;
    const int num_distortion_params = 5;
    const int num_extrinsics_per_image = 6;

    CameraCalibrationFunctor(const std::vector<std::vector<Eigen::Vector3d>>& wp,
                             const std::vector<std::vector<Eigen::Vector2d>>& ip)
        : world_points_all(wp), image_points_all(ip), num_images(wp.size()) {}

    int inputs() const {
        return num_intrinsics_params + num_distortion_params + num_images * num_extrinsics_per_image;
    }

    int values() const {
        int total = 0;
        for (const auto& pts : image_points_all) {
            total += 2 * pts.size();
        }
        return total;
    }

    int operator()(const InputType& parameters, ValueType& residuals) const {
        residuals.resize(values());

        // Unpack intrinsic parameters
        Eigen::Matrix3d K;
        K.setZero();
        K(0, 0) = parameters(0); // fx
        K(1, 1) = parameters(1); // fy
        K(0, 2) = parameters(2); // cx
        K(1, 2) = parameters(3); // cy
        K(0, 1) = parameters(4); // skew
        K(2, 2) = 1.0;

        // Unpack distortion parameters
        Eigen::VectorXd distortion_params = parameters.segment(num_intrinsics_params, num_distortion_params);

        int residual_idx = 0;
        for (int i = 0; i < num_images; ++i) {
            int current_extrinsics_offset = num_intrinsics_params + num_distortion_params + i * num_extrinsics_per_image;
            Eigen::Vector3d rvec = parameters.segment<3>(current_extrinsics_offset);
            Eigen::Vector3d tvec = parameters.segment<3>(current_extrinsics_offset + 3);

            Eigen::Matrix3d R_mat = rodriguesToRotation(rvec);

            for (size_t j = 0; j < world_points_all[i].size(); ++j) {
                Eigen::Vector2d predicted_pixel = projectPoints(world_points_all[i][j],
                                                                K, R_mat, tvec, distortion_params);

                Eigen::Vector2d observed_pixel = image_points_all[i][j];
                Eigen::Vector2d residual_vec = observed_pixel - predicted_pixel;

                residuals(residual_idx++) = residual_vec.x();
                residuals(residual_idx++) = residual_vec.y();
            }
        }
        return 0;
    }

    int df(const InputType& x, JacobianType& jac) const {
        Eigen::NumericalDiff<CameraCalibrationFunctor> num_diff(*this);
        jac.resize(values(), inputs());
        num_diff.df(x, jac);
        return 0;
    }
};

int main() {
    // Values taken from dataset
    const cv::Size board_size(11, 7); 
    const float square_size_mm = 30.0f; 
    
    std::vector<std::string> image_filenames;
    cv::glob("data/imgs/*.png", image_filenames); 
    
    if (image_filenames.empty()) {
        std::cerr << "No images found! Please update the path to your chessboard images." << std::endl;
        return -1;
    }

    std::vector<std::vector<cv::Point2f>> image_points_all_cv;
    std::vector<std::vector<cv::Point3f>> world_points_all_cv;

    // Generating 3D world points template (Z=0 plane)
    std::vector<cv::Point3f> object_points_template;
    for (int i = 0; i < board_size.height; ++i) {
        for (int j = 0; j < board_size.width; ++j) {
            object_points_template.push_back(cv::Point3f(j * square_size_mm, i * square_size_mm, 0.0f));
        }
    }

    // Stage 1 : Corner Detection
    for (const auto& filename : image_filenames) {
        cv::Mat image = cv::imread(filename);
        if (image.empty()) {
            std::cerr << "Could not load image: " << filename << std::endl;
            continue;
        }

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, board_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

        if (found) {
            // To Refine corner positions
            cv::cornerSubPix(gray, corners, cv::Size(11, 7), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
            
            image_points_all_cv.push_back(corners);
            world_points_all_cv.push_back(object_points_template);
        } else {
            std::cout << "Could not find corners in: " << filename << std::endl;
        }
    }

    // Convert to Eigen format
    std::vector<std::vector<Eigen::Vector3d>> world_points_all_eigen(world_points_all_cv.size());
    std::vector<std::vector<Eigen::Vector2d>> image_points_all_eigen(image_points_all_cv.size());

    for (size_t i = 0; i < world_points_all_cv.size(); ++i) {
        world_points_all_eigen[i].reserve(world_points_all_cv[i].size());
        for (const auto& p : world_points_all_cv[i]) {
            world_points_all_eigen[i].push_back(Eigen::Vector3d(p.x, p.y, p.z));
        }
    }
    for (size_t i = 0; i < image_points_all_cv.size(); ++i) {
        image_points_all_eigen[i].reserve(image_points_all_cv[i].size());
        for (const auto& p : image_points_all_cv[i]) {
            image_points_all_eigen[i].push_back(Eigen::Vector2d(p.x, p.y));
        }
    }

    // Stage 2 : Homography Estimation 
    std::vector<Eigen::Matrix3d> homographies;
    
    for (size_t i = 0; i < image_points_all_cv.size(); ++i) {
        // Project 3D points to Z=0 plane
        std::vector<cv::Point2f> world_points_2d;
        for (const auto& p3d : world_points_all_cv[i]) world_points_2d.push_back(cv::Point2f(p3d.x, p3d.y)); 
        
        
        cv::Mat H_cv = cv::findHomography(world_points_2d, image_points_all_cv[i], cv::RANSAC);
        if (H_cv.empty()) {
            std::cerr << "Failed to compute homography for image " << i << std::endl;
            continue;
        }
        
        // Convert to Eigen
        Eigen::Matrix3d H_eigen;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) H_eigen(r, c) = H_cv.at<double>(r, c); 
        }

        homographies.push_back(H_eigen);
    }


    // Stage 3 : Zhang's Method for Initial Intrinsics 
    Eigen::MatrixXd V(2 * homographies.size(), 6);
    for (size_t i = 0; i < homographies.size(); ++i) {
        
        Eigen::Vector3d h1 = homographies[i].col(0);
        Eigen::Vector3d h2 = homographies[i].col(1);

        // Constructing the matrix V

        // First constraint: h1^T * B * h2 = 0
        V.row(2 * i) << h1(0) * h2(0),
                        h1(0) * h2(1) + h1(1) * h2(0),
                        h1(1) * h2(1),
                        h1(2) * h2(0) + h1(0) * h2(2),
                        h1(2) * h2(1) + h1(1) * h2(2),
                        h1(2) * h2(2);

        // Second constraint: h1^T * B * h1 - h2^T * B * h2 = 0
        V.row(2 * i + 1) << h1(0) * h1(0) - h2(0) * h2(0),
                            2 * (h1(0) * h1(1) - h2(0) * h2(1)),
                            h1(1) * h1(1) - h2(1) * h2(1),
                            2 * (h1(2) * h1(0) - h2(2) * h2(0)),
                            2 * (h1(2) * h1(1) - h2(2) * h2(1)),
                            h1(2) * h1(2) - h2(2) * h2(2);
    }

    // Solve using SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeFullV);
    Eigen::VectorXd b = svd.matrixV().col(5);

    // Reconstruct B matrix
    Eigen::Matrix3d B;
    B << b(0), b(1), b(3),
         b(1), b(2), b(4),
         b(3), b(4), b(5);

    // Extract intrinsic parameters with better numerical stability
    double v0 = (B(0,1) * B(0,2) - B(0,0) * B(1,2)) / (B(0,0) * B(1,1) - B(0,1) * B(0,1));
    double lambda = B(2,2) - (B(0,2) * B(0,2) + v0 * (B(0,1) * B(0,2) - B(0,0) * B(1,2))) / B(0,0);
    
    // Add bounds checking for numerical stability
    if (lambda <= 0 || B(0,0) <= 0) {
        std::cerr << "Numerical instability in intrinsic estimation. Using fallback values." << std::endl;
        // fallback values 
        lambda = 500000; 
        B(0,0) = 1e-6;   
    }
    
    double fx = std::sqrt(lambda / B(0,0));
    double fy = std::sqrt(lambda * B(0,0) / (B(0,0) * B(1,1) - B(0,1) * B(0,1)));
    double skew = -B(0,1) * fx * fx * fy / lambda;
    double cx = skew * v0 / fy - B(0,2) * fx * fx / lambda;
    double cy = v0;

    // Sanity check and bound the values
    fx = std::max(100.0, std::min(fx, 5000.0));
    fy = std::max(100.0, std::min(fy, 5000.0));
    skew = std::max(-50.0, std::min(skew, 50.0));

    Eigen::Matrix3d K_initial;
    K_initial << fx, skew, cx,
                 0,  fy,   cy,
                 0,  0,    1;

    // std::cout << "Initial Intrinsic Matrix K:\n" << K_initial << std::endl;

    // Stage 4 : Extract Extrinsic Parameters
    std::vector<Eigen::Matrix3d> R_initial_all;
    std::vector<Eigen::Vector3d> t_initial_all;
    Eigen::Matrix3d K_inv = K_initial.inverse();

    for (const auto& H : homographies) {
        Eigen::Vector3d h1 = H.col(0);
        Eigen::Vector3d h2 = H.col(1);
        Eigen::Vector3d h3 = H.col(2);

        double lambda1 = 1.0 / (K_inv * h1).norm();
        double lambda2 = 1.0 / (K_inv * h2).norm();
        double lambda_scale = (lambda1 + lambda2) / 2.0;

        Eigen::Vector3d r1 = lambda_scale * K_inv * h1;
        Eigen::Vector3d r2 = lambda_scale * K_inv * h2;
        Eigen::Vector3d r3 = r1.cross(r2);

        Eigen::Matrix3d R_approx;
        R_approx.col(0) = r1;
        R_approx.col(1) = r2;
        R_approx.col(2) = r3;

        // Ensure proper rotation matrix
        Eigen::JacobiSVD<Eigen::Matrix3d> svd_R(R_approx, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R_corrected = svd_R.matrixU() * svd_R.matrixV().transpose();

        if (R_corrected.determinant() < 0) R_corrected.col(2) *= -1;
        

        Eigen::Vector3d t_extracted = lambda_scale * K_inv * h3;

        R_initial_all.push_back(R_corrected);
        t_initial_all.push_back(t_extracted);
    }

    // Stage 5 : Setup optimization
    int num_intrinsics_params = 5;
    int num_distortion_params = 5;
    int num_extrinsics_per_image = 6;
    int total_num_parameters = num_intrinsics_params + num_distortion_params + image_points_all_eigen.size() * num_extrinsics_per_image;
    
    Eigen::VectorXd parameters(total_num_parameters);

    // Initialize parameters
    parameters(0) = K_initial(0, 0); // fx
    parameters(1) = K_initial(1, 1); // fy
    parameters(2) = K_initial(0, 2); // cx
    parameters(3) = K_initial(1, 2); // cy
    parameters(4) = K_initial(0, 1); // skew

    // Initialize distortion coefficients to zero
    parameters.segment<5>(num_distortion_params).setZero();

    // Initialize extrinsic parameters
    for (size_t i = 0; i < image_points_all_eigen.size(); ++i) {
        int offset = num_intrinsics_params + num_distortion_params + i * num_extrinsics_per_image;
        Eigen::Vector3d rvec = rotationToRodrigues(R_initial_all[i]);
        parameters.segment<3>(offset) = rvec;
        parameters.segment<3>(offset + 3) = t_initial_all[i];
    }

    // Create functor and solver
    CameraCalibrationFunctor functor(world_points_all_eigen, image_points_all_eigen);
    Eigen::LevenbergMarquardt<CameraCalibrationFunctor> lm_solver(functor);

    // Set solver parameters
    lm_solver.parameters.maxfev = 1000;
    lm_solver.parameters.ftol = 1e-8;
    lm_solver.parameters.xtol = 1e-8;
    lm_solver.parameters.gtol = 1e-8;

    // Perform optimization
    Eigen::LevenbergMarquardtSpace::Status status = lm_solver.minimize(parameters);
    
    std::cout << "Optimization Status: " << status << std::endl;

    // Extract final results
    Eigen::Matrix3d K_final;
    K_final.setZero();
    K_final(0, 0) = parameters(0);
    K_final(1, 1) = parameters(1);
    K_final(0, 2) = parameters(2);
    K_final(1, 2) = parameters(3);
    K_final(0, 1) = parameters(4);
    K_final(2, 2) = 1.0;

    Eigen::VectorXd dist_final = parameters.segment(num_intrinsics_params, num_distortion_params);

    std::cout << "Final Intrinsic Matrix K:\n" << K_final << std::endl;
    std::cout << "Final Distortion Coefficients (k1, k2, k3, p1, p2):\n" << dist_final.transpose() << std::endl;
    std::cout << "Final RMS Error: " << std::sqrt(lm_solver.fvec.squaredNorm() / lm_solver.fvec.size()) << std::endl;

    return 0;
}