#ifndef STEREO_TRIANGULATION_H
#define STEREO_TRIANGULATION_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <vector>

// Stereo configuration
struct StereoConfig {
    double baseline = 0.10;
    cv::Vec3d baselineDirection = cv::Vec3d(1.0, 0.0, 0.0);

    double minDisparity = 2.0;
    double maxReprojError = 2.0;
    double maxEpipolarError = 1.5;
    double minParallaxDegrees = 1.0;
    double minDepth = 0.05;
    double maxDepth = 10.0;
};

// Result of stereo triangulation
struct TriangulatedPoint {
    Eigen::Vector3d point3d = Eigen::Vector3d::Zero();
    cv::Point2f pointLeft;
    cv::Point2f pointRight;
    double disparity = 0;
    double depth = 0;
    double reprojError = 0;
    bool valid = false;
};

struct StereoTriangulationResult {
    std::vector<TriangulatedPoint> points;
    int numValid;
    int numRejected;
    double meanDepth;
    double meanReprojError;
};

// StereoTriangulator: Compute 3D from matched 2D points
class StereoTriangulator {
public:
    StereoTriangulator();

    // Set camera intrinsics (same camera, two positions)
    void setCameraParams(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);

    // Set stereo configuration
    void setConfig(const StereoConfig& config);

    StereoTriangulationResult triangulate(
        const std::vector<cv::Point2f>& pointsLeft,
        const std::vector<cv::Point2f>& pointsRight
    );

    TriangulatedPoint triangulatePoint(
        const cv::Point2f& left,
        const cv::Point2f& right
    );

private:
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    double fx_, fy_, cx_, cy_;
    bool paramsSet_;
    StereoConfig config_;

    cv::Mat P1_;
    cv::Mat P2_;

    void computeProjectionMatrices();
};

#endif
