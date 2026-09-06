#ifndef HYBRID_POSE_SOLVER_H
#define HYBRID_POSE_SOLVER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <string>

// Diamond Marker observation (4 chessboard corners)
// Surveyed target geometry; finite pixel noise, not an exact pose prior.

struct DiamondObservation {
    Eigen::Vector3d point3d;
    Eigen::Vector2d point2d;
};

// ORB Feature observation
// Normal weight, WITH Huber robust kernel - tolerates outliers
struct ORBObservation {
    Eigen::Vector3d point3d;   // 3D point from stereo triangulation
    Eigen::Vector2d point2d;   // Current frame pixel coordinate
};


// Configuration for Hybrid optimization
struct HybridConfig {
    // Higher = Diamond pose is more trusted
    double diamondWeight = 4.0;  // 1 / (0.5 px)^2; tunable, not measured covariance

    // ORB weight (baseline)
    double orbWeight = 1.0 / 2.25;

    // Huber kernel delta for ORB edges
    double huberDelta = 2.0;  // whitened 2D residual norm

    // G2O iterations
    int maxIterations = 100;

    // Verbose output
    bool verbose = false;
};

struct HybridPoseResult {
    // Final optimized pose
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();

    // OpenCV-compatible vectors
    cv::Vec3d rvec = cv::Vec3d(0, 0, 0);
    cv::Vec3d tvec = cv::Vec3d(0, 0, 0);

    // Error metrics
    double initialChi2 = 0;
    double finalChi2 = 0;
    double initialRobustCost = 0;
    double finalRobustCost = 0;
    std::string status = "not_run";
    double diamondError = 0;
    double orbMeanError = 0;

    // Statistics
    int iterations = 0;
    int numOrbInliers = 0;
    int numOrbOutliers = 0;
    bool converged = false;
};


// HybridPoseSolver: Diamond Prior + ORB Features with G2O
class HybridPoseSolver {
public:
    HybridPoseSolver();

    // Set camera intrinsics
    void setCameraParams(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);

    // Set optimization configuration
    void setConfig(const HybridConfig& config);

    // Main optimization function
    // Inputs:
    //   - initialRvec, initialTvec: Initial pose guess (from Diamond solvePnP)
    //   - diamondObs: 4 observations from Diamond marker corners
    //   - orbObs: N observations from ORB feature matching
    //
    // Pipeline inside:
    //   1. Build G2O graph with 1 camera pose vertex
    //   2. Add Diamond edges (high weight, no robust kernel)
    //   3. Add ORB edges (normal weight, Huber kernel)
    //   4. Optimize with Levenberg-Marquardt
    //   5. Return refined pose
    HybridPoseResult optimize(
        const cv::Vec3d& initialRvec,
        const cv::Vec3d& initialTvec,
        const std::vector<DiamondObservation>& diamondObs,
        const std::vector<ORBObservation>& orbObs
    );

    HybridPoseResult optimizeDiamondOnly(
        const cv::Vec3d& initialRvec,
        const cv::Vec3d& initialTvec,
        const std::vector<DiamondObservation>& diamondObs
    );

private:
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    double fx_, fy_, cx_, cy_;
    bool paramsSet_;
    HybridConfig config_;

    // Compute reprojection error for a set of 3D-2D pairs
    double computeReprojError(
        const cv::Vec3d& rvec,
        const cv::Vec3d& tvec,
        const std::vector<Eigen::Vector3d>& points3d,
        const std::vector<Eigen::Vector2d>& points2d
    );
};

#endif
