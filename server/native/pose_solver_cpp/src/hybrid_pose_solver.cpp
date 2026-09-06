#include "hybrid_pose_solver.h"

// G2O core
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sba/edge_project_xyz.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <limits>
#include <stdexcept>
#include <cmath>

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace g2o;

HybridPoseSolver::HybridPoseSolver()
    : fx_(0), fy_(0), cx_(0), cy_(0), paramsSet_(false) {}

void HybridPoseSolver::setCameraParams(const Mat& cameraMatrix, const Mat& distCoeffs) {
    if (cameraMatrix.type()!=CV_64F || cameraMatrix.rows!=3 || cameraMatrix.cols!=3 ||
        !cv::checkRange(cameraMatrix) || !cv::checkRange(distCoeffs))
        throw std::invalid_argument("Invalid camera parameters");
    cameraMatrix_ = cameraMatrix.clone();
    distCoeffs_ = distCoeffs.clone();

    fx_ = cameraMatrix.at<double>(0, 0);
    fy_ = cameraMatrix.at<double>(1, 1);
    cx_ = cameraMatrix.at<double>(0, 2);
    cy_ = cameraMatrix.at<double>(1, 2);

    paramsSet_ = true;
}

void HybridPoseSolver::setConfig(const HybridConfig& config) {
    if (!std::isfinite(config.diamondWeight) || config.diamondWeight<=0 ||
        !std::isfinite(config.orbWeight) || config.orbWeight<=0 ||
        !std::isfinite(config.huberDelta) || config.huberDelta<=0 || config.maxIterations<=0)
        throw std::invalid_argument("Invalid optimizer configuration");
    config_ = config;
}

HybridPoseResult HybridPoseSolver::optimize(
    const Vec3d& initialRvec,
    const Vec3d& initialTvec,
    const vector<DiamondObservation>& diamondInput,
    const vector<ORBObservation>& orbInput
) {
    HybridPoseResult result;

    if (!paramsSet_) {
        throw std::invalid_argument("Camera parameters not set");
    }
    auto diamondObs = diamondInput;
    auto orbObs = orbInput;
    // Native v2 receives distorted pixels. EdgeSE3ProjectXYZ is pinhole-only.
    auto undistortObservations = [&](auto& observations) {
        if (observations.empty()) return;
        vector<Point2d> raw, corrected;
        for (const auto& obs : observations) raw.emplace_back(obs.point2d.x(), obs.point2d.y());
        undistortPoints(raw, corrected, cameraMatrix_, distCoeffs_, noArray(), cameraMatrix_,
                        TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 50, 1e-12));
        for (size_t i=0; i<observations.size(); ++i)
            observations[i].point2d = Vector2d(corrected[i].x, corrected[i].y);
    };
    undistortObservations(diamondObs);
    undistortObservations(orbObs);

    // Setup G2O Optimizer
    typedef BlockSolver<BlockSolverTraits<6, 3>> BlockSolverType;
    typedef LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(
            std::make_unique<LinearSolverType>()
        )
    );

    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(config_.verbose);

    // 2. Add Camera Pose Vertex (SE3)
    // Convert rvec, tvec → SE3Quat
    Mat R_mat;
    Mat rvec_mat = (Mat_<double>(3,1) << initialRvec[0], initialRvec[1], initialRvec[2]);
    Rodrigues(rvec_mat, R_mat);

    Matrix3d R_eigen;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_eigen(i, j) = R_mat.at<double>(i, j);

    Vector3d t_eigen(initialTvec[0], initialTvec[1], initialTvec[2]);
    SE3Quat pose_se3(R_eigen, t_eigen);
    if (diamondObs.size()!=4) throw std::invalid_argument("Expected four Diamond observations");
    const Vector3d a=diamondObs[1].point3d-diamondObs[0].point3d;
    const Vector3d b=diamondObs[2].point3d-diamondObs[0].point3d;
    if (a.cross(b).norm()<1e-12) throw std::invalid_argument("Degenerate Diamond geometry");
    auto validateDepth = [&](const auto& observations) {
        for (const auto& obs : observations)
            if (!obs.point3d.allFinite() || !obs.point2d.allFinite() ||
                pose_se3.map(obs.point3d).z()<=1e-6)
                throw std::invalid_argument("Nonfinite observation or nonpositive initial depth");
    };
    validateDepth(diamondObs); validateDepth(orbObs);

    VertexSE3Expmap* vPose = new VertexSE3Expmap();
    vPose->setId(0);
    vPose->setEstimate(pose_se3);
    optimizer.addVertex(vPose);

    int vertexId = 1;

    // Add Diamond Marker Edges HIGH WEIGHT
    // Information is an explicit inverse pixel-variance tuning parameter.
    for (const auto& obs : diamondObs) {
        // Fixed 3D point vertex
        VertexPointXYZ* vPoint = new VertexPointXYZ();
        vPoint->setId(vertexId);
        vPoint->setEstimate(obs.point3d);
        vPoint->setFixed(true);  // 3D point is known exactly
        optimizer.addVertex(vPoint);

        EdgeSE3ProjectXYZ* edge = new EdgeSE3ProjectXYZ();
        edge->setVertex(0, vPoint);
        edge->setVertex(1, vPose);
        edge->setMeasurement(obs.point2d);
        edge->fx = fx_;
        edge->fy = fy_;
        edge->cx = cx_;
        edge->cy = cy_;

        // HIGH WEIGHT for Diamond
        // Information Matrix = Ω, chi2 = e^T * Ω * e
        // Higher Ω → this constraint dominates the optimization
        Matrix2d info = Matrix2d::Identity() * config_.diamondWeight;
        edge->setInformation(info);

        optimizer.addEdge(edge);

        vertexId++;
    }

    // Add ORB Feature Edges NORMAL WEIGHT, HUBER KERNEL
    vector<EdgeSE3ProjectXYZ*> orbEdges;

    for (const auto& obs : orbObs) {
        // Fixed 3D point from stereo triangulation
        VertexPointXYZ* vPoint = new VertexPointXYZ();
        vPoint->setId(vertexId);
        vPoint->setEstimate(obs.point3d);
        vPoint->setFixed(true);
        optimizer.addVertex(vPoint);

        // Projection edge
        EdgeSE3ProjectXYZ* edge = new EdgeSE3ProjectXYZ();
        edge->setVertex(0, vPoint);
        edge->setVertex(1, vPose);
        edge->setMeasurement(obs.point2d);
        edge->fx = fx_;
        edge->fy = fy_;
        edge->cx = cx_;
        edge->cy = cy_;

        // NORMAL WEIGHT for ORB
        Matrix2d info = Matrix2d::Identity() * config_.orbWeight;
        edge->setInformation(info);

        // HUBER ROBUST KERNEL for ORB
        // Errors < delta: quadratic penalty (normal least squares)
        // Errors > delta: linear penalty (outlier suppression)
        RobustKernelHuber* huber = new RobustKernelHuber();
        huber->setDelta(config_.huberDelta);
        edge->setRobustKernel(huber);

        optimizer.addEdge(edge);

        orbEdges.push_back(edge);
        vertexId++;
    }

    // Run Optimization
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    result.initialChi2 = optimizer.activeChi2();
    result.initialRobustCost = optimizer.activeRobustChi2();

    result.iterations = optimizer.optimize(config_.maxIterations);
    optimizer.computeActiveErrors();  // Recompute after any rejected/rolled-back LM step.
    result.finalChi2 = optimizer.activeChi2();
    result.finalRobustCost = optimizer.activeRobustChi2();

    // Extract Optimized Pose
    SE3Quat optimized_se3 = vPose->estimate();
    result.rotation = optimized_se3.rotation().toRotationMatrix();
    result.translation = optimized_se3.translation();

    // Convert back to rvec, tvec
    Mat R_out(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_out.at<double>(i, j) = result.rotation(i, j);

    Mat rvec_out;
    Rodrigues(R_out, rvec_out);
    result.rvec = Vec3d(rvec_out.at<double>(0), rvec_out.at<double>(1), rvec_out.at<double>(2));
    result.tvec = Vec3d(result.translation.x(), result.translation.y(), result.translation.z());

    // Count inliers/outliers based on Huber threshold
    double sumOrbInlierError = 0.0;

    vector<Eigen::Vector3d> diamondPts3d;
    vector<Eigen::Vector2d> diamondPts2d;
    for (const auto& obs : diamondObs) {
        diamondPts3d.push_back(obs.point3d);
        diamondPts2d.push_back(obs.point2d);
    }
    result.diamondError = computeReprojError(result.rvec, result.tvec, diamondPts3d, diamondPts2d);

    // ORB errors and inlier counting
    for (auto* edge : orbEdges) {
        edge->computeError();
        double err = edge->chi2(); 

        if (sqrt(err) < config_.huberDelta) {
            result.numOrbInliers++;
            sumOrbInlierError += sqrt(err / config_.orbWeight);
        } else {
            result.numOrbOutliers++;
        }
    }

    if (result.numOrbInliers > 0) {
        result.orbMeanError = sumOrbInlierError / result.numOrbInliers;
    } else {
        result.orbMeanError = std::numeric_limits<double>::quiet_NaN();
    }

    // Iteration count is NOT a convergence certificate. Check local numerical
    // stationarity of the actual robust objective under left SE(3) perturbations.
    const SE3Quat finalPose = vPose->estimate();
    const double cost = result.finalRobustCost;
    double maximumScaledGradient = 0;
    const double eps = 1e-6;
    for (int j=0; j<6; ++j) {
        Vector6 step = Vector6::Zero(); step[j] = eps;
        vPose->setEstimate(SE3Quat::exp(step) * finalPose);
        optimizer.computeActiveErrors(); double plus = optimizer.activeRobustChi2();
        vPose->setEstimate(SE3Quat::exp(-step) * finalPose);
        optimizer.computeActiveErrors(); double minus = optimizer.activeRobustChi2();
        double gradient = (plus - minus) / (4 * eps);
        double curvature = std::max(1.0, (plus + minus - 2 * cost) / (2 * eps * eps));
        maximumScaledGradient = std::max(maximumScaledGradient, std::abs(gradient) / std::sqrt(curvature));
        if (!std::isfinite(plus) || !std::isfinite(minus)) maximumScaledGradient = std::numeric_limits<double>::infinity();
    }
    vPose->setEstimate(finalPose);
    optimizer.computeActiveErrors();
    result.converged = std::isfinite(cost) && maximumScaledGradient < 1e-4 &&
                       cost <= result.initialRobustCost + 1e-8 * std::max(1.0, result.initialRobustCost);
    result.status = result.converged ? "numerically_stationary" : "not_stationary_or_cost_increased";
    return result;
}

HybridPoseResult HybridPoseSolver::optimizeDiamondOnly(
    const Vec3d& initialRvec,
    const Vec3d& initialTvec,
    const vector<DiamondObservation>& diamondObs
) {
    vector<ORBObservation> emptyOrb;
    return optimize(initialRvec, initialTvec, diamondObs, emptyOrb);
}

double HybridPoseSolver::computeReprojError(
    const Vec3d& rvec,
    const Vec3d& tvec,
    const vector<Eigen::Vector3d>& points3d,
    const vector<Eigen::Vector2d>& points2d
) {
    if (points3d.empty()) return 0.0;

    vector<Point3d> objPts;
    for (const auto& p : points3d) {
        objPts.push_back(Point3d(p.x(), p.y(), p.z()));
    }

    vector<Point2d> imgPts;
    for (const auto& p : points2d) {
        imgPts.push_back(Point2d(p.x(), p.y()));
    }

    // Project without distortion (observations are pre-undistorted)
    vector<Point2d> projected;
    Mat zeroDist;
    projectPoints(objPts, rvec, tvec, cameraMatrix_, zeroDist, projected);

    double totalError = 0.0;
    for (size_t i = 0; i < projected.size(); i++) {
        double dx = projected[i].x - imgPts[i].x;
        double dy = projected[i].y - imgPts[i].y;
        totalError += sqrt(dx * dx + dy * dy);
    }

    return totalError / projected.size();
}
