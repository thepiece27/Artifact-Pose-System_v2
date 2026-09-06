#include "stereo_triangulation.h"

using namespace cv;
using namespace std;
using namespace Eigen;

StereoTriangulator::StereoTriangulator()
    : fx_(0), fy_(0), cx_(0), cy_(0), paramsSet_(false) {}

void StereoTriangulator::setCameraParams(const Mat& cameraMatrix, const Mat& distCoeffs) {
    cameraMatrix_ = cameraMatrix.clone();
    distCoeffs_ = distCoeffs.clone();

    fx_ = cameraMatrix.at<double>(0, 0);
    fy_ = cameraMatrix.at<double>(1, 1);
    cx_ = cameraMatrix.at<double>(0, 2);
    cy_ = cameraMatrix.at<double>(1, 2);

    paramsSet_ = true;
    computeProjectionMatrices();
}

void StereoTriangulator::setConfig(const StereoConfig& config) {
    if (!std::isfinite(config.baseline) || config.baseline <= 0 ||
        cv::norm(config.baselineDirection - cv::Vec3d(1,0,0)) > 1e-12)
        CV_Error(Error::StsBadArg, "Native compatibility stereo supports positive horizontal baseline only");
    config_ = config;
    if (paramsSet_) {
        computeProjectionMatrices();
    }
}

void StereoTriangulator::computeProjectionMatrices() {
    // Left camera: at origin, identity rotation
    // P1 = K * [I | 0]
    Mat Rt1 = Mat::eye(3, 4, CV_64F);
    P1_ = cameraMatrix_ * Rt1;

    // P2 = K * [I | t] where t = baseline * direction
    Mat Rt2 = Mat::eye(3, 4, CV_64F);
    Vec3d t = config_.baseline * config_.baselineDirection;
    Rt2.at<double>(0, 3) = -t[0]; 
    Rt2.at<double>(1, 3) = -t[1];
    Rt2.at<double>(2, 3) = -t[2];
    P2_ = cameraMatrix_ * Rt2;
}

TriangulatedPoint StereoTriangulator::triangulatePoint(
    const Point2f& left,
    const Point2f& right
) {
    TriangulatedPoint result;
    result.pointLeft = left;
    result.pointRight = right;
    result.valid = false;
    if (!paramsSet_) CV_Error(Error::StsBadArg, "Camera parameters not set");
    vector<Point2d> rawLeft{{left.x,left.y}}, rawRight{{right.x,right.y}}, ul, ur;
    const TermCriteria criteria(TermCriteria::COUNT | TermCriteria::EPS,50,1e-12);
    undistortPoints(rawLeft, ul, cameraMatrix_, distCoeffs_, noArray(), cameraMatrix_, criteria);
    undistortPoints(rawRight, ur, cameraMatrix_, distCoeffs_, noArray(), cameraMatrix_, criteria);
    const auto l = ul[0], r = ur[0];

    // Compute disparity (only meaningful for horizontal baseline)
    result.disparity = l.x - r.x;

    if (result.disparity < config_.minDisparity || abs(l.y-r.y)/std::sqrt(2.0) > config_.maxEpipolarError) {
        return result; 
    }

    // Triangulate using DLT method
    // Build the 4x4 system: A * X = 0
    Mat A(4, 4, CV_64F);

    // Row 0, 1: from left camera
    Mat(l.x * P1_.row(2) - P1_.row(0)).copyTo(A.row(0));
    Mat(l.y * P1_.row(2) - P1_.row(1)).copyTo(A.row(1));

    // Row 2, 3: from right camera
    Mat(r.x * P2_.row(2) - P2_.row(0)).copyTo(A.row(2));
    Mat(r.y * P2_.row(2) - P2_.row(1)).copyTo(A.row(3));

    // SVD to find null space
    Mat w, u, vt;
    SVD::compute(A, w, u, vt, SVD::FULL_UV);

    // Solution is the last row of Vt
    Mat X = vt.row(3).t();

    double W = X.at<double>(3);
    if (abs(W) < 1e-10) {
        return result; 
    }

    double X3d = X.at<double>(0) / W;
    double Y3d = X.at<double>(1) / W;
    double Z3d = X.at<double>(2) / W;

    if (!std::isfinite(X3d) || !std::isfinite(Y3d) || !std::isfinite(Z3d) ||
        Z3d < config_.minDepth || Z3d > config_.maxDepth) {
        return result; 
    }
    Vec3d rayLeft((l.x-cx_)/fx_, (l.y-cy_)/fy_, 1);
    Vec3d rayRight((r.x-cx_)/fx_, (r.y-cy_)/fy_, 1);
    double cosine = rayLeft.dot(rayRight)/(cv::norm(rayLeft)*cv::norm(rayRight));
    double angle = std::acos(std::max(-1.0,std::min(1.0,cosine)))*180.0/CV_PI;
    if (angle < config_.minParallaxDegrees) return result;

    result.point3d = Vector3d(X3d, Y3d, Z3d);
    result.depth = Z3d;

    // Compute reprojection error
    // Project 3D point back to both images
    vector<Point3f> pt3d = {Point3f(X3d, Y3d, Z3d)};

    vector<Point2f> projLeft, projRight;
    Vec3d rvecZero(0, 0, 0);
    Vec3d tvecZero(0, 0, 0);
    Vec3d tvecRight = -(config_.baseline * config_.baselineDirection);

    projectPoints(pt3d, rvecZero, tvecZero, cameraMatrix_, distCoeffs_, projLeft);
    projectPoints(pt3d, rvecZero, tvecRight, cameraMatrix_, distCoeffs_, projRight);

    double errLeft = norm(Point2f(projLeft[0].x - left.x, projLeft[0].y - left.y));
    double errRight = norm(Point2f(projRight[0].x - right.x, projRight[0].y - right.y));
    result.reprojError = (errLeft + errRight) / 2.0;

    // Quality check
    if (std::max(errLeft,errRight) <= config_.maxReprojError) {
        result.valid = true;
    }

    return result;
}

StereoTriangulationResult StereoTriangulator::triangulate(
    const vector<Point2f>& pointsLeft,
    const vector<Point2f>& pointsRight
) {
    StereoTriangulationResult result;
    result.numValid = 0;
    result.numRejected = 0;
    result.meanDepth = 0;
    result.meanReprojError = 0;

    if (!paramsSet_ || pointsLeft.size() != pointsRight.size()) {
        return result;
    }

    double sumDepth = 0;
    double sumError = 0;

    for (size_t i = 0; i < pointsLeft.size(); i++) {
        TriangulatedPoint tp = triangulatePoint(pointsLeft[i], pointsRight[i]);
        result.points.push_back(tp);

        if (tp.valid) {
            result.numValid++;
            sumDepth += tp.depth;
            sumError += tp.reprojError;
        } else {
            result.numRejected++;
        }
    }

    if (result.numValid > 0) {
        result.meanDepth = sumDepth / result.numValid;
        result.meanReprojError = sumError / result.numValid;
    }

    return result;
}
