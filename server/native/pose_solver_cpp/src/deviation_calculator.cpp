#include "deviation_calculator.h"
#include <cmath>
#include <stdexcept>

namespace { constexpr double RAD2DEG = 180.0 / CV_PI; }
DeviationCalculator::DeviationCalculator() = default;

void DeviationCalculator::setConfig(const DeviationConfig& config) {
    if (!std::isfinite(config.transTolerance) || !std::isfinite(config.rotTolerance) ||
        config.transTolerance <= 0 || config.rotTolerance <= 0)
        throw std::invalid_argument("Tolerances must be finite and positive");
    config_ = config;
}

PoseDeviation DeviationCalculator::calculate(const cv::Vec3d& rg, const cv::Vec3d& tg,
                                             const cv::Vec3d& rc, const cv::Vec3d& tc) {
    cv::Mat Rgcv, Rccv;
    cv::Rodrigues(rg, Rgcv); cv::Rodrigues(rc, Rccv);
    Eigen::Matrix3d Rg, Rc;
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
        Rg(i,j)=Rgcv.at<double>(i,j); Rc(i,j)=Rccv.at<double>(i,j);
    }
    return calculate(Rg,Eigen::Map<const Eigen::Vector3d>(tg.val),Rc,Eigen::Map<const Eigen::Vector3d>(tc.val));
}

PoseDeviation DeviationCalculator::calculate(const Eigen::Matrix3d& Rg, const Eigen::Vector3d& tg,
                                             const Eigen::Matrix3d& Rc, const Eigen::Vector3d& tc) {
    if (!Rg.allFinite() || !Rc.allFinite() || !tg.allFinite() || !tc.allFinite())
        throw std::invalid_argument("Nonfinite pose");
    PoseDeviation dev;
    Eigen::Vector3d displacement = -Rc.transpose()*tc + Rg.transpose()*tg;
    dev.deltaX=displacement.x(); dev.deltaY=displacement.y(); dev.deltaZ=displacement.z();
    dev.translationMag=displacement.norm();
    Eigen::Matrix3d correctionRotation=Rc*Rg.transpose();
    // Euler components are display-only; tolerance uses the SO(3) geodesic angle.
    rotationToEuler(correctionRotation,dev.deltaTilt,dev.deltaPan,dev.deltaRoll);
    dev.rotationMag=Eigen::AngleAxisd(correctionRotation).angle()*RAD2DEG;
    dev.withinTransTolerance=dev.translationMag<=config_.transTolerance;
    dev.withinRotTolerance=dev.rotationMag<=config_.rotTolerance;
    dev.withinTolerance=dev.withinTransTolerance && dev.withinRotTolerance;
    return dev;
}

void DeviationCalculator::rotationToEuler(const Eigen::Matrix3d& R,double& roll,double& pitch,double& yaw) {
    double sy=std::hypot(R(0,0),R(1,0));
    roll=std::atan2(sy<1e-6 ? -R(1,2) : R(2,1),sy<1e-6 ? R(1,1) : R(2,2))*RAD2DEG;
    pitch=std::atan2(-R(2,0),sy)*RAD2DEG;
    yaw=sy<1e-6 ? 0 : std::atan2(R(1,0),R(0,0))*RAD2DEG;
}

MotorCommand deviationToMotorCommand(const PoseDeviation&,double,double,bool) {
    // Uncalibrated actuator mapping is unsafe even if the visual pose is correct.
    return MotorCommand{};
}
