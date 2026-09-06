#ifndef DEVIATION_CALCULATOR_H
#define DEVIATION_CALCULATOR_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct PoseDeviation {
    // === Translation Deviations (meters) ===
    double deltaX;
    double deltaY;
    double deltaZ;

    double deltaPan;
    double deltaTilt;
    double deltaRoll;

    double translationMag;   // sqrt(dX^2 + dY^2 + dZ^2)
    double rotationMag;      // SO(3) geodesic angle in degrees (Euler is display-only)

    bool withinTransTolerance;
    bool withinRotTolerance;
    bool withinTolerance;

    PoseDeviation() :
        deltaX(0), deltaY(0), deltaZ(0),
        deltaPan(0), deltaTilt(0), deltaRoll(0),
        translationMag(0), rotationMag(0),
        withinTransTolerance(false), withinRotTolerance(false), withinTolerance(false) {}
};

struct DeviationConfig {
    double transTolerance = 0.010;  // 10mm — relaxed to match hardware precision
    double rotTolerance = 1.0;      // 1.0 deg — must be >= servo minimum step

    // Motor hardware constraints
    double servoMinDeg = 1.0;       // Minimum servo step (degrees); angles smaller
                                    // than half this are zeroed (dead zone)
    bool sequentialMode = true;     // If true: when both trans+rot needed, send
                                    // translation first; rotation on next iteration
    double stepsPerMm = 860.0;      // Stepper motor steps per mm
};


class DeviationCalculator {
public:
    DeviationCalculator();

    void setConfig(const DeviationConfig& config);

    /* Calculate deviation between current and golden pose
        Input: rvec, tvec for both poses (OpenCV format)
        Output: PoseDeviation with all components
       Math:
        R_deviation = R_current * R_golden^(-1)
        C_deviation = -R_current^T*T_current + R_golden^T*T_golden
        Euler angles extracted from R_deviation
    */

    PoseDeviation calculate(
        const cv::Vec3d& rvecGolden,
        const cv::Vec3d& tvecGolden,
        const cv::Vec3d& rvecCurrent,
        const cv::Vec3d& tvecCurrent
    );

    // Convenience: calculate from Eigen format
    PoseDeviation calculate(
        const Eigen::Matrix3d& R_golden,
        const Eigen::Vector3d& T_golden,
        const Eigen::Matrix3d& R_current,
        const Eigen::Vector3d& T_current
    );

    // Extract Euler angles from rotation matrix
    void rotationToEuler(
        const Eigen::Matrix3d& R,
        double& roll,   
        double& pitch, 
        double& yaw  
    );

private:
    DeviationConfig config_;
};

// MotorCommand: Suggested motor movements

struct MotorCommand {

    double moveX = 0;
    double moveZ = 0;

    double rotatePan = 0; 
    double rotateTilt = 0;
    int priority = 0;
    // Priority: 0 = no movement, 1 = translation, 2 = rotation, 3 = both
};

// Disabled compatibility function: returns a zero command until kinematics exist.
// - Rotation angles are rounded to nearest servoMinDeg; angles in the
//   dead zone (|angle| < 0.5 * servoMinDeg) are zeroed.
// - Sequential mode: when both translation and rotation are needed,
//   only translation is sent this step (rotation deferred to next iteration).
MotorCommand deviationToMotorCommand(
    const PoseDeviation& dev,
    double stepsPerMm    = 860.0,  // Stepper motor steps/mm
    double servoMinDeg   = 1.0,    // Minimum servo step (degrees)
    bool   sequentialMode = true   // Translation-first priority
);

#endif
