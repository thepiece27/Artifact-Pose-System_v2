#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstring>
#include <limits>
#include "quadtree.h"
#include "orb_matcher.h"
#include "stereo_triangulation.h"
#include "hybrid_pose_solver.h"
#include "deviation_calculator.h"
namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Bytes = py::array_t<uint8_t, py::array::c_style>;

static void positive(double v, const char* name) {
    if (!std::isfinite(v) || v <= 0) throw py::value_error(std::string(name) + " must be positive and finite");
}
static void matrix(const Array& a, py::ssize_t cols, const char* name) {
    if (a.ndim() != 2 || a.shape(1) != cols) throw py::value_error(std::string(name) + " has invalid shape");
    for (py::ssize_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a.data()[i])) throw py::value_error(std::string(name) + " must be finite");
}
static cv::Vec3d vec(const Array& a) {
    if (a.size() != 3 || !(a.ndim() == 1 || (a.ndim() == 2 && (a.shape(0) == 1 || a.shape(1) == 1))))
        throw py::value_error("Pose vector must have exactly three entries");
    cv::Vec3d v(a.data()[0], a.data()[1], a.data()[2]);
    for (double x : {v[0], v[1], v[2]}) if (!std::isfinite(x)) throw py::value_error("Nonfinite pose");
    return v;
}
static std::pair<cv::Mat, cv::Mat> camera(const Array& k, const Array& d) {
    matrix(k, 3, "K");
    if (k.shape(0) != 3) throw py::value_error("K must be (3,3)");
    positive(k.data()[0], "fx"); positive(k.data()[4], "fy");
    if (std::abs(k.data()[1]) > 1e-12 || std::abs(k.data()[3]) > 1e-12 ||
        std::abs(k.data()[6]) > 1e-12 || std::abs(k.data()[7]) > 1e-12 || std::abs(k.data()[8]-1) > 1e-12)
        throw py::value_error("K must be zero-skew pinhole intrinsics");
    const auto n = d.size();
    if (!(n == 4 || n == 5 || n == 8 || n == 12 || n == 14) ||
        !(d.ndim() == 1 || (d.ndim() == 2 && (d.shape(0) == 1 || d.shape(1) == 1))))
        throw py::value_error("Invalid distortion vector");
    for (py::ssize_t i=0; i<n; ++i) if (!std::isfinite(d.data()[i])) throw py::value_error("Nonfinite distortion");
    return {cv::Mat(3,3,CV_64F,const_cast<double*>(k.data())).clone(),
            cv::Mat(1,static_cast<int>(n),CV_64F,const_cast<double*>(d.data())).clone()};
}
static cv::Mat desc(const Bytes& a) {
    if (a.ndim() != 2 || a.shape(1) != 32) throw py::value_error("Descriptors must be uint8 (N,32)");
    return cv::Mat(static_cast<int>(a.shape(0)),32,CV_8U,const_cast<uint8_t*>(a.data())).clone();
}
static py::array_t<double> array3(const cv::Vec3d& v) {
    py::array_t<double> a(3);
    std::memcpy(a.mutable_data(), v.val, 3*sizeof(double)); return a;
}
static std::vector<cv::Point2f> pixels(const Array& a) {
    matrix(a, 2, "pixels"); std::vector<cv::Point2f> p;
    for (py::ssize_t i=0; i<a.shape(0); ++i) p.emplace_back(a.data()[2*i],a.data()[2*i+1]);
    return p;
}
static py::list matches(const std::vector<cv::DMatch>& found) {
    py::list out;
    for (const auto& m:found) { py::dict d; d["queryIdx"]=m.queryIdx; d["trainIdx"]=m.trainIdx; d["distance"]=m.distance; out.append(d); }
    return out;
}

PYBIND11_MODULE(pose_solver_cpp, m) {
    m.doc() = "Validated native API v2. T_cw poses; distorted pixels; metres. No motor actuation.";
    m.attr("API_VERSION") = 2;
    m.def("extract_with_quadtree", [](Bytes image, int maximum, int minimum, int depth, int perLeaf) {
        if (image.size()==0 || !(image.ndim()==2 || (image.ndim()==3 && image.shape(2)==3)))
            throw py::value_error("Expected nonempty uint8 gray/BGR image");
        if (maximum<=0 || minimum<=0 || depth<=0 || depth>16 || perLeaf<=0) throw py::value_error("Invalid QuadTree limits");
        cv::Mat input(static_cast<int>(image.shape(0)),static_cast<int>(image.shape(1)),
                      image.ndim()==2 ? CV_8UC1 : CV_8UC3, image.mutable_data());
        auto r=extractWithQuadTree(input,maximum,minimum,depth,perLeaf);
        py::dict out; py::list kp, cells;
        for (const auto& k:r.keypoints) {
            py::dict item; item["x"]=k.pt.x; item["y"]=k.pt.y; item["size"]=k.size;
            item["angle"]=k.angle; item["response"]=k.response; item["octave"]=k.octave; kp.append(item);
        }
        py::array_t<uint8_t> descriptors({static_cast<py::ssize_t>(r.keypoints.size()),py::ssize_t(32)});
        if (!r.descriptors.empty()) std::memcpy(descriptors.mutable_data(),r.descriptors.data,r.descriptors.total());
        for (const auto& c:r.gridCells) cells.append(py::make_tuple(c.x,c.y,c.width,c.height));
        out["keypoints"]=kp; out["descriptors"]=descriptors; out["grid_cells"]=cells;
        out["total_detected"]=r.totalDetected; return out;
    }, py::arg("image").noconvert(), py::arg("max_features")=5000, py::arg("min_node_size")=64, py::arg("max_depth")=7, py::arg("features_per_leaf")=4);

    m.def("match_stereo", [](Bytes a, Bytes b, double ratio) {
        if (!(ratio>0 && ratio<1)) throw py::value_error("Invalid ratio");
        ORBMatcher matcher; MatchConfig cfg; cfg.ratioThreshold=ratio; matcher.setConfig(cfg);
        return matches(matcher.matchBruteForce(desc(a),desc(b)));
    }, py::arg("desc_left").noconvert(), py::arg("desc_right").noconvert(), py::arg("ratio_threshold")=0.75);

    m.def("match_with_3d_reference", [](Array kp, Bytes d, Array xyz, Bytes reference, double ratio) {
        matrix(kp,2,"keypoints"); matrix(xyz,3,"points_3d"); auto dc=desc(d), dr=desc(reference);
        if (kp.shape(0)!=dc.rows || xyz.shape(0)!=dr.rows) throw py::value_error("Descriptor/keypoint/landmark counts differ");
        if (!(ratio>0 && ratio<1)) throw py::value_error("Invalid ratio");
        ORBMatcher matcher; MatchConfig cfg; cfg.ratioThreshold=ratio; matcher.setConfig(cfg);
        auto found=matcher.matchBruteForce(dc,dr);
        py::array_t<double> p3({static_cast<py::ssize_t>(found.size()),py::ssize_t(3)});
        py::array_t<double> p2({static_cast<py::ssize_t>(found.size()),py::ssize_t(2)});
        for (size_t i=0;i<found.size();++i) {
            std::memcpy(p3.mutable_data()+i*3,xyz.data()+found[i].trainIdx*3,3*sizeof(double));
            std::memcpy(p2.mutable_data()+i*2,kp.data()+found[i].queryIdx*2,2*sizeof(double));
        }
        py::dict out; out["points_3d"]=p3; out["points_2d"]=p2; out["num_matches"]=found.size(); out["matches"]=matches(found); return out;
    }, py::arg("current_keypoints"),py::arg("current_descriptors").noconvert(),py::arg("ref_points_3d"),py::arg("ref_descriptors").noconvert(),py::arg("ratio_threshold")=0.75);

    m.def("triangulate_stereo", [](Array left, Array right, Array k, Array d, double baseline) {
        auto lp=pixels(left), rp=pixels(right); auto [K,D]=camera(k,d); positive(baseline,"baseline");
        if (lp.size()!=rp.size()) throw py::value_error("Stereo counts differ");
        StereoTriangulator tri; StereoConfig cfg; cfg.baseline=baseline; tri.setConfig(cfg); tri.setCameraParams(K,D);
        auto r=tri.triangulate(lp,rp);
        py::array_t<double> xyz({static_cast<py::ssize_t>(lp.size()),py::ssize_t(3)});
        py::array_t<bool> mask(lp.size());
        for(size_t i=0;i<lp.size();++i) {
            mask.mutable_data()[i]=r.points[i].valid;
            for(int j=0;j<3;++j) xyz.mutable_data()[i*3+j]=r.points[i].valid?r.points[i].point3d[j]:std::numeric_limits<double>::quiet_NaN();
        }
        py::dict out; out["points_3d"]=xyz; out["valid_mask"]=mask; out["num_valid"]=r.numValid;
        out["mean_depth"]=r.meanDepth; out["mean_reproj_error"]=r.meanReprojError; return out;
    },py::arg("points_left"),py::arg("points_right"),py::arg("camera_matrix"),py::arg("dist_coeffs"),py::arg("baseline"));

    m.def("hybrid_optimize", [](Array rv,Array tv,Array d3,Array d2,Array o3,Array o2,Array k,Array d,
                               double dw,double ow,double huber,int iterations) {
        auto r=vec(rv),t=vec(tv); auto [K,D]=camera(k,d);
        matrix(d3,3,"diamond_3d"); matrix(d2,2,"diamond_2d"); matrix(o3,3,"orb_3d"); matrix(o2,2,"orb_2d");
        if(d3.shape(0)!=4 || d2.shape(0)!=4 || o3.shape(0)!=o2.shape(0)) throw py::value_error("Invalid observation counts");
        positive(dw,"diamond_weight"); positive(ow,"orb_weight"); positive(huber,"huber_delta");
        if(iterations<=0 || iterations>10000) throw py::value_error("Invalid iteration limit");
        std::vector<DiamondObservation> diamond;
        std::vector<ORBObservation> orb;
        for(py::ssize_t i=0;i<4;++i) diamond.push_back({Eigen::Map<const Eigen::Vector3d>(d3.data()+i*3),Eigen::Map<const Eigen::Vector2d>(d2.data()+i*2)});
        for(py::ssize_t i=0;i<o3.shape(0);++i) orb.push_back({Eigen::Map<const Eigen::Vector3d>(o3.data()+i*3),Eigen::Map<const Eigen::Vector2d>(o2.data()+i*2)});
        HybridPoseSolver solver; solver.setCameraParams(K,D); HybridConfig cfg;
        cfg.diamondWeight=dw; cfg.orbWeight=ow; cfg.huberDelta=huber; cfg.maxIterations=iterations; solver.setConfig(cfg);
        auto fit=solver.optimize(r,t,diamond,orb);
        py::dict out; out["rvec"]=array3(fit.rvec); out["tvec"]=array3(fit.tvec);
        out["initial_chi2"]=fit.initialChi2; out["final_chi2"]=fit.finalChi2;
        out["initial_robust_cost"]=fit.initialRobustCost; out["final_robust_cost"]=fit.finalRobustCost;
        out["diamond_error"]=fit.diamondError; out["orb_mean_error"]=fit.orbMeanError;
        out["num_orb_inliers"]=fit.numOrbInliers; out["num_orb_outliers"]=fit.numOrbOutliers;
        out["iterations"]=fit.iterations; out["converged"]=fit.converged; out["status"]=fit.status; return out;
    },py::arg("initial_rvec"),py::arg("initial_tvec"),py::arg("diamond_points_3d"),py::arg("diamond_points_2d"),
       py::arg("orb_points_3d"),py::arg("orb_points_2d"),py::arg("camera_matrix"),py::arg("dist_coeffs"),
       py::arg("diamond_weight")=4.0,py::arg("orb_weight")=1.0/2.25,py::arg("huber_delta")=2.0,py::arg("max_iterations")=100);

    m.def("calculate_deviation", [](Array rg,Array tg,Array rc,Array tc,double tt,double rt,
                                    double steps,double servo,bool sequential) {
        positive(tt,"translation tolerance"); positive(rt,"rotation tolerance");
        positive(steps,"steps_per_mm"); positive(servo,"servo_min_deg"); (void)sequential;
        DeviationCalculator calc; DeviationConfig cfg; cfg.transTolerance=tt; cfg.rotTolerance=rt; calc.setConfig(cfg);
        auto v=calc.calculate(vec(rg),vec(tg),vec(rc),vec(tc));
        py::dict out; out["delta_x"]=v.deltaX; out["delta_y"]=v.deltaY; out["delta_z"]=v.deltaZ;
        out["translation_magnitude"]=v.translationMag; out["rotation_magnitude"]=v.rotationMag;
        out["within_tolerance"]=v.withinTolerance; out["within_translation_tolerance"]=v.withinTransTolerance;
        out["within_rotation_tolerance"]=v.withinRotTolerance; out["translation_frame"]="diamond_world_camera_center_current_minus_golden";
        py::dict cmd; cmd["enabled"]=false; cmd["reason"]="Actuator kinematics not calibrated"; out["motor_command"]=cmd; return out;
    },py::arg("rvec_golden"),py::arg("tvec_golden"),py::arg("rvec_current"),py::arg("tvec_current"),
       py::arg("trans_tolerance")=0.01,py::arg("rot_tolerance")=1.0,py::arg("steps_per_mm")=860.0,
       py::arg("servo_min_deg")=1.0,py::arg("sequential_mode")=true);
}
