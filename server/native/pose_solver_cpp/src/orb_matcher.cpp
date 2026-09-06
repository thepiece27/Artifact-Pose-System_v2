#include "orb_matcher.h"

using namespace cv;
using namespace std;
using namespace Eigen;

ORBMatcher::ORBMatcher() {
    initMatcher();
}

void ORBMatcher::setConfig(const MatchConfig& config) {
    config_ = config;
    initMatcher();
}

void ORBMatcher::initMatcher() {
    if (config_.method == MatchConfig::BF_HAMMING) {
        matcher_ = BFMatcher::create(NORM_HAMMING, false); // Explicit mutual KNN below.
    } else {
        matcher_ = makePtr<FlannBasedMatcher>(
            makePtr<flann::LshIndexParams>(12, 20, 2)
        );
    }
}

vector<DMatch> ORBMatcher::matchBruteForce(
    const Mat& desc1,
    const Mat& desc2
) {
    if (desc1.rows < 2 || desc2.rows < 2) {
        return {};
    }

    vector<vector<DMatch>> knnMatches;
    matcher_->knnMatch(desc1, desc2, knnMatches, 2);

    auto forward = filterByRatioTest(knnMatches);
    vector<vector<DMatch>> reverseKnn;
    matcher_->knnMatch(desc2, desc1, reverseKnn, 2);
    auto reverse = filterByRatioTest(reverseKnn);
    vector<int> reverseIndex(desc2.rows, -1);
    for (const auto& m : reverse) reverseIndex[m.queryIdx] = m.trainIdx;
    vector<DMatch> mutual;
    for (const auto& m : forward)
        if (reverseIndex[m.trainIdx] == m.queryIdx) mutual.push_back(m);
    return mutual;
}

vector<DMatch> ORBMatcher::filterByRatioTest(
    const vector<vector<DMatch>>& knnMatches
) {
    vector<DMatch> goodMatches;

    for (const auto& m : knnMatches) {
        if (m.size() < 2) continue;

        // Lowe's ratio test
        if (m[0].distance < config_.ratioThreshold * m[1].distance) {
            // Additional filter: max Hamming distance
            if (m[0].distance <= config_.maxHammingDistance) {
                goodMatches.push_back(m[0]);
            }
        }
    }

    return goodMatches;
}

Match3D2DResult ORBMatcher::matchWith3DReference(
    const vector<KeyPoint>& currentKeypoints,
    const Mat& currentDescriptors,
    const vector<Eigen::Vector3d>& refPoints3d,
    const Mat& refDescriptors
) {
    Match3D2DResult result;
    result.numMatches = 0;
    result.meanDistance = 0;

    if (currentDescriptors.empty() || refDescriptors.empty()) {
        return result;
    }

    if (refPoints3d.size() != (size_t)refDescriptors.rows || currentKeypoints.size() != (size_t)currentDescriptors.rows) {
        return result;  // Mismatch between 3D points and descriptors
    }

    // Match current → reference
    vector<DMatch> goodMatches = matchBruteForce(currentDescriptors, refDescriptors);

    // Build 3D-2D correspondences
    float sumDist = 0;
    for (const auto& m : goodMatches) {
        Match3D2D match;
        match.point3d = refPoints3d[m.trainIdx];
        match.point2d = Vector2d(
            currentKeypoints[m.queryIdx].pt.x,
            currentKeypoints[m.queryIdx].pt.y
        );
        match.refIdx = m.trainIdx;
        match.matchDistance = m.distance;

        result.matches.push_back(match);
        sumDist += m.distance;
    }

    result.numMatches = (int)result.matches.size();
    if (result.numMatches > 0) {
        result.meanDistance = sumDist / result.numMatches;
    }

    return result;
}
