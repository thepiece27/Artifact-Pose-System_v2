#include "quadtree.h"

using namespace cv;
using namespace std;

QuadNode::QuadNode(Rect2f bound, int d)
    : boundary(bound), isLeaf(true), depth(d) {}

QuadNode::~QuadNode() {
    for (auto* child : children) {
        delete child;
    }
}

//QuadTree

QuadTree::QuadTree(int imageWidth, int imageHeight, int minNodeSize, int maxDepth, int featuresPerLeaf)
    : minNodeSize(minNodeSize), maxDepth(maxDepth), featuresPerLeaf(featuresPerLeaf) {
    root = new QuadNode(Rect2f(0, 0, imageWidth, imageHeight), 0);
}

QuadTree::~QuadTree() {
    delete root;
}

void QuadTree::subdivide(QuadNode* node) {
    float halfW = node->boundary.width / 2.0f;
    float halfH = node->boundary.height / 2.0f;
    float x = node->boundary.x;
    float y = node->boundary.y;
    int nextDepth = node->depth + 1;

    if (halfW < minNodeSize || halfH < minNodeSize) {
        return;
    }

    if (nextDepth > maxDepth) {
        return;
    }

    node->children.push_back(new QuadNode(Rect2f(x, y, halfW, halfH), nextDepth));               // TL
    node->children.push_back(new QuadNode(Rect2f(x + halfW, y, halfW, halfH), nextDepth));       // TR
    node->children.push_back(new QuadNode(Rect2f(x, y + halfH, halfW, halfH), nextDepth));       // BL
    node->children.push_back(new QuadNode(Rect2f(x + halfW, y + halfH, halfW, halfH), nextDepth)); // BR
    node->isLeaf = false;

    for (const auto& kp : node->keypoints) {
        for (auto* child : node->children) {
            if (child->boundary.contains(Point2f(kp.pt.x, kp.pt.y))) {
                child->keypoints.push_back(kp);
                break;
            }
        }
    }

    node->keypoints.clear();

    // Đệ quy: tiếp tục chia các ô con có nhiều hơn 1 keypoint
    for (auto* child : node->children) {
        if (child->keypoints.size() > 1) {
            subdivide(child);
        }
    }
}

// Thu thập keypoint tốt nhất (response cao nhất) từ mỗi leaf node
void QuadTree::collectBest(QuadNode* node, vector<KeyPoint>& result) {
    if (node->isLeaf) {
        if (!node->keypoints.empty()) {
            // Chọn keypoint có response mạnh nhất trong ô này
            auto ranked = node->keypoints;
            stable_sort(ranked.begin(), ranked.end(),
                [](const KeyPoint& a, const KeyPoint& b) {
                    return a.response > b.response;
                });
            const auto count = std::min(ranked.size(), static_cast<size_t>(featuresPerLeaf));
            result.insert(result.end(), ranked.begin(), ranked.begin()+count);
        }
        return;
    }

    for (auto* child : node->children) {
        collectBest(child, result);
    }
}

// Phân phối keypoints bằng quadtree
vector<KeyPoint> QuadTree::distribute(vector<KeyPoint>& keypoints) {
    for (auto* child : root->children) delete child;
    root->children.clear();
    root->isLeaf = true;
    root->keypoints = keypoints;
    subdivide(root);
    vector<KeyPoint> result;
    collectBest(root, result);
    return result;
}

void QuadTree::collectLeaves(QuadNode* node, vector<Rect2f>& leaves) const {
    if (node->isLeaf) {
        leaves.push_back(node->boundary);
        return;
    }
    for (auto* child : node->children) {
        collectLeaves(child, leaves);
    }
}

vector<Rect2f> QuadTree::getLeafBoundaries() const {
    vector<Rect2f> leaves;
    collectLeaves(root, leaves);
    return leaves;
}

QuadTreeResult extractWithQuadTree(
    const Mat& image,
    int maxInitialFeatures,
    int minNodeSize,
    int maxDepth,
    int featuresPerLeaf
) {
    QuadTreeResult result;
    if (image.empty() || image.depth() != CV_8U || (image.channels()!=1 && image.channels()!=3) ||
        maxInitialFeatures<=0 || minNodeSize<=0 || maxDepth<=0 || maxDepth>16 || featuresPerLeaf<=0)
        CV_Error(Error::StsBadArg, "Invalid image or QuadTree configuration");

    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    Ptr<ORB> orb = ORB::create(maxInitialFeatures);
    vector<KeyPoint> allKeypoints;
    orb->detect(gray, allKeypoints);
    result.totalDetected = allKeypoints.size();

    QuadTree qt(image.cols, image.rows, minNodeSize, maxDepth, featuresPerLeaf);
    result.keypoints = qt.distribute(allKeypoints);
    result.gridCells = qt.getLeafBoundaries();

    orb->compute(gray, result.keypoints, result.descriptors);

    return result;
}
