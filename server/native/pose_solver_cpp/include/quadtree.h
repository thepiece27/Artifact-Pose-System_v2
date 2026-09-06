#ifndef QUADTREE_H
#define QUADTREE_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

// Node in the quadtree
struct QuadNode {
    cv::Rect2f boundary;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<QuadNode*> children;
    bool isLeaf;
    int depth;

    QuadNode(cv::Rect2f bound, int d);
    ~QuadNode();
    QuadNode(const QuadNode&) = delete;
    QuadNode& operator=(const QuadNode&) = delete;
};

// Quadtree distributes keypoints evenly across the image
class QuadTree {
public:
    QuadTree(int imageWidth, int imageHeight, int minNodeSize = 64, int maxDepth = 7, int featuresPerLeaf = 4);
    ~QuadTree();
    QuadTree(const QuadTree&) = delete;
    QuadTree& operator=(const QuadTree&) = delete;

    std::vector<cv::KeyPoint> distribute(std::vector<cv::KeyPoint>& keypoints);
    std::vector<cv::Rect2f> getLeafBoundaries() const;

private:
    QuadNode* root;
    int minNodeSize;
    int maxDepth;
    int featuresPerLeaf;

    void subdivide(QuadNode* node);
    void collectBest(QuadNode* node, std::vector<cv::KeyPoint>& result);
    void collectLeaves(QuadNode* node, std::vector<cv::Rect2f>& leaves) const;
};

// Main function: extract ORB keypoints + distribute using Quadtree
struct QuadTreeResult {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Rect2f> gridCells;
    int totalDetected;
};

QuadTreeResult extractWithQuadTree(
    const cv::Mat& image,
    int maxInitialFeatures = 5000,
    int minNodeSize = 64,
    int maxDepth = 7,
    int featuresPerLeaf = 4
);

#endif
