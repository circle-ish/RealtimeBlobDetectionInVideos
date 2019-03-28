#ifndef AUXFUNS_H
#define AUXFUNS_H

#include "BasicBlob.h"
#include "BlobList.h"

#include <map>
#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

typedef std::vector<std::vector<cv::Point> > Contours;

CLASS classify(std::vector<double> *feat);
std::vector<double> SVM_predict(BlobList *pBlobList, std::map<int, std::vector<double> > &blobs);
int extractFeature(BlobList * pBlobList, cv::Mat &blobMask, Contours contours);

inline bool lessX(const cv::Point& p1, const cv::Point& p2) {
    return p1.x < p2.x;
}

inline bool lessY(const cv::Point& p1, const cv::Point& p2) {
    return p1.y < p2.y;
}

#endif // AUXFUNS_H
