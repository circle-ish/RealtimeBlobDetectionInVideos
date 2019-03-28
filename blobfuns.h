#ifndef BLOBFUNS_H_INCLUDE
#define BLOBFUNS_H_INCLUDE

#include "BasicBlob.h"
#include "BlobList.h"
#include "auxfuns.h"
#include "fgseg.h"

#include <iostream>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"

//blob extraction functions (included in 'blobextrac.cpp')
int extractBlobs(cv::Mat &fgmask, BlobList *pBlobList, Contours &contours);
IplImage *paintBlobImage(IplImage* frame, BlobList *pBlobList);

//blob classification functions (included in 'blobclassify.cpp')
int classifyBlobs(cv::Mat fgmask, BlobList *pBlobList);
IplImage *paintBlobClasses(IplImage* frame, BlobList *pBlobList);

//stationary blob detection functions (included in 'blobstationary.cpp')
int detectStationaryForeground(cv::Mat fgmask, cv::Mat &fgmask_counter, cv::Mat &sfgmask);
int classifyStationaryBlobs(
        cv::Mat &matFrame,
        tForegroundSegmentationVATS describtor,
        IplImage *outFrame,
        cv::Mat &fgMask,
        cv::Mat &fgmask_counter);
int detectRevealedForeground(
        cv::Mat &matFrame,
        tForegroundSegmentationVATS describtor,
        cv::Mat &sfgMask,
        BlobList * blobList);
#endif
