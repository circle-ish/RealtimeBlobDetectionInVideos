#include "blobfuns.h"

#define FPS 25 //framerate of the input video
#define MIN_SECS 10.0 //minimum number of seconds to consider a foreground pixel as stationary

#define C_COST 1 //increment cost for stationary detection
#define D_COST 5 //penalization cost for stationary detection

/**
 *	Function to detect stationary foreground based on accumulator techniques. All the input arguments must be 
 *  initialized prior using this function.
 *
 * \param frame Input image
 * \param fgmask Foreground/Background segmentation mask (1-channel, uint8)  
 * \param fgmask_counter counter for stationary pixels (1-channel, float) 
  obtained in the analysis of the previous frame (to be updated in this function)
 * \param sfgmask Foregroung segmentation mask (1-channel, uint8)  with the stationary pixels
 *
 * \return Operation code (negative if not succesfull operation) 
 */
int detectStationaryForeground(cv::Mat fgmask, cv::Mat & fgmask_counter, cv::Mat &sfgmask) {
	//check input validity and return -1 if any is not valid
    if (!fgmask.data || !fgmask_counter.data) {
        std::cout << "Some of the input parameters of detectStationaryForeground are not initialized." << std::endl;
        return -1;
    }
	
    //num frames to static
    int numframes2static = (int)(FPS * MIN_SECS);

    //operate with fgmask to update fgmask_counter
    double con = 255.0 / (double)numframes2static;

    //counting
    fgmask.convertTo(fgmask, CV_32F, 1.0/255.0);
    fgmask_counter = fgmask_counter + (con * fgmask); //increase foreground; C_COST already implied
    cv::Mat bgmask = cv::Scalar::all(1) - fgmask;
    fgmask_counter = fgmask_counter - ((double)D_COST * con * bgmask); //decrease background
    cv::threshold(fgmask_counter, fgmask_counter, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(fgmask_counter, fgmask_counter, 0, 0, cv::THRESH_TOZERO);

//double Min, Max;
//cv::Mat tmp;
//cv::minMaxIdx(fgmask_counter, &Min, &Max);
//fgmask_counter.convertTo(tmp, CV_8U, 255.0/(Max-Min),-255.0*Min/(Max-Min));
//cv::imshow("counter", tmp);

//operate with fgmask_counter to update sfgmask
    cv::threshold(fgmask_counter, sfgmask, 200, 255, cv::THRESH_BINARY); //confidence threshold = 200
	
	return 1;
}

int detectRevealedForeground(
        cv::Mat &matFrame,
        tForegroundSegmentationVATS describtor,
        cv::Mat &sfgMask,
        BlobList * blobList) {

    BasicBlob *b;
    cv::Mat sfgConv;

    for(int i = 0; i < blobList->getBlobNum(); i++) {
        b = blobList->getBlob(i);
        cv::Rect rect(b->getX(), b->getY(), b->getWidth(), b->getHeight());
        cv::Mat tmp[3];
        Contours contours[3];

        //edge detection
        cv::cvtColor(matFrame(rect), tmp[0], CV_BGR2GRAY);
        tmp[0].convertTo(tmp[0], CV_8UC1);

        double Min, Max;
        cv::Mat means[3];
        std::vector<cv::Mat> channels;
        cv::split(describtor.mean(rect), means);
        for (int i = 0; i < 3; i++) {
            cv::minMaxIdx(means[i], &Min, &Max);
            means[i].convertTo(means[i], CV_8U, 255.0/(Max-Min),-255.0*Min/(Max-Min));
            channels.push_back(means[i]);
        }
        cv::merge(channels, tmp[2]);
        cv::cvtColor(tmp[2], tmp[2], CV_BGR2GRAY);
        tmp[1] = sfgMask(rect);

        cv::Canny(tmp[0], tmp[0], 40, 120, 3);
        cv::Canny(tmp[1], tmp[1], 40, 120, 3);
        cv::Canny(tmp[2], tmp[2], 40, 120, 3);

        cv::findContours(tmp[0], contours[0], CV_RETR_LIST, CV_CHAIN_APPROX_NONE); //current img
        cv::findContours(tmp[1], contours[1], CV_RETR_LIST, CV_CHAIN_APPROX_NONE); //sfgMask
        cv::findContours(tmp[2], contours[2], CV_RETR_LIST, CV_CHAIN_APPROX_NONE); //bg

        double notSupported = 1.0;
        double revealedBg = 0, abandonedFg = 0, max = 0;

        for (int i = 0; i < contours[0].size(); i++) {
            for (int j = 0; j < contours[1].size(); j++) {
//                if (contours[0][i].checkVector(2) >= 0
//                        && contours[2][j].checkVector(2) >= 0
//                        && (contours[0][i].depth() == CV_32F || contours[0][i].depth() == CV_32S)
//                        && contours[0][i].depth() == contours[2][j].depth() ) {

//                    continue;
//                }

                revealedBg = cv::matchShapes(contours[1][i], contours[2][j], CV_CONTOURS_MATCH_I1, notSupported);
                if (revealedBg > max) max = revealedBg;
            }
            revealedBg += max;
        }

        for (int i = 0; i < contours[0].size(); i++) {
            for (int j = 0; j < contours[1].size(); j++) {
//                if (contours[1][i].checkVector(2) >= 0
//                        && contours[0][j].checkVector(2) >= 0
//                        && (contours[1][i].depth() == CV_32F || contours[1][i].depth() == CV_32S)
//                        && contours[1][i].depth() == contours[0][j].depth() ) {

//                    continue;
//                }

                abandonedFg = cv::matchShapes(contours[1][i], contours[0][j], CV_CONTOURS_MATCH_I1, notSupported);
                if (abandonedFg > max) max = abandonedFg;

            }
            abandonedFg += max;
        }

        if (revealedBg > abandonedFg) { //it's revealed background
            if (!sfgConv.data) {
                cv::threshold(sfgMask, sfgConv, 100, 1, THRESH_BINARY);
                channels.clear();
                channels.push_back(sfgConv);
                channels.push_back(sfgConv);
                channels.push_back(sfgConv);
                cv::merge(channels, sfgConv);
            }

            cv::Mat a = matFrame(rect) * sfgConv(rect);
            a.convertTo(a, CV_32F);
            describtor.mean(rect) = a;
            describtor.variance(rect) = cv::Scalar::all(describtor.initialVariance);

            blobList->delBlob(i);
            --i;
        }
    }

    return 1;
}

int classifyStationaryBlobs(
        cv::Mat &matFrame,
        tForegroundSegmentationVATS describtor,
        IplImage *outFrame,
        cv::Mat &fgMask,
        cv::Mat &fgmask_counter) {

    if (!fgMask.data || !fgmask_counter.data) {
        std::cout << "Some of the input parameters of classifyStationaryBlobs are not initialized." << std::endl;
        return 0;
    }

    cv::Mat sfgMask;
    detectStationaryForeground(fgMask, fgmask_counter, sfgMask);
    sfgMask.convertTo(sfgMask, CV_8UC1);
    BlobList * blobList = new BlobList();
    int ret = classifyBlobs(sfgMask.clone(), blobList);
    cv::imshow("confi", sfgMask);

    detectRevealedForeground(matFrame, describtor, sfgMask, blobList);

    for(int i = 0; i < blobList->getBlobNum(); i++) {
        if (blobList->getBlob(i)->getlabel() == PERSON) {
            blobList->delBlob(i);
            --i;
        }
    }

    if (blobList->getBlobNum() != 0) {
        paintBlobClasses(outFrame, blobList);
    }
    return 1;
}
