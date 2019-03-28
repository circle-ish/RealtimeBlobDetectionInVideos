//file description
/**
 * \file main.cpp 
 * \author Juan Carlos San Miguel
 * \date 09/03/2013
 * \brief Scheme for lab sessions of VATS (I2TIC Master - EPS (UAM)
 * \version 1.0\n 
 *
*/

//standard and OpenCV functions
#include <opencv2/opencv.hpp>
#include "fgseg.h"

//include for blob-related functions
#include "blobfuns.h" 

#define INPUT_VIDEO	"../PETS06_S1-T1-C_3_abandoned_object_cif_mpeg.mpg"

int main(int argc, char *argv[])
{	
    //image initialization (counters, auxiliar images,...)
    //reading arguments; see documention
    tForegroundSegmentationVATS descriptor;

    char* inputvideo = (char*)INPUT_VIDEO;

	//programm arguments (only for the fg segmentation)
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-p") == 0) { //video path
            inputvideo = argv[i+1];
        } else if (strcmp(argv[i], "-a") == 0) { //learning rate for model update
            descriptor.learnRate = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-v") == 0) { //initial variance for model update
            descriptor.initialVariance = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-s") == 0) { //use shadows: 1=yes
            descriptor.shadowDetection = argv[i+1][0];
        } else {
            std::cout << "Unknown argument" << std::endl;
            return -1;
        }
    }

	// Required variables for the program
    IplImage *frame=NULL; //, *bg=NULL; //images for background subtraction
    IplImage *outFrame=NULL; //output images for blob extraction and blob labels
    int ret;
	
    double timeStart=0,timeEnd=0,timeTotal=0;
    int i = 0;
	
    //read video file & first frame
    Mat matFrame;
    cv::VideoCapture cap;
    cv::VideoWriter videowriter;

    cap.open(inputvideo);
    if (!cap.isOpened()) {
        std::cout << "Could not open video file " << inputvideo << std::endl;
        return -1;
    }
    cap >> matFrame;

	//module initialization (background subtraction, bloblist,...)
	BlobList *blobList = new BlobList();
		
    //create output windows
    cvNamedWindow("Blob Video", CV_WINDOW_AUTOSIZE);
    namedWindow("Foreground mask");

    //initializing simple bgk model (for shadow suppression) with first image
    descriptor.bg = matFrame;

    //create output writer
    videowriter.open("result.mpg", CV_FOURCC('P','I','M','1'), 25, matFrame.size(), 0);
    if (!videowriter.isOpened()) {
        std::cout << "Could not open videowriter" << std::endl;
        return -1;
    }

    cv::Mat fgcounter = cv::Mat::zeros(matFrame.size(), CV_32FC1);
    cv::Mat mask;

    int keyboard = '0';
    do {
        i++;
        timeStart =((double)cvGetTickCount()/(cvGetTickFrequency()*1000.) );

        //mask containing the detected foreground objects; cleaning mask
        mask = Mat::zeros(matFrame.size(), CV_8UC1);

		//background subtraction (final foreground mask must be placed in 'fg' variable)
        VATS_Foreground_Segmentation(descriptor, matFrame, descriptor.shadowDetection, mask);

        //no need to classify as long as nothing is detected
        imshow("Foreground mask", mask);
//        cv::waitKey(0);
        if (cv::countNonZero(mask) == 0) {
            continue;
        }

        //blob classification
        //only paint when blobs were found
        frame = new IplImage(matFrame);
        if (((ret = classifyBlobs(mask.clone(), blobList)) != 1)
            || ((outFrame = paintBlobClasses(frame, blobList)) == NULL)) {

            cvShowImage("Blob Video", frame);
            continue;
        }

        //stationary blob detection
        classifyStationaryBlobs(matFrame, descriptor, outFrame, mask, fgcounter);

        //show results visually
        if (outFrame == NULL) {
            outFrame = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 3);
            cvCopy(frame, outFrame);
        }
        cvShowImage("Blob Video", outFrame);
        keyboard = waitKey(10);

        if (keyboard == 'f') {
            std::cout << "current frame: " << i << std::endl;
        }

        //get time
        timeEnd = ((double)cvGetTickCount()/(cvGetTickFrequency()*1000.) );
        timeTotal = timeTotal + timeEnd - timeStart;

        //write frame result to video
//        videowriter << cv::cvarrToMat(outFrame);
		
        //release memory of temporal images
        if (outFrame != NULL) cvReleaseImage( &outFrame );

    } while (cap.read(matFrame) && ((char)keyboard != 'q') && ((char)keyboard != 27));

	//destroy all resources
    delete blobList;
    if (frame != NULL) cvReleaseImage( &frame );

	return 1;
}
