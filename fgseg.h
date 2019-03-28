#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

#ifndef MYHEADER_H
#define MYHEADER_H

//serves as a global container class to store, change and access parameters
 class tForegroundSegmentationVATS {
 public:
	tForegroundSegmentationVATS() {
        learnRate = 0.000005;
        initialVariance = 10.0;
        shadowDetection = '1';
		thresSat = 0.5;
		thresHue = 1;
		thresLow = 0.4;
		thresHigh = 0.6;
    }

    //bgk model for shadowsuppression
	Mat bg;
	//matrix of means and variance for every pixel of the input image and for every channel
	Mat mean;
	Mat variance;
	//adaptionrate for bgk model updat
	double learnRate;
	//value for freshly initialized bgk model gaussians
	double initialVariance;
	//using shadow suppression; 1=yes
	char shadowDetection;
	//thresholds for shadow suppression function
	double thresSat;
	double thresHue; 
	double thresLow;
	double thresHigh;
		
} ;

//requested function; not necessary in a c++ implementation
char VATS_Foreground_Segmentation_Start(
	tForegroundSegmentationVATS descriptor);

//requested function; not necessary in a c++ implementation
char VATS_Foreground_Segmentation_Stop(
	tForegroundSegmentationVATS descriptor);

//main function; initializes and updates bgk model;
//extracts foreground mask
//
//gets desscriptor with all necessary parameters,
//the current image frame,
//the switch if shadow suppression should be used
//and an initialized return mask
char VATS_Foreground_Segmentation(
	tForegroundSegmentationVATS &descriptor, 
	Mat input,  
	char shadows, 
	Mat &mask);

//performs shadow detection
//
//gets desscriptor with all necessary parameters,
//the current image frame,
//and a to zeros initialized return mask
char shadowDetection(
	tForegroundSegmentationVATS &descriptor,  
	Mat imageBGR,
	Mat &shadowMask);

#endif // MYHEADER_H
