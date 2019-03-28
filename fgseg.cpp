#include "fgseg.h"


char VATS_Foreground_Segmentation_Start(tForegroundSegmentationVATS descriptor){
	return 255;
}

char VATS_Foreground_Segmentation_Stop(tForegroundSegmentationVATS descriptor) {
	return 255;
}

char VATS_Foreground_Segmentation(
	tForegroundSegmentationVATS &descriptor, 
	Mat input, 
	char shadows, 
	Mat &mask) {

    //INITIALIZING
    double initial_variance = descriptor.initialVariance;

    if (!descriptor.mean.data) {
        //initializing mean values to pixel values
        input.convertTo(descriptor.mean, CV_32FC3);

        //initizlising simple bgk model
        descriptor.bg = input.clone();
    }
    if (!descriptor.variance.data){
        descriptor.variance = Mat(input.size(), CV_32FC3, Scalar::all(initial_variance));
    }

    //channels have to be handled seperated in order to use the threshold function
    Mat input_rgb[3], mean_rgb[3], variance_rgb[3], mask_rgb[3], aux;
    input.convertTo(aux, descriptor.mean.type());
    split(aux,input_rgb);
    split(descriptor.mean,mean_rgb);
    split(descriptor.variance,variance_rgb);

    //a pixel is background when it's - roughly said - within I(x,y) - mean(x,y) -3*stdDeviation(x,y) < 6*stdDeviation(x,y)
    absdiff(input_rgb[0], mean_rgb[0], input_rgb[0]);
    absdiff(input_rgb[1], mean_rgb[1], input_rgb[1]);
    absdiff(input_rgb[2], mean_rgb[2], input_rgb[2]);
    threshold((input_rgb[0] - (variance_rgb[0]*3)), mask_rgb[0], 0, 1, THRESH_BINARY);
    threshold((input_rgb[1] - (variance_rgb[1]*3)), mask_rgb[1], 0, 1, THRESH_BINARY);
    threshold((input_rgb[2] - (variance_rgb[2]*3)), mask_rgb[2], 0, 1, THRESH_BINARY);

    mask = mask_rgb[0] | mask_rgb[1] | mask_rgb[2]; // Binary OR with the 3 channels

    //apply shadow suppression
    if (shadows == '1') {
        Mat shadowMask(mask.size(), mask.type(), cv::Scalar(0));
        shadowDetection(descriptor, input, shadowMask);
        if (shadowMask.data) {
            mask = mask - shadowMask;

            shadowMask.convertTo(shadowMask, CV_8UC1, 255);
//            imshow("Shadow mask", shadowMask);
        }
    }

    //UPDATING

	//learningrate for the update step of the gaussian model
    double alpha = descriptor.learnRate;
    Mat aux2, mask_3C;

    mask.convertTo(mask, CV_32F, 1.0/255.0);
    input.convertTo(input, CV_32F);

    if (descriptor.mean.data) { // Running average for the mean
		//applying update only to background pixel
        //using three channel mask is necessary for the input.mul step
        vector<Mat> channels;
        channels.push_back((1.0 - mask));
		channels.push_back(channels[0]);
		channels.push_back(channels[0]);
        merge(channels, mask_3C);
        aux = input.mul(mask_3C, alpha);

        //bg update, CV_8U
        aux.convertTo(aux, CV_8U);
        descriptor.bg = descriptor.bg*(1-alpha) + aux;

        //gaussian update of the mean according to lecture, CV_32F
        aux.convertTo(aux2, CV_32FC3);
        descriptor.mean = descriptor.mean*(1-alpha) + aux2;
	}

    if (descriptor.variance.data) {	// Running average for the variance
        vector<Mat> channels;
        channels.push_back((1.0 - mask));
		channels.push_back(channels[0]);
		channels.push_back(channels[0]);
        merge(channels, mask_3C);
        aux = input.mul(mask_3C);
        aux.convertTo(aux2, CV_32FC3);
	
		//update of standard deviation according to lecture
        cv::pow((aux2 - descriptor.mean), 2, aux2);
        cv::pow(descriptor.variance, 2, aux);
        cv::sqrt(aux2*alpha + (1-alpha)*aux, descriptor.variance);
	}

	//in order to display the mask it's values has to streched between 0..255
    double Min, Max;
    cv::minMaxIdx(mask, &Min, &Max);
    mask.convertTo(mask, CV_8U, 255.0/(Max-Min),-255.0*Min/(Max-Min));
	
	return 255;
}

//shadow detection by chromatic analysis according to 
//http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=948679
char shadowDetection(
	tForegroundSegmentationVATS &descriptor, 
	Mat imageBGR,
	Mat &shadowMask ) { //should be initialized to 0

		//transformation to HSV space needed
		Mat temp, temp2, imageHSV[3], bgHSV[3];
		cvtColor(descriptor.bg, temp, CV_RGB2HSV); 
        cvtColor(imageBGR, temp2, CV_RGB2HSV);
		split(temp, bgHSV); // Separate subs in three matrices (channels)
        split(temp2, imageHSV);

		//default thresholds according to original source
		double thresSat = descriptor.thresSat;
		double thresHue = descriptor.thresHue;
		double thresLow = descriptor.thresLow;
		double thresHigh = descriptor.thresHigh;
		Mat testRatio, testSat, testHue;

		//image to background comparison; see paper
		imageHSV[2].convertTo(temp, CV_32F);
		bgHSV[2].convertTo(temp2, CV_32F);
        testRatio = temp.mul(1.0/temp2);
		absdiff(imageHSV[1], bgHSV[1], testSat);
        absdiff(imageHSV[0], bgHSV[0], testHue);

		//pixelwise check with thresholds
        //general approach for efficient pixelwise approach from
		//OpenCV documentation page
		int channels = testRatio.channels();
		int nRows = testRatio.rows;
		int nCols = testRatio.cols * channels;

		if (testRatio.isContinuous() && testSat.isContinuous() && testHue.isContinuous()) {
			nCols *= nRows;
			nRows = 1;
		}

		float *pRatio;
		uchar *pSat, *pHue;
		for(int r = 0; r < nRows; ++r) {

			//running pointer
			pRatio = testRatio.ptr<float>(r);
			pSat = testSat.ptr<uchar>(r);
			pHue = testHue.ptr<uchar>(r);
			for(int c = 0; c < nCols; ++c) {

				//actual comparison and shadowMask creation
				if (thresLow <= pRatio[c] 
					&& pRatio[c] <= thresHigh
					&& pSat[c] <= thresSat
					&& (pHue[c] <= thresHue || (360 - pHue[c] <= thresHue))) {

						shadowMask.at<uchar>(r, c) = 1;
				}
			}
		}

		return 255;
}
