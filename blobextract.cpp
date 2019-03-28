#include "blobfuns.h"

/**
 *	Blob extraction from 1-channel image (binary). The extraction is performed based
 *	on the analysis of the connected components. All the input arguments must be 
 *  initialized when using this function.
 *
 * \param fgmask Foreground/Background segmentation mask (1-channel) 
 * 			will be overwritten by blobMask
 * \param pBlobList List to store the blobs found 
 * \param return value of detected blobs
 *
 * \return Operation code (negative if not succesfull operation) 
 */
int extractBlobs(cv::Mat &fgmask, BlobList *pBlobList, Contours &contours) {

	//check input conditions and return -1 if any is not satisfied
    if (!fgmask.data || pBlobList == NULL) {
        std::cout << "Variables for extractBlobs() not properly initialized" << std::endl;
        return -1;
    }

    cv::Mat blobMask = cv::Mat::zeros(fgmask.rows + 2, fgmask.cols + 2, CV_8UC1);
	//neighbor connectivity for blobs
    int connectivity = 8;
    bool blobFound = false;

    //dilation & erosion for better extraction results
	//notice: un-square kernels and multiple iterations
    cv::Mat kernel;
    int dilationSize = 1;
    kernel = getStructuringElement(
                cv::MORPH_ELLIPSE, //MORPH_RECT MORPH_ELLIPSE MORPH_CROSS
                cv::Size(2 * dilationSize + 1, 2 * dilationSize + 3),
                cv::Point(dilationSize, dilationSize));

    dilate(fgmask, fgmask, kernel);
//    cv::imshow("dil1", fgmask);

    int erosionSize = 2;
    kernel = getStructuringElement(
               cv::MORPH_CROSS,
               cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1),
               cv::Point(erosionSize, erosionSize));

    erode(fgmask, fgmask, kernel, cv::Point(-1, -1), 2);
//    cv::imshow("eroded", fgmask);

    dilationSize = 2;
    kernel = getStructuringElement(
                cv::MORPH_CROSS, //MORPH_RECT MORPH_ELLIPSE
                cv::Size(2 * dilationSize + 1, 2 * dilationSize + 3),
                cv::Point(dilationSize, dilationSize));

    dilate(fgmask, fgmask, kernel, cv::Point(-1, -1), 2);
//    cv::imshow("dil2", fgmask);

	//check pixel value; mark blobs in blobMask; erase pixel in input mask
    uchar *pMask;
    for(int r = 0; r < fgmask.rows; ++r) {

        //running pointer
        pMask = fgmask.ptr<uchar>(r);
        for(int c = 0; c < fgmask.cols; ++c) {
            if (pMask[c] == 255) {

                //extract connected component (blob)
                cv::floodFill(
                            fgmask,
                            blobMask,
                            cv::Point2i(c,r),   //seeding point
                            0,                  //new value for input image
                            0,                  //rect
                            1,                  //lowDiff //doesn't matter for binary image
                            1,                  //highDiff
                            connectivity | (255 << 8));

                blobFound = true;
            }
        }
    }

    if (blobFound) {
        cv::findContours(
                    blobMask, //only looking at area where blobs where found
                    contours,               //vector of points for every blob
                    CV_RETR_EXTERNAL,       //doesn't retrieve blobs in other blobs
                    CV_CHAIN_APPROX_NONE);//takes the whole contour points

        //drawing blobMask is important to countNonZero() in extractFeature()
        cv::Scalar color(255, 255, 255);
        cv::drawContours(blobMask, contours, -1, color, -1);
//    cv::imshow("after contour", blobMask.clone());
//    cv::waitKey(0);

        fgmask = blobMask;
    } else {
        return -1;
    }

    return 1;
}


/**
 *	Draw blobs with different rectangles on the image 'frame'. All the input arguments must be 
 *  initialized when using this function.
 *
 * \param frame Input image 
 * \param pBlobList List to store the blobs found 
 *
 * \return Image containing the draw blobs. If no blobs have to be painted 
 *  or arguments are wrong, the function returns NULL. The memory of this image 
 * is created inside this function so it has to be released after its use by the 
 * function calling 'paintBlobImage'.
 */
IplImage *paintBlobImage(IplImage* frame, BlobList *pBlobList) {

	//check input conditions and return NULL if any is not satisfied
    if (frame == NULL || pBlobList == NULL || pBlobList->getBlobNum() == 0) {
        std::cout << "Frame or BlobList sent into paintBlobImage are not initialized" << std::endl;
		return NULL;
	}

    //required variables to paint
    IplImage *blobImage = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 3);
	cvCopy(frame, blobImage);
	BasicBlob *blobToPaint;
	cv::Point left_upper_corner, right_bottom_corner;

	//paint each blob of the list
	for(int i = 0; i < pBlobList->getBlobNum(); i++) 	
	{
		//Get blob in the list
		blobToPaint = pBlobList->getBlob(i);

		//Get the corners coordinates
		left_upper_corner.x = blobToPaint->getX();
		left_upper_corner.y = blobToPaint->getY();
		right_bottom_corner.x = blobToPaint->getX() + blobToPaint->getWidth();
		right_bottom_corner.y = blobToPaint->getY() + blobToPaint->getHeight(); //TODO

		//Paint rectangle 
        cvRectangle(
             blobImage,
             left_upper_corner,  
             right_bottom_corner,  
             cvScalar(0, 0, 255, 0),  //Color
             1,  //Thickness
             8,  //Line type
             0 );  //Shift
	}

	//destroy all resources (if required)
	//...
	
	//return the image to show
	return blobImage;
}
