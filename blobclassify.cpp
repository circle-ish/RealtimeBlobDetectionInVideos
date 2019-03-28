#include "blobfuns.h"
#include "auxfuns.h"

/**
 *	Blob classificaion between the five available classes in 'BasicBlob.h' (see CLASS typedef). All the input arguments must be 
 *  initialized when using this function.
 *
 * \param frame InputOutput image
 * \param fgmask Input Foreground/Background segmentation mask (1-channel)
 * \param pBlobList InputOutput List to store the blobs found
 *
 * \return Operation code (negative if not succesfull operation) 
 */
int classifyBlobs(cv::Mat fgmask, BlobList *pBlobList) {

    //check input conditions and return -1 if any is not satisfied
    if (!fgmask.data || pBlobList==NULL) {
        std::cout << "Some of the input parameters of classifyBlobs are NULL" << std::endl;
        return -1;
    }

    //clear blob list (to fill with this function)
    pBlobList->clear();

    //required variables for connected component analysis
    std::vector<std::vector<cv::Point> > contours;
    int ret;

    //exctract blobs
    ret = extractBlobs(fgmask, pBlobList, contours);
    if (ret != -1) ret = extractFeature(pBlobList, fgmask, contours);

//    std::vector<double> result = SVM_predict(pBlobList);
//    for( std::vector<double>::const_iterator i = result.begin(); i != result.end(); ++i)
//        std::cout << *i << ' ';

    return ret;
}


/**
 *	Draw blobs (and its classes) with different color rectangles on the image 'frame'. All the input arguments must be 
 *  initialized when using this function.
 *
 * \param frame Input image 
 * \param pBlobList List to store the blobs found 
 *
 * \return Image containing the draw blobs. If no blobs have to be painted 
 *  or arguments are wrong, the function returns NULL. The memory of this image 
 * is created inside this function so it has to be released after its use by the 
 * function calling 'paintBlobClasses'.
 */
IplImage *paintBlobClasses(IplImage* frame, BlobList *pBlobList) {

    //check input conditions and return NULL if any is not satisfied
    if (frame == NULL || pBlobList == NULL || pBlobList->getBlobNum() == 0) {
        std::cout << "Frame or BlobList sent into paintBlobClasses are not initialized" << std::endl;
        return NULL;
    }

    //required variables to paint
    IplImage *blobImage = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 3);
    cvCopy(frame, blobImage);
    BasicBlob *blobToPaint;
    cv::Point left_upper_corner, right_bottom_corner, text_origin;
    int B, G, R;
    char *buf;
    CvFont font;

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
        text_origin.x = left_upper_corner.x + (blobToPaint->getWidth()/4);
        text_origin.y = left_upper_corner.y - 0.8;

        switch (blobToPaint->getlabel()) {
        case 0:
            buf = "UNKNOWN";
            B=0; G=0; R=0;		//Black
            break;
        case 1:
            buf = "PERSON";
            B=255; G=0; R=0;	//Blue
            break;
        case 2:
            buf = "GROUP";
            B=0; G=255; R=0;	//Green
            break;
        case 3:
            buf = "CAR";
            B=0; G=0; R=255;	//Red
            break;
        case 4:
            buf = "OBJECT";
            B=0; G=255; R=255;	//Yellow
            break;
        default:
            buf = "OTHER";
            B=255; G=255; R=255;	//White
            break;
        }

        cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 0.6, 0.6, 0);

        //Paint rectangle
        cvRectangle( blobImage,
             left_upper_corner,
             right_bottom_corner,
             cv::Scalar(B, G, R),	//Color
             1,						//Thickness
             8,						//Line type
             0 );					//Shift

        cvPutText( blobImage,
            buf,							//Text
            text_origin,					//Origin
            &font,							//Font
            cv::Scalar(B, G, R));			//Color

        }

    //destroy all resources (if required)
    //...

    //return the image to show
    return blobImage;
}
