#include "auxfuns.h"

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "fgseg.h"

//dimension of the feature vector for classifing
#define DIM 3
//number of classes
#define NUMBER_OF_CLASSES 2
#define TRAINING_POINTS 4
#define ATTRIBUTE_NUMBER 4
#define BLOB_MIN_WIDTH 5
#define BLOB_MIN_HEIGHT 5
//if the distance between the borders of blobs
// is smaller they are joined 
#define BLOB_THRES 10

/*
 * gaussian values of features
 *
  humans:
asp mean: 0.431592
cir mean: 0.00056607
dev mean: 0.0101646
asp var: 0.215107
cir var: 0.0759732
dev var: 0.0446574

cars:
asp mean: 2.39313
cir mean: 0.000680065
dev mean: 0
asp var: 0.211897
cir var: 0.0180711
dev var: 0.000190222
  */

//class of blob is decided over the multidimensional distance to the center of the class
CLASS classify(std::vector<double> * feat) {
    //comparing with persons and cars
	//values from above
    double meansArray[NUMBER_OF_CLASSES][DIM] = {{
                                                     0.431592,
                                                     0.000566071,
                                                     0.0101646}, //human mean
                                                    {2.39313,
                                                     0.000680065,
                                                     0}}; //car mean
    double avgDevArray[NUMBER_OF_CLASSES][DIM] = {{
                                                      0.215107,
                                                      0.0759732,
                                                      0.0446574}, //human var
                                                     {0.211897,
                                                      0.0180711,
                                                      0.000190222}};

    cv::Mat means = cv::Mat(DIM, NUMBER_OF_CLASSES, CV_64F, meansArray); //cols: classes(human, cars); rows: feats
    cv::Mat avgDev = cv::Mat(DIM, NUMBER_OF_CLASSES, CV_64F, avgDevArray);

	//final class decision values
    double distances[NUMBER_OF_CLASSES]; //humans, cars

	//features of current blob
    double params[DIM];
    for (int i = 0; i < DIM; i++) {
        params[i] = feat->at(i);      //AR, circularity, sym
    }

    cv::Mat param_vector = cv::Mat(DIM, 1, CV_64F, params);

    // The closest to zero the better the fit to one Gaussian
    for (int i = 0; i < NUMBER_OF_CLASSES; i++) {
        cv::Mat aux;
        cv::absdiff(param_vector, means.col(i), aux);
        cv::Mat relative_distance_to_mean = aux / avgDev.col(i);
        cv::pow(relative_distance_to_mean, 2.0, aux);
        cv::Scalar tmp = cv::sum(aux);
        distances[i] = tmp[0];
        distances[i] = sqrt(distances[i])  / feat->at(feat->size() - 1); //dividing by area
    }

    //td::cout << "AR = " << params[0] << "; Cir = " << params[1] << "; sy = " << params[2]
    //          << "hu = " << distances[0] << "; car = " << distances[1] << std::endl;

    //getting min distance
    int index = 0;
    for(int i = 1; i < NUMBER_OF_CLASSES; i++) {
        if(distances[i] < distances[index]) {
            index = i;
        }
    }

    int min = index;
    switch(index) {
        case 0: index = PERSON;
                break;
        case 1: index = CAR;
                break;
        default: index = OBJECT;
                break;
    }

	//experience shows:
    if (index != OBJECT && distances[min] > 0.1) index = OBJECT;

    return (CLASS)index;
}

std::vector<double> SVM_predict (BlobList *pBlobList, std::map<int, std::vector<double> > &blobs) {

//    std::vector<double> response;
//    for (std::map<int,std::vector<double> >::iterator it=blobs.begin(); it!=blobs.end(); ++it) {

////        cv::Mat sampleMat = (cv::Mat_<float>(1,2) << i,j);
//        cv::Mat sample = (cv::Mat_<float>(4,1) << it->second[0], it->second[1], it->second[3]);
//        response.push_back(SVM.predict(sample));
//    }
//    return response;
}

/*
 * - add its features to every blob in the list
 * - blobMask needed for area
 *  
 *  extracts three features:
 *  	aspect ratio
 *  	circularity
 *  	symitricity
 *
 *  symitricity was thought of as an indicator of the deformation of an object (shape shifting human vs car)
 *
 * overlapping or near blobs are joint; means the new blobs is assigned with the outer border of both blobs
 *
 * calls the classification function
 */
int extractFeature(BlobList * pBlobList, cv::Mat &blobMask, Contours contours) {

    int lastId = -1;

	//for every found blob
    for(int i = 0; i < (int)contours.size(); i++ ) {
        auto minMaxX = std::minmax_element(contours[i].begin(), contours[i].end(), lessX);
        auto minMaxY = std::minmax_element(contours[i].begin(), contours[i].end(), lessY);

        float newX = (*(minMaxX.first)).x - 1; //blobMask was created with 1 Pixel border
        float newY = (*(minMaxY.first)).y - 1;
        float newW = (*(minMaxX.second)).x - 1 - newX;
        float newH = (*(minMaxY.second)).y - 1 - newY;

        //check size of blob using BLOB_MIN_WIDTH & HEIGHT (valid = true)
        if (BLOB_MIN_WIDTH <= newW && BLOB_MIN_HEIGHT <= newH) {

            BasicBlob * blob;

            //if blob is inside of other one or has a mutual area join them
            for(int j = 0; j < pBlobList->getBlobNum(); j++) {
                blob = pBlobList->getBlob(j);
                float x = blob->getX();
                float y = blob->getY();
                float w = blob->getWidth();
                float h = blob->getHeight();

                if (newX + newW + BLOB_THRES < x
                        || x + w + BLOB_THRES < newX
                        || newY + newH + BLOB_THRES < y
                        || y + h + BLOB_THRES < newY) {

                    continue; // no overlap
                }

                //joining blobs
                newW = std::max(newX + newW, x + w);
                newH = std::max(newY + newH, y + h);
                newX = std::min(newX, x);
                newY = std::min(newY, y);
                newW -= newX;
                newH -= newY;

                //delete old blob
                pBlobList->delBlob(j);

                //recheck blob overlap -> start the loop anew
                j = 0;
            }

            //add new blob
            //include blob in 'pBlobList' if it is validBasicBlob *blob = new BasicBlob();
            blob = new BasicBlob();
            blob->setX(newX);
            blob->setY(newY);
            blob->setHeight(newH);
            blob->setWidth(newW);

            pBlobList->addBlob(blob);
        }
    }

	//over every found blob
    for (int i = 0; i < pBlobList->getBlobNum(); i++) {
        BasicBlob * blob = pBlobList->getBlob(i);

		//finally assign ID
        if (lastId == -1) {
            lastId = 1;
        }
        blob->setID(++lastId);

        //extracting features of blobs for classification
        // Aspect Ratio calculation
        double AR;
        if (blob->getHeight() != 0.0)
            AR = blob->getWidth() / blob->getHeight();

        // Circularity calculation
        double area = cv::countNonZero(blobMask(cv::Rect(
                                                 blob->getX(),
                                                 blob->getY(),
                                                 blob->getWidth(),
                                                 blob->getHeight()
                                                 )));
        double circularity = contours[0].size()/ (2 * sqrt(M_PI) * area);

        // Symmetricity calculation
        double sumLeft = 0.0, sumDevLeft = 0.0;
        double sumRight = 0.0, sumDevRight = 0.0;
        int j;

        // Get middle point in x axis
        double middlePoint = blob->getWidth() / 2.0;

        // Following for means: for every point in a contour
        for (j=0; j<(int)contours[i].size(); j++) {
            if (contours[i][j].x <= middlePoint) {
                sumLeft += contours[i][j].x;
            }
            else {
                sumRight += contours[i][j].x;
            }
        }
        double meanLeft = sumLeft / (double)j;
        double meanRight = sumRight / (double)j;

        for (j=0; j<(int)contours[i].size(); j++) {
            if (contours[i][j].x <= middlePoint) {
                sumDevLeft += contours[i][j].x - meanLeft;
            }
            else {
                sumDevRight += contours[i][j].x - meanRight;
            }
        }
        double stdDevLeft = (sumDevLeft / (double)j) / blob->getWidth();
        double stdDevRight = (sumDevRight / (double)j) / blob->getWidth();
        double biggerStdDev = std::max(stdDevLeft, stdDevRight);

        std::vector<double> * v = new std::vector<double>;
        v->push_back(AR);
        v->push_back(circularity);
        v->push_back(biggerStdDev);
        v->push_back(area);

        //classify blobs
        CLASS label = classify(v);
        blob->setlabel(label);

        delete v;
    }

    return 1;
}
