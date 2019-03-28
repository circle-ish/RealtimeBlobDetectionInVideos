//class description
/**
 * \class BasicBlob
 * \brief Class to describe a foreground blob
 * 
 */

#ifndef BASICBLOB_H_INCLUDE
#define BASICBLOB_H_INCLUDE

// Additional includes
//#include <opencv2/opencv.hpp>
#include <opencv2/legacy/blobtrack.hpp> //for CvBlob structure
#include <vector>

/// Maximun number of char in the blob's format
const int MAX_FORMAT = 1024;

/// Type of labels for blobs
typedef enum {	
	UNKNOWN=0, 
	PERSON=1, 
	GROUP=2, 
	CAR=3, 
	OBJECT=4
} CLASS;

// Class definition
class BasicBlob {
private:
	// OpenCV blob
	CLASS label;
	int FrameStart;
	CvBlob blob;	
	char format[MAX_FORMAT];	
    IplImage *ObjectMask;

public:
	// Default Constructor
	BasicBlob();

	// Extended Constructor (for copy)
	BasicBlob(CvBlob* blob);
	// Extended Constructor (for copy)
	BasicBlob(CvBlob* blob, char* format = "ID: Posicion:(x,y), Dimension(w,h)");
	
	// Default Destructor
	virtual ~BasicBlob();

	//Method to get the label of the blob
	CLASS	getlabel();
	//Method to get the x coordinate of the top-left point of the blob bounding box
	float	getX();
	//Method to get the y coordinate of the top-left point of the blob bounding box
	float	getY();
	//Method to get the width of the blob bounding box
	float	getWidth();
	//Method to get the heigth of the blob bounding box
	float	getHeight();
	//Method to get the id  of the blob
	int		getID();
	//Method to get OpenCV blob structure
	CvBlob* getBlob();
	//Method to get a string describing the blob
	char*	getFormat();
	//Method to get frameStart of the blob
    int getFrameStart();

	//Method to set the label of the blob
	void setlabel(CLASS newlabel);
	//Method to set the x coordinate of the top-left point of the blob bounding box
	void setX(float x);
	//Method to set the y coordinate of the top-left point of the blob bounding box
	void setY(float y);
	//Method to set the width of the blob bounding box
	void setWidth(float w);
	//Method to set the height of the blob bounding box
	void setHeight(float h);
	//Method to set the id  of the blob
	void setID(int ID);
	//Method to set frameStart of the blob
	void setFrameStart(int frameStart);
	//Method to set OpenCV blob structure
	void setBlob(CvBlob* blob);
	//Method to set a string describing the blob
	void setFormat(char* format);

	//Prints some features of all the blobs in a file
	virtual void write(char *filename);

	//Prints some features of blob in stdout
	void printInfo();
	//Prints some features of blob in a file handler
	void writeInfo(FILE* fp);

	//mev: Method to fill ObjectMask field with an already compute Mask
	void setObjectMask(IplImage *Mask);
	//mev: Method to get ObjectMask field from a Blob
	IplImage *getObjectMask();
};

#endif
