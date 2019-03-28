//class description
/**
 * \class BlobList
 * \brief Class to describe a list of blobs
 * 
 */

#ifndef BLOBLIST_H_INCLUDE
#define BLOBLIST_H_INCLUDE

#include <vector> //for having lists of elements
#include "BasicBlob.h"//blob structure

class BlobList {
private:
	 std::vector<BasicBlob *> *pBlob;
	
public:
	// Class Constructor
	BlobList();

	// Extended Class constructor (with copy)
	BlobList(std::vector<BasicBlob *> *pBlob);

	//Class destructor
	virtual ~BlobList();

	// Method to add a blob to the list
	int addBlob(BasicBlob * blob);

	// Method to clear all blobs in the list
	int clear();

	// Method to delete a blob by using its position
	int delBlob(int n);

	// Method to delete a blob by using its ID
	int delBlobByID(int ID);

	// Method to get the number of blobs stored in the list
	int getBlobNum();

	// Method to get a blob by using its position
	BasicBlob * getBlob(int n);

	// Method to get a blob by using its ID
	BasicBlob * getBlobByID(int ID);

	// Method to set a blob in a certain position
	int setBlob(int n, BasicBlob * Blob);

	// Method to set a blob in a certain position (?)
	int setBlobByID(int ID, BasicBlob * Blob);

	//Prints some features of all the blobs in a file
	int write(char *filename);
};

#endif
