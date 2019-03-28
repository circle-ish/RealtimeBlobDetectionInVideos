#ifndef BLOBLIST_CPP
#define BLOBLIST_CPP

//Additional includes
#include "BlobList.h"

/**
 *	Standard class Constructor. 
 *
 */
BlobList::BlobList() {
	this->pBlob = new std::vector<BasicBlob *>;
}

/**
 *	Class Constructor with an initial blob array
 *
 * \param pBlob Initial vector of blobs
 *
 */

BlobList::BlobList(std::vector<BasicBlob *> *pblob) {
	this->pBlob = pblob;
}

/**
 *	Standard class destructor
 *
 */

BlobList::~BlobList() {
	if(this->getBlobNum() > 0)
		this->clear();
	if (this->pBlob != NULL) {
		delete this->pBlob;
		this->pBlob = NULL;
	}
}

/**
 *	Method to add a blob in the blob's list
 *
 * \param blob Blob to add. The type of the blob depends on the blob list inizialization.
 *
 * \return Returns operation code (if error returns a value < 0)
 */

int BlobList::addBlob(BasicBlob * blob) {
	this->pBlob->push_back(blob);
	return 0;
}

/**
 *	Method to clear all the blobs in the list
 *
 * \return Returns operation code (if error returns a value < 0)
 */

int BlobList::clear() {
	for(int i = 0; i < this->getBlobNum(); i++) {
		delete this->getBlob(i);
	}
	this->pBlob->clear();
	return 0;
}

/**
 *	Method to delete a blob from the blob's list
 *
 * \param n Position of the blob to delete
 *
 * \return Returns operation code (if error returns a value < 0)
 */

int BlobList::delBlob(int n) {
	delete this->getBlob(n);
	std::vector<BasicBlob *>::iterator iter;
	iter = this->pBlob->begin();
	iter = iter+n;
	this->pBlob->erase(iter);
	return 0;
}

/**
 *	Method to delete a blob from the blob's list
 *
 * \param ID Blob identification number for blob to delete
 *
 * \return Returns operation code (if error returns a value < 0)
 */

int BlobList::delBlobByID(int ID) {
	std::vector<BasicBlob *>::iterator iter;
	int i = 0;
	iter = this->pBlob->begin();
	
	for(i = 0; i < (int)this->pBlob->size(); i++) {
		if (this->pBlob->at(i)->getID() == ID) {
			delete this->getBlob(i);
			iter = iter + i;
			this->pBlob->erase(iter);
			return 0;
		}
	}
	return -1;	
}

/**
 *	Method to get the numbers of blobs stored in the blob list
 *
 * \return Returns the numbers of blobs
 */

int BlobList::getBlobNum() {
	return (int)this->pBlob->size();
}

/**
 *	Method to get a blob from the blob's list.
 *
 * \param n Blob position  for blob to get
 *
 * \return Blob type stored in the list
 */

BasicBlob * BlobList::getBlob(int n) {
	return this->pBlob->at(n);	
}

/**
 *	Method to get a blob from the blob's list
 *
 * \param n Blob ID for blob to get
 *
 * \return Blob type stored in the list
 */

BasicBlob * BlobList::getBlobByID(int ID) {
	int i = 0;
	
	for(i = 0; i < (int)this->pBlob->size(); i++) {
		if (this->pBlob->at(i)->getID() == ID) {
			return this->pBlob->at(i);
		}
	}
	return NULL;	
}

/**
 *	Method to set a blob in the blob's list.
 *
 * \param n Blob position  for blob to set
 * \param blob Blob (same type as the blob type stored in the list)
 *
 * \return Operation code
 */

int BlobList::setBlob(int n, BasicBlob * Blob) {
	if (n > (int)this->pBlob->size())
		return -1;
	std::vector<BasicBlob *>::iterator iter = this->pBlob->begin();
	iter = iter+n;
	delete this->pBlob->at(n);
	this->pBlob->insert(iter,Blob);
	return 0;
}

/**
 *	Method to set a blob in the blob's list.
 *
 * \param n Blob ID  for blob to set
 * \param blob Blob (same type as the blob type stored in the list)
 *
 * \return Operation code
 */

int BlobList::setBlobByID(int ID, BasicBlob * Blob) {
	int i = 0;
	std::vector<BasicBlob *>::iterator iter = this->pBlob->begin();
	
	for(i = 0; i < (int)this->pBlob->size(); i++) {
		if (this->pBlob->at(i)->getID() == ID) {
			delete this->pBlob->at(i);
			this->pBlob->insert(iter,Blob);
			return 0;
		}
		iter++;
	}
	return -1;
}
	

/**
 *	Prints some features of all the blobs in a file
 *
 * \param filename output filename to print blob extraction results
 *
 * \return Operation code
 */

int BlobList::write(char *filename) {
	return -1;
}

#endif