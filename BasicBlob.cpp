//Additional includes
#include "BasicBlob.h"

/**
 *	Standard class Constructor. 
 *
 */
BasicBlob::BasicBlob() {
	this->label = UNKNOWN;
	this->format[0] = '\0';	
	this->ObjectMask=NULL;
}

/**
 *	Class Constructor with initial data
 *
 *	\param blob OpenCV blob initial data 
 *
 */
BasicBlob::BasicBlob(CvBlob* blob) {
	this->setBlob(blob);
	this->format[0] = '\0';		
}

/**
 *	Class Constructor with initial data
 *
 *	\param blob OpenCV blob initial data
 *	\param format Initial description of the blob
 *
 */
BasicBlob::BasicBlob(CvBlob* blob, char* format) {
	this->setBlob(blob);
	this->setFormat(format);	
}

/**
 *	Standard class destructor. 
 *
 */
BasicBlob::~BasicBlob() {

	if (this->ObjectMask!=NULL)
		cvReleaseImage(&(this->ObjectMask));
}

/**
 *	Method to get the label of the blob
 *
 * \return Returns the blob label
 */
CLASS BasicBlob::getlabel()
{
	return this->label;
}

/**
 *	Method to get the x coordinate of the top-left point of the blob bounding box
 *
 * \return Returns the x value
 */
float BasicBlob::getX() {
	return this->blob.x;
}

/**
 *	Method to get the y coordinate of the top-left point of the blob bounding box
 *
 * \return Returns the y value
 */
float BasicBlob::getY() {
	return this->blob.y;
}

/**
 *	Method to get the width of the blob bounding box
 *
 * \return Returns the blob witdh
 */
float BasicBlob::getWidth() {
	return this->blob.w;
}

/**
 *	Method to get the height of the blob bounding box
 *
 * \return Returns the blob height
 */
float BasicBlob::getHeight() {
	return this->blob.h;
}

/**
 *	Method to get the ID of the blob
 *
 * \return Returns the blob ID
 */
int	BasicBlob::getID() {
	return this->blob.ID;
}

/**
 *	Method to get the frame start of the blob
 *
 * \return Returns the frame number when the blob started to appear
 */
int BasicBlob::getFrameStart()
{
	return this->FrameStart;
}

/**
 *	Method to get the OpenCV blob structure
 *  The structure is:
 *		typedef struct CvBlob { 
 *								int x; 
 *								int y;
 *								int width;
 *								int height;
 *								int ID;
 *							  }
 *
 * \return Returns the structure
 */
CvBlob* BasicBlob::getBlob() {
	/*CvBlob* output = new CvBlob();
	output->h = this->blob.h;
	output->w = this->blob.w;
	output->x = this->blob.x;
	output->y = this->blob.y;
	output->ID = this->blob.ID;
	return output;
	*/
	return &(this->blob);
}

/**
 *	Method to get information about the blob.
 *	For the string that contains the information, new memory is allocated. 
 *	After using the string, it should be released
 *
 * \return Returns the string containing the data
 */
char* BasicBlob::getFormat() {
	return this->format;
}


/**
 *	Method to set the blob label
 *
 * \param x The new blob label
 */
void BasicBlob::setlabel(CLASS newlabel)
{
	this->label = newlabel;
}

/**
 *	Method to set the x coordinate of the top-left point of the blob bounding box
 *
 * \param x The new top-left point x value
 */
void BasicBlob::setX(float x) {
	this->blob.x = x;
}

/**
 *	Method to set the y coordinate of the top-left point of the blob bounding box
 *
 * \param y The new top-left point y value
 */
void BasicBlob::setY(float y) {
	this->blob.y = y;
}

/**
 *	Method to set the width of the blob bounding box
 *
 * \param w The new blob witdh
 */
void BasicBlob::setWidth(float w) {
	this->blob.w = w;
}

/**
 *	Method to set the height of the blob bounding box
 *
 * \param h The new blob height
 */
void BasicBlob::setHeight(float h) {
	this->blob.h = h;
}

/**
 *	Method to set blob ID
 *
 * \param ID The new blob ID
 */
void BasicBlob::setID(int ID) {
	this->blob.ID = ID;
}

/**
 *	Method to set frame start
 *
 * \param frameStart The new blob frame start
 */
void BasicBlob::setFrameStart(int frameStart)
{
	this->FrameStart = frameStart;
}

/**
 *	Method to set the OpenCV blob structure information
 *  The structure is:
 *		typedef struct CvBlob { 
 *								int x; 
 *								int y;
 *								int width;
 *								int height;
 *								int ID;
 *							  }
 *
 * \param blob Pointer to the new structure
 */
void BasicBlob::setBlob(CvBlob* blob) {
	this->blob.x = blob->x;
	this->blob.y = blob->y;
	this->blob.w = blob->w;
	this->blob.h = blob->h;
	this->blob.ID = blob->ID;
}

/**
 *	Method to concatenate a description of the blob
 *
 * \param format String that contains additional info
 */
void BasicBlob::setFormat(char* format) {
	int i = 0;
	for(i = 0; ((i < MAX_FORMAT-1) & (format[i] != '\0')); i++)
		this->format[i] = format[i];
	this->format[i] = '\0';
	return;
}

/**
 *	Method to write data of the blob in a file
 *
 * \param filename Name of the file
 */
void BasicBlob::write(char *filename) {
	FILE *pf = fopen(filename,"a+");
	this->writeInfo(pf);
	fclose(pf);
}

/**
 *	Method to print some features of blob in stdout
 *
 */
void BasicBlob::printInfo() {
	fprintf(stdout,"ID %d: (%f,%f), (%f,%f)\n",this->blob.ID, this->blob.x, 
		    this->blob.y, this->blob.w, this->blob.h);
}

/**
 *	Method to print some features of blob in a file handler
 *
 * \param fp file handelr Name of the file
 */
void BasicBlob::writeInfo(FILE* fp) {
	if (fp != NULL) 
		fprintf(fp,"%d: (%f,%f), (%f,%f)",this->blob.ID, this->blob.x, this->blob.y, this->blob.w, this->blob.h);
	else 
		fprintf(stdout,"No se pudo realizar la escritura\n");
}

//mev: Method to fill ObjectMask field with an already computed Mask
void BasicBlob::setObjectMask(IplImage *Mask){
		
	ObjectMask=cvCreateImage(cvGetSize((IplImage *)Mask),IPL_DEPTH_8U, 1); 
	ObjectMask->origin=Mask->origin;
	cvCopy(Mask,ObjectMask);
}
//mev: Method to get ObjectMask field from a Blob
IplImage *BasicBlob::getObjectMask(){
	IplImage *Mask = cvCloneImage(this->ObjectMask);
	Mask->origin=ObjectMask->origin;
	return Mask;
	//cvCopy(this->ObjectMask,Mask);
}
