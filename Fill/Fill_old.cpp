
#ifndef __CONFIG_H
#define __CONFIG_H

#ifdef _DEBUG
#define CG_LIB_BIN_FILE(libname) libname "d" ".lib"
#else
#define CG_LIB_BIN_FILE(libname) libname ".lib"
#endif // _DEBUG

#include <cv.h>
#include <highgui.h>
#include <stdio.h>
using namespace cv;

#pragma comment(lib,CG_LIB_BIN_FILE("opencv_calib3d231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_contrib231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_core231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_features2d231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_flann231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_gpu231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_haartraining_engine"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_highgui231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_imgproc231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_legacy231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_ml231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_objdetect231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_ts231"))
#pragma comment(lib,CG_LIB_BIN_FILE("opencv_video231"))

#pragma comment(lib,CG_LIB_BIN_FILE("libjasper"))
#pragma comment(lib,CG_LIB_BIN_FILE("libjpeg"))
#pragma comment(lib,CG_LIB_BIN_FILE("libpng"))
#pragma comment(lib,CG_LIB_BIN_FILE("libtiff"))
#pragma comment(lib,CG_LIB_BIN_FILE("zlib"))

#pragma comment(lib,"Vfw32.lib")
#pragma comment(lib,"Comctl32.lib")

///////////////////////////////////////////////////////////////////////////////


Mat image0,image,erosion_dst, dilation_dst, gray, mask;
int ffillMode = 1;
int loDiff = 20, upDiff = 20;
int connectivity = 4;
int isColor = true;
bool useMask = false;
int newMaskVal = 255;

//closing and threshold parameters
int const max_BINARY_value = 255;
int threshold_value = 95;
int colour = 255;
int erosion_elem = 0;
int erosion_size = 7;
int dilation_elem = 0;
int dilation_size = 0.5;
int threshold_type = 0;

void Threshold_D( int, void* );
void Erosion(int, void*);
void Dilation (int, void*);


void Dilation( int, void* )
{
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size +1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( image, dilation_dst, element );
  imshow( "Dilation", dilation_dst );
}

void Erosion( int, void* )
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( erosion_size+1, erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

  /// Apply the erosion operation
  erode( dilation_dst, erosion_dst, element );
  imshow( "Erosion", erosion_dst );
}

void Threshold_D (int, void*)
{
	
	threshold(erosion_dst, erosion_dst,  threshold_value,  max_BINARY_value, threshold_type);
	imshow( "Threshold", erosion_dst );
}

void onMouse( int event, int x, int y, int, void* )
{
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		{

    Point seed = Point(x,y);
    int lo = ffillMode == 0 ? 0 : loDiff;
    int up = ffillMode == 0 ? 0 : upDiff;
    int flags = connectivity + (newMaskVal << 8) + (ffillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
    int b = (unsigned)theRNG() & 255;
    int g = (unsigned)theRNG() & 255;
    int r = (unsigned)theRNG() & 255;
    Rect ccomp;

    Scalar newVal = isColor ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);
    Mat dst = isColor ? erosion_dst : gray;
    int area;
    
    if( useMask )
    {
        threshold(mask, mask, 1, 128, CV_THRESH_BINARY);
        area = floodFill(dst, mask, seed, newVal, &ccomp, Scalar(lo, lo, lo),
                  Scalar(up, up, up), flags);
        imshow( "mask", mask );
    }
    else
    {
        area = floodFill(dst, seed, newVal, &ccomp, Scalar(lo, lo, lo),
                  Scalar(up, up, up), flags);
    }
    
    imshow("image", dst);
	
    printf("pixels were repainted\n", area);
	imwrite("E:\\Projects_Neurostar\\Bin\\Debug\\Pax6_p01.08_segm.jpg", dst);
	break;
		}
	}
}


int main( int argc, char** argv )
{
    char* filename = argc >= 2 ? argv[1] : (char*)"Pax6_p01.08_segm.jpg";
    image0 = imread(filename, 1);
    
    if( image0.empty() )
    {
        printf("Image empty. Usage: ffilldemo <image_name>\n");
        return 0;
    }
    image0.copyTo(image);
	
    cvtColor(image0, gray, CV_BGR2GRAY);
    mask.create(image0.rows+2, image0.cols+2, CV_8UC1);

	namedWindow( "Dilation", CV_WINDOW_NORMAL);
	Dilation(0,0);
	namedWindow( "Erosion", CV_WINDOW_NORMAL);
	Erosion(0,0);
	namedWindow( "Threshold", CV_WINDOW_NORMAL);
	Threshold_D(0,0);

    namedWindow( "image", CV_WINDOW_NORMAL);
    createTrackbar( "lo_diff", "image", &loDiff, 255, 0);
    createTrackbar( "up_diff", "image", &upDiff, 255, 0);
    setMouseCallback( "image", onMouse);

    for(;;)
    {
		imshow("image", isColor ? image : gray);

        int c = waitKey(0);
        switch( (char)c )
        {
        case 27:
            printf( "Exiting ...\n");
            return 0;
        case 'c':
            if( isColor )
            {
                printf("Grayscale mode is set\n");
                cvtColor(image0, gray, CV_BGR2GRAY);
                mask = Scalar::all(0);
                isColor = false;
            }
            else
            {
                printf("Color mode is set\n");
                image0.copyTo(image);
                mask = Scalar::all(0);
                isColor = true;
            }
            break;
        case 'm':
            if( useMask )
            {
                destroyWindow( "mask" );
                useMask = false;
            }
            else
            {
                namedWindow( "mask", 0 );
                mask = Scalar::all(0);
                imshow("mask", mask);
                useMask = true;
            }
            break;
        case 'r':
            printf("Original image is restored\n");
            image0.copyTo(image);
            cvtColor(erosion_dst, gray, CV_BGR2GRAY);
            mask = Scalar::all(0);
            break;
        case 's':
            printf("Simple floodfill mode is set\n");
            ffillMode = 0;
            break;
        case 'f':
            printf("Fixed Range floodfill mode is set\n");
            ffillMode = 1;
            break;
        case 'g':
            printf("Gradient (floating range) floodfill mode is set\n");
            ffillMode = 2;
            break;
        case '4':
            printf("4-connectivity mode is set\n");
            connectivity = 4;
            break;
        case '8':
            printf("8-connectivity mode is set\n");
            connectivity = 8;
            break;
        }
    }

    return 0;
}

#endif // ___CONFIG_H