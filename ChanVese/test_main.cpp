#include "ChanVese.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <omp.h>
#include <algorithm>

using namespace std;
using namespace cv;

int main(int, char**)
{
	Mat src = imread("D:\\yzm\\data\\img_data\\ROI\\img_4_4\\9.bmp",0); // input gray scale image
 	Mat phi, temp;
	double scale = 300. / min(src.rows, src.cols);
	if (scale < 1){
		resize(src, temp, Size(src.cols*scale, src.rows * scale));
	}

	temp = src.clone();
  	// Create object of ChanVese class
	ChanVese c;

	//initialize
	Mat mask = Mat::zeros(temp.rows, temp.cols, CV_64FC1);
	for (int i = round(temp.rows/3); i < round(temp.rows/3*2); i++)
	{
		for (int j = round(temp.cols/3); j < round(temp.cols/3*2); j++)
		{
			mask.ptr<double>(i)[j]= 1;
		}
	}
	

  	// Segmentation
    phi = c.ChanVese_seg(temp, mask, 1000);
	system("pause");
	//namedWindow("Phi",0);
	//imshow("Phi",phi);
	//waitKey(-1);
	
	
	return 0;
}
