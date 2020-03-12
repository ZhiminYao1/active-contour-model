
// ChanVese Segmentation Algorithm C++ file                    
// Based on Project Paper Image Segmentation Using the Chan-Vese Algorithm    
// By: Zhimin Yao  
// Date: 2018.11


#include "ChanVese.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <vector>
#include "opencv\cv.h"

using namespace std;
using namespace cv;

ChanVese :: ChanVese()
{
	this->mu = 0.2;
	this->h = 1e-5;
	this->dt = 100;
	this->stop_parameter = 0.32 ;
}

ChanVese :: ChanVese(double mu, double h, double dt,double stop_parameter)
{
	this->mu = mu;
	this->h = h;
	this->dt = dt;
	this->stop_parameter = stop_parameter;
}

// Assuming for 2D case smooth step 
vector<double> ChanVese :: Heavyside_avg(Mat& src, Mat& phi, double epsilon)
{

	double c1_num = 0;
	double c1_den = 0;
	double c2_num = 0;
	double c2_den = 0;
	
	for(unsigned i=0; i<src.rows; i++)
	{
		#pragma omp parallel for
		for(unsigned j=0; j<src.cols; j++)
		{
			double H_phi = 0.5*(1+(2.0/CV_PI)*atan2(phi.ptr<double>(i)[j],epsilon));//╫вт╬
			c1_num += src.ptr<double>(i)[j]*H_phi;
			c1_den += H_phi;
			c2_num += src.ptr<double>(i)[j]*(1-H_phi);
			c2_den += 1-H_phi;
		}
	}

	double c1_phi = c1_num/c1_den;   // check equation (6) from the paper.
	double c2_phi = c2_num/c2_den;   // check equation (7) from the paper.	
	
	vector<double> c_phi;

	c_phi.push_back(c1_phi);
	c_phi.push_back(c2_phi);

	return c_phi;
}

Mat ChanVese::ChanVese_seg(Mat& temp, Mat& mask, unsigned iteration){

	Mat new_phi;
	Mat temp_1;
	temp.convertTo(temp_1, CV_64FC1);
	mask.convertTo(mask, CV_8UC1);
	//initial phi0
	Mat mask_1, d_mask_0, d_mask_1, phi_0;
	distanceTransform(mask, d_mask_0, CV_DIST_L2, CV_DIST_MASK_5);
	/*cout << d_mask_0.type() << endl;
	d_mask_0.convertTo(d_mask_0, CV_64FC1);
	cout << d_mask_0.type() << endl;*/
	
   	distanceTransform(1-mask, d_mask_1, CV_DIST_L2, CV_DIST_MASK_5);
	
	mask.convertTo(mask_1, CV_32FC1);
	Mat phi0 =- d_mask_0 + d_mask_1 + mask_1 - 0.5;
	phi0.convertTo(phi_0, CV_64FC1);
	namedWindow("phi_0", 0);
	imshow("phi_0", phi_0);
	waitKey();
	//main loop
 	for (unsigned n = 0; n < iteration; n++){
		
		//calculate c1,c2
		vector<double> c;
  		c = Heavyside_avg(temp_1, phi_0, h);
		Mat force_image = Mat::zeros(temp_1.rows, temp_1.cols, CV_64FC1);
		for (unsigned i = 0; i < force_image.rows; i++){
			for (unsigned j = 0; j < force_image.cols; j++){
				force_image.ptr<double>(i)[j] = -pow((temp_1.ptr<double>(i)[j] - c[0]), 2) + pow((temp_1.ptr<double>(i)[j] - c[1]), 2);
			}
		}
	/*	namedWindow("force_image", 0);
		imshow("force_image", force_image);
		waitKey();*/
		//calculate the external force of the image
		Mat Curv = Kappa(phi_0);
		double minVal, maxVal;
		minMaxIdx(abs(Curv), &minVal, &maxVal);
		Mat force = mu*Curv / maxVal + force_image;
	    
		double minval_1,maxval_1;
		minMaxIdx(abs(force), &minval_1, &maxval_1);
		force = force / maxval_1;

		//check stop
		Mat old_phi = phi_0;
		Mat new_phi = phi_0 + dt*force;
		//new_phi = phi_0;
		//namedWindow("old_phi", 0);
		//imshow("old_phi", old_phi);
		//waitKey();

		/*namedWindow("new_phi", 0);
		imshow("new_phi", new_phi);
		waitKey();*/

		//check stop
		bool flag = checkStop(old_phi, new_phi, dt,stop_parameter);
		cout << flag << endl;
		phi_0 = new_phi.clone();
		//show result
 		Mat mask=new_phi<=0;
		/*dilate(mask, mask, Mat(), Point(-1, -1));
		erode(mask, mask, Mat(), Point(-1, -1));*/
		namedWindow("mask", 0);
		imshow("mask", mask);
		waitKey();
		vector<vector<Point>> contours;
		findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		vector<vector<Point>>::iterator itc = contours.begin();
		while (itc != contours.end()){
			if (itc->size() < 20){
				itc = contours.erase(itc);
			}
			else{
				++itc;
			}
		}

		if (!flag){ 
			Mat img = temp.clone();
			drawContours(img, contours, -1, Scalar(0,0,255), 1);
			namedWindow("img", 0);
			imshow("img", img);
			waitKey();
			cout << "iteration " << n << endl;
		}
		else
		{
			drawContours(temp, contours, -1, Scalar(0, 0, 255), 1);
			imwrite("result.bmp", temp);
			//mask.convertTo(mask, CV_64FC1);
			Mat global = new_phi <= 0;
			imwrite("global_region.bmp", global);
			cout << ".............seg done.............." << endl;
			break;
		}
  		
	}
	return new_phi;
}


Mat ChanVese::Kappa(Mat& phi){
	Mat P, dx, dy, dxx, dyy, dxy;
	copyMakeBorder(phi, P, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(1));
	dy = P(Range(3, P.rows-1), Range(2, phi.cols + 2))- P(Range(1, phi.rows + 1), Range(2, phi.cols + 2));
	dx = P(Range(2, phi.rows + 2), Range(3, P.cols-1)) - P(Range(2, phi.rows + 2), Range(1, phi.cols + 1));
	dyy = P(Range(3, P.rows - 1), Range(2, phi.cols + 2)) + P(Range(1, phi.rows + 1), Range(2, phi.cols + 2)) -2 * phi;
	dxx = P(Range(2, phi.rows + 2), Range(3, P.cols - 1)) + P(Range(2, phi.rows + 2), Range(1, phi.cols + 1)) - 2 * phi;
	dxy = 0.25*(P(Range(3, P.rows-1), Range(3, P.cols-1)) - P(Range(1, phi.rows + 1), Range(3, P.cols-1)) + P(Range(3, P.rows-1), Range(1, phi.cols + 1)) - P(Range(1, phi.rows + 1), Range(1, phi.cols + 1)));
	
	Mat G = Mat::zeros(phi.rows, phi.cols, CV_64FC1);
	Mat G_2 = Mat::zeros(phi.rows, phi.cols, CV_64FC1);
	Mat G_1 = dx.mul(dx) + dy.mul(dy);
	for (unsigned i = 0; i < phi.rows; i++)
	{
		for (unsigned j = 0; j < phi.cols; j++){
			G.ptr<double>(i)[j] = pow(G_1.ptr<double>(i)[j], 0.5);
			G_2.ptr<double>(i)[j] = pow(G_1.ptr<double>(i)[j], 1.5);
		}
	}

	Mat k = (dxx.mul(dy.mul(dy)) - 2*dxy.mul(dx.mul(dy)) + dyy.mul(dx.mul(dx))) / G_2 + (float)1e-10;

	Mat KG = k.mul(G);
	
	for (int i = 0; i < KG.rows; i++){
		KG.ptr<double>(i)[0] = (float)1e-10;
		KG.ptr<double>(i)[KG.cols - 1] = (float)1e-10;
	}
	for (int j = 0; j < KG.cols; j++){
		KG.ptr<double>(0)[j] = (float)1e-10;
		KG.ptr<double>(KG.rows - 1)[j] = (float)1e-10;
	}
	double minVal, maxVal;
	minMaxIdx(abs(KG), &minVal, &maxVal);
	KG = KG / maxVal;

	return KG;
}


bool ChanVese::checkStop(Mat&old_phi, Mat&new_phi, double dt,double stop_parameter){


	bool indicator ;
	double Q = 0.0;
	double M = 0.0;

	for (unsigned i = 0; i < new_phi.rows; i++){
		#pragma omp parallel for
		for (unsigned j = 0; j < new_phi.cols; j++){
			if (abs(new_phi.ptr<double>(i)[j]) <= 0.5){
				M++;
				Q += abs(new_phi.ptr<double>(i)[j] - old_phi.ptr<double>(i)[j]);
			}
		}
	}

	if (M != 0){
		Q /= (double)M;
	}
	else{
		Q = 0.0;
	}

	if (Q<=dt*stop_parameter*stop_parameter){
		indicator = true;
	}
	else{
		indicator = false;
	}

	return indicator;
}