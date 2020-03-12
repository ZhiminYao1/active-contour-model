// ChanVese Segmentation Algorithm C++ file                    
// Based on Project Paper Image Segmentation Using the Chan-Vese Algorithm    
// By: Zhimin Yao  
// Date: 2018.11

#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

class ChanVese
{
	private :
	
		// Parameters 
		double mu;
		double h;
		double dt;
		double stop_parameter;
		// functions as needed
		vector<double> Heavyside_avg(Mat& src, Mat& phi, double h);
		Mat Kappa(Mat& phi);
		bool checkStop(Mat& old_phi, Mat& new_phi, double dt, double stop_parameter);
		
	public :
		// Constructors 
		ChanVese();
		ChanVese(double mu, double h, double dt,double stop_parameter);
		// Segmentation 
		Mat ChanVese_seg(Mat& temp, Mat& mask, unsigned iteration);
};

 
