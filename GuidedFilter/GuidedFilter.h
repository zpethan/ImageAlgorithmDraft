#ifndef __GUIDED_FILTER__H__
#define __GUIDED_FILTER__H__

#include <opencv2/opencv.hpp>

//导向滤波
void GuidedFilter(cv::Mat& srcImage, cv::Mat& guidedImage, cv::Mat& outputImage, int radius, double eps);

//快速导向滤波
void FastGuidedFilter(cv::Mat& srcImage, cv::Mat& guidedImage, cv::Mat& outputImage, int radius, double eps, int samplingRate);

#endif