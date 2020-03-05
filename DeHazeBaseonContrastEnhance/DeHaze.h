#ifndef __DEHAZE_BASED_ON_CONTRAST_ENHANCE_H__
#define __DEHAZE_BASED_ON_CONTRAST_ENHANCE_H__

#include <opencv2/opencv.hpp>
#include <vector>

//接口:利用优化对比度算法进行静态图像去雾
void DeHazeBaseonContrastEnhance(cv::Mat& srcImg, cv::Mat& dstImg, cv::Size& transBlockSize, float fLambda, int guidedRadius, double eps, float fGamma = 1);

#endif