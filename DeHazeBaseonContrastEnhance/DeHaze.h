#ifndef __DEHAZE_BASED_ON_CONTRAST_ENHANCE_H__
#define __DEHAZE_BASED_ON_CONTRAST_ENHANCE_H__

#include <opencv2/opencv.hpp>
#include <vector>

//接口:利用优化对比度算法进行静态图像去雾
void DeHazeBaseonContrastEnhance(cv::Mat& srcImg, cv::Mat& dstImg, cv::Size& transBlockSize, float fLambda, int guidedRadius, double eps, float fGamma = 1);

//估计图像大气光值
void EstimateAirlight(cv::Mat& srcImage, cv::Size& minSize, std::vector<float>& vAtom);

//估计粗透射率
void EstimateTransmission(cv::Mat& srcImage, cv::Mat& transmission, cv::Size& transBlockSize, float costLambda, std::vector<float>& vAtom);

//细化透射率
void RefiningTransmission(cv::Mat& transmission, cv::Mat& srcImage, cv::Mat& refinedTransmission, int r, double eps);

//重建图像
void RestoreImage(cv::Mat& srcImage, cv::Mat& transmission, cv::Mat& dstImage, std::vector<float>& vAtom);

//导向滤波
void GuidedFileter(cv::Mat& guidedImage, cv::Mat& inputImage, cv::Mat& outPutImage, int r, double eps);

//gamma校正
void GammaTransform(cv::Mat &image, cv::Mat &dist, double gamma);

#endif