#include "GuidedFilter.h"

void GuidedFilter(cv::Mat& srcImage, cv::Mat& guidedImage, cv::Mat& outputImage, int filterSize, double eps)
{
	try
	{
		if (srcImage.empty() || guidedImage.empty() || filterSize <= 0 || eps < 0 ||
			srcImage.channels() != 1 || guidedImage.channels() != 1)
		{
			throw "params input error";
		}
		cv::Mat srcImageP, srcImageI, meanP, meanI, meanIP, meanII, varII, alfa, beta;
		srcImage.convertTo(srcImageP, CV_32FC1);
		guidedImage.convertTo(srcImageI, CV_32FC1);
		cv::boxFilter(srcImageP, meanP, CV_32FC1, cv::Size(filterSize, filterSize));
		cv::boxFilter(srcImageI, meanI, CV_32FC1, cv::Size(filterSize, filterSize));
		cv::boxFilter(srcImageI.mul(srcImageP), meanIP, CV_32FC1, cv::Size(filterSize, filterSize));
		cv::boxFilter(srcImageI.mul(srcImageI), meanII, CV_32FC1, cv::Size(filterSize, filterSize));
		varII = meanII - meanI.mul(meanI); 
		alfa = (meanIP - meanI.mul(meanP)) / (varII + eps);
		beta = meanP - alfa.mul(meanI);
		cv::boxFilter(alfa, alfa, CV_32FC1, cv::Size(filterSize, filterSize));
		cv::boxFilter(beta, beta, CV_32FC1, cv::Size(filterSize, filterSize));
		outputImage = (alfa.mul(srcImageI) + beta);
	}
	catch (cv::Exception& e)
	{
		throw e;
	}
	catch (std::exception& e)
	{
		throw e;
	}
}

void FastGuidedFilter(cv::Mat& srcImage, cv::Mat& guidedImage, cv::Mat& outputImage, int filterSize, double eps, int samplingRate)
{
	try
	{
		if (srcImage.empty() || guidedImage.empty() || filterSize <= 0 || eps < 0 ||
			srcImage.channels() != 1 || guidedImage.channels() != 1 || samplingRate < 1)
		{
			throw "params input error";
		}
		cv::Mat srcImageP, srcImageSubI, srcImageI, meanP, meanI, meanIP, meanII, var, alfa, beta;
		
		cv::resize(srcImage, srcImageP, cv::Size(srcImage.cols / samplingRate, srcImage.rows / samplingRate));
		cv::resize(guidedImage, srcImageSubI, cv::Size(srcImage.cols / samplingRate, srcImage.rows / samplingRate));

		filterSize = filterSize / samplingRate;

		srcImageP.convertTo(srcImageP, CV_32FC1);
		guidedImage.convertTo(srcImageI, CV_32FC1);
		srcImageSubI.convertTo(srcImageSubI, CV_32FC1);
		cv::boxFilter(srcImageP, meanP, CV_32FC1, cv::Size(filterSize, filterSize));
		cv::boxFilter(srcImageSubI, meanI, CV_32FC1, cv::Size(filterSize, filterSize));
		cv::boxFilter(srcImageSubI.mul(srcImageP), meanIP, CV_32FC1, cv::Size(filterSize, filterSize));
		cv::boxFilter(srcImageSubI.mul(srcImageSubI), meanII, CV_32FC1, cv::Size(filterSize, filterSize));
		var = meanII - meanI.mul(meanI);
		alfa = (meanIP - meanI.mul(meanP)) / (var + eps);
		beta = meanP - alfa.mul(meanI);
		cv::boxFilter(alfa, alfa, CV_32FC1, cv::Size(filterSize, filterSize));
		cv::boxFilter(beta, beta, CV_32FC1, cv::Size(filterSize, filterSize));
		cv::resize(alfa, alfa, cv::Size(srcImage.cols, srcImage.rows));
		cv::resize(beta, beta, cv::Size(srcImage.cols, srcImage.rows));
		outputImage = alfa.mul(srcImageI) + beta;
	}
	catch (cv::Exception& e)
	{
		throw e;
	}
	catch (std::exception& e)
	{
		throw e;
	}
}

