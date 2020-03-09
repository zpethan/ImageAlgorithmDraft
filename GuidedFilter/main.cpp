#include "GuidedFilter.h"
#pragma comment(lib, "opencv_core2410d.lib")
#pragma comment(lib, "opencv_imgproc2410d.lib")
#pragma comment(lib, "opencv_features2d2410d.lib")
#pragma comment(lib, "opencv_highgui2410d.lib")

int main()
{
	cv::namedWindow("srcImage", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("dstImage", cv::WINDOW_AUTOSIZE);
	cv::Mat dstImage;
	try
	{
		std::string fileName = "test_GuidedFilter.jpg";
		cv::Mat imInput = cv::imread(fileName, 1);
		if (imInput.empty() || imInput.data == nullptr)
		{
			throw "File test_GuidedFilter.jpg not exist.";
		}
		cv::imshow("srcImage", imInput);
		cv::waitKey(10);
		std::vector<cv::Mat> vSplitImage;
		cv::Mat channelImage;
		for (int i = 0; i < imInput.channels(); i++)
		{
			vSplitImage.push_back(cv::Mat(imInput.rows, imInput.cols, CV_8UC1));
		}
		cv::split(imInput, vSplitImage);
		double exec_time = (double)cv::getTickCount();
		for (int i = 0; i < imInput.channels(); i++)
		{
			//导向滤波和快速导向滤波选一个测试
			GuidedFilter(vSplitImage[i], vSplitImage[i], vSplitImage[i], 9, 75);
			//FastGuidedFilter(vSplitImage[i], vSplitImage[i], vSplitImage[i], 19, 75, 2);
		}
		cv::merge(vSplitImage, dstImage);
		dstImage.convertTo(dstImage, CV_8U);
		exec_time = ((double)cv::getTickCount() - exec_time)*1000. / cv::getTickFrequency();
		std::cout << exec_time / 1000.0 << std::endl;
		cv::imshow("dstImage", dstImage);
	}
	catch (cv::Exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	//cv::imshow("dstImage", dstImage);
	cv::waitKey(0);

	return 0;
}