#include "DeHaze.h"

//估计图像大气光值
void EstimateAirlight(cv::Mat& srcImage, cv::Size& minSize, std::vector<float>& vAtom);

//估计粗透射率
void EstimateTransmission(cv::Mat& srcImage, cv::Mat& transmission, cv::Size& transBlockSize, float costLambda, std::vector<float>& vAtom);

//细化透射率
void RefiningTransmission(cv::Mat& transmission, cv::Mat& srcImage, cv::Mat& refinedTransmission, int r, double eps);

//重建图像
void RestoreImage(cv::Mat& srcImage, cv::Mat& transmission, cv::Mat& dstImage, std::vector<float>& vAtom);

//导向滤波
void GuidedFilter(cv::Mat& guidedImage, cv::Mat& inputImage, cv::Mat& outPutImage, int r, double eps);

//gamma校正
void GammaTransform(cv::Mat &image, cv::Mat &dist, double gamma);

void DeHazeBaseonContrastEnhance(cv::Mat& srcImg, cv::Mat& dstImg, cv::Size& transBlockSize, float fLambda, int guidedRadius, double eps, float fGamma /*= 1*/)
{
	try
	{
		if (srcImg.data == nullptr || srcImg.empty() || transBlockSize.width <= 0 || transBlockSize.height <= 0 || fLambda <= 0 || guidedRadius <= 0)
		{
			throw "error:Input params error.";
		}
		std::vector<float> vAtom;
		cv::Mat transmission;
		if (srcImg.channels() == 3)
		{
			vAtom.push_back(255);
			vAtom.push_back(255);
			vAtom.push_back(255);
		}
		else
		{
			vAtom.push_back(255);
		}
		EstimateAirlight(srcImg, cv::Size(20, 20), vAtom);
		EstimateTransmission(srcImg, transmission, transBlockSize, fLambda, vAtom);
		RefiningTransmission(transmission, srcImg, transmission, guidedRadius, eps);
		RestoreImage(srcImg, transmission, dstImg, vAtom);
		GammaTransform(dstImg, dstImg, fGamma);
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

void EstimateAirlight(cv::Mat& srcImage, cv::Size& minSize, std::vector<float>& vAtom)
{
	try
	{
		if (minSize.height <= 0 || minSize.width <= 0)
		{
			throw "params error";
		}
		if ((srcImage.channels() == 3 && vAtom.size() != 3) || (srcImage.channels() != 3 && vAtom.size() == 3))
		{
			throw "params error";
		}
		cv::Mat holeImage = srcImage;
		int width = holeImage.cols;
		int height = holeImage.rows;
		while (width*height > minSize.height*minSize.width)
		{
			std::vector<cv::Mat> fourSection;
			cv::Mat ulImage = holeImage(cv::Rect(0, 0, int(width / 2), int(height / 2)));
			cv::Mat urImage = holeImage(cv::Rect(int(width / 2), 0, width - int(width / 2), int(height / 2)));
			cv::Mat brImage = holeImage(cv::Rect(int(width / 2), int(height / 2), width - int(width / 2), height - int(height / 2)));
			cv::Mat blImage = holeImage(cv::Rect(0, int(height / 2), int(width / 2), height - int(height / 2)));
			fourSection.push_back(ulImage);
			fourSection.push_back(urImage);
			fourSection.push_back(brImage);
			fourSection.push_back(blImage);
			double maxScore = 0;
			double score = 0;

			cv::Mat tempMat;
			for (int i = 0; i < 4; i++)
			{

				cv::Mat meanMat, stdMat;
				cv::meanStdDev(fourSection[i], meanMat, stdMat);
				//分为3通道和单通道
				score = fourSection[i].channels() == 3 ? ((meanMat.at<double>(0, 0) - stdMat.at<double>(0, 0)) + (meanMat.at<double>(1, 0) - stdMat.at<double>(1, 0)) + (meanMat.at<double>(2, 0) - stdMat.at<double>(2, 0))) : (meanMat.at<double>(0, 0) - stdMat.at<double>(0, 0));
				if (score > maxScore)
				{
					maxScore = score;
					holeImage = fourSection[i];
					width = holeImage.cols;
					height = holeImage.rows;
				}
			}
		}

		int nDistance = 0;
		int nMinDistance = 65536;
		for (int nY = 0; nY < height; nY++)
		{
			cv::Vec3b* data = nullptr;
			uchar* udata = nullptr;
			if (holeImage.channels() == 3)
			{
				data = holeImage.ptr<cv::Vec3b>(nY);
			}
			else
			{
				udata = holeImage.ptr<uchar>(nY);
			}

			for (int nX = 0; nX < width; nX++)
			{
				if (holeImage.channels() == 3)
				{

					nDistance = int(sqrt(float(255 - data[nX][0])*float(255 - (uchar)data[nX][0])
						+ float(255 - (uchar)data[nX][1])*float(255 - (uchar)data[nX][1])
						+ float(255 - (uchar)data[nX][2])*float(255 - (uchar)data[nX][2])));
					if (nMinDistance > nDistance)
					{
						nMinDistance = nDistance;
						vAtom[0] = (uchar)data[nX][0];
						vAtom[1] = (uchar)data[nX][1];
						vAtom[2] = (uchar)data[nX][2];
					}
				}
				else
				{
					nDistance = int(sqrt(float(255 - (uchar)udata[nX])*float(255 - (uchar)udata[nX])));
					if (nMinDistance > nDistance)
					{
						nMinDistance = nDistance;
						vAtom[0] = (uchar)udata[nX];
					}
				}
			}
		}
		//矫正大气光，并非作者代码操作，某些条件下有利于降低去雾后的色偏，但是可能稍微降低去雾效果
		if (srcImage.channels() == 3)
		{
			auto smallest = std::min_element(std::begin(vAtom), std::end(vAtom));
			int idxMin = std::distance(std::begin(vAtom), smallest);
			auto largest = std::max_element(std::begin(vAtom), std::end(vAtom));
			int idxMax = std::distance(std::begin(vAtom), largest);
			if (idxMax + idxMin == 1)
			{
				if (vAtom[idxMax] - vAtom[idxMin] > 11)
				{
					vAtom[idxMin] = ((vAtom[idxMax] + vAtom[2]) / 2);
				}
				if (vAtom[idxMax] - vAtom[2] > 11)
				{
					vAtom[2] = ((vAtom[idxMax] + vAtom[idxMin]) / 2);
				}
			}
			else if ((idxMax + idxMin == 2) && (idxMax != idxMin))
			{
				if (vAtom[idxMax] - vAtom[idxMin] > 11)
				{
					vAtom[idxMin] = ((vAtom[idxMax] + vAtom[1]) / 2);
				}
				if (vAtom[idxMax] - vAtom[1] > 11)
				{
					vAtom[1] = ((vAtom[idxMax] + vAtom[idxMin]) / 2);
				}
			}
			else if (idxMax + idxMin == 3)
			{
				if (vAtom[idxMax] - vAtom[idxMin] > 11)
				{
					vAtom[idxMin] = ((vAtom[idxMax] + vAtom[0]) / 2);
				}
				if (vAtom[idxMax] - vAtom[0] > 11)
				{
					vAtom[0] = ((vAtom[idxMax] + vAtom[idxMin]) / 2);
				}
			}
			vAtom[0] = vAtom[0] * 0.95;
			vAtom[1] = vAtom[1] * 0.95;
			vAtom[2] = vAtom[2] * 0.95;
		}
		else
		{
			vAtom[0] = vAtom[0] * 0.95;
		}
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

void ChannelEstimate(cv::Mat& oneChannelImg, int atom, float trans, int& nSLoss, int& nSquaredOuts, int& nOuts)
{
	int nX, nY;
	int nOutPut;
	uchar* data;
	int nTrans = (int)(1.0f / trans*128.0f);
	nSLoss = 0;
	nSquaredOuts = 0;
	nOuts = 0;
	for (nY = 0; nY < oneChannelImg.rows; nY++)
	{
		data = oneChannelImg.ptr<uchar>(nY);
		for (nX = 0; nX < oneChannelImg.cols; nX++)
		{
			nOutPut = ((data[nX] - atom)*nTrans + 128 * atom) >> 7;
			nSquaredOuts += nOutPut*nOutPut;
			nOuts += nOutPut;
			if (nOutPut>0 && nOutPut < 255)
			{
				continue;
			}
			else if (nOutPut > 255)
			{
				nSLoss += (nOutPut - 255)*(nOutPut - 255);
			}
			else if (nOutPut < 0)
			{
				nSLoss += nOutPut*nOutPut;
			}
		}
	}
}

float EstimateBlockTrans(cv::Mat& roiMat, float costLambda, std::vector<float>& vAtom)
{
	int channels = roiMat.channels();
	std::vector<cv::Mat> vSplitChannels;
	for (int i = 0; i < channels; i++)
	{
		vSplitChannels.push_back(cv::Mat(roiMat.rows, roiMat.cols, CV_8UC1));
	}
	cv::split(roiMat, vSplitChannels);
	int nNumofPixels = channels == 3 ? roiMat.cols*roiMat.rows * 3 : roiMat.cols*roiMat.rows;
	int iterCount;
	int channelCount;
	int nSumofSquaredOuts, nSumofOuts, nSumofSLoss;
	float fMean, fCost, fMinCost;
	int nSquaredOuts, nOuts, nSLoss;
	float trans = 0.3;
	float foptTrans;
	for (iterCount = 0; iterCount < 7; iterCount++)
	{
		nSumofSquaredOuts = 0;
		nSumofOuts = 0;
		nSumofSLoss = 0;
		for (channelCount = 0; channelCount < channels; channelCount++)
		{
			ChannelEstimate(vSplitChannels[channelCount], int(vAtom[channelCount]), trans, nSLoss, nSquaredOuts, nOuts);
			nSumofSquaredOuts += nSquaredOuts;
			nSumofOuts += nOuts;
			nSumofSLoss += nSLoss;
		}
		fMean = float(nSumofOuts) / float(nNumofPixels);
		fCost = costLambda * float(nSumofSLoss) / float(nNumofPixels) - (float(nSumofSquaredOuts) / float(nNumofPixels) - fMean*fMean);
		if (iterCount == 0 || fMinCost > fCost)
		{
			fMinCost = fCost;
			foptTrans = trans;
		}
		trans += 0.1f;
	}
	return foptTrans;
}

void EstimateTransmission(cv::Mat& srcImage, cv::Mat& transmission, cv::Size& transBlockSize, float costLambda, std::vector<float>& vAtom)
{
	try
	{
		int startX, startY;
		transmission = cv::Mat(srcImage.rows, srcImage.cols, CV_32FC1, cv::Scalar(0.3));
		float blockTrans;
		cv::Mat roiMat;
		for (startY = 0; startY < srcImage.rows; startY += transBlockSize.height)
		{
			for (startX = 0; startX < srcImage.cols; startX += transBlockSize.width)
			{
				int endX( __min(startX + transBlockSize.width, srcImage.cols) );
				int endY( __min(startY + transBlockSize.height, srcImage.rows) );
				roiMat = srcImage(cv::Rect(startX, startY, endX - startX, endY - startY));
				blockTrans = EstimateBlockTrans(roiMat, costLambda, vAtom);
				transmission(cv::Rect(startX, startY, endX - startX, endY - startY)) = cv::Scalar(blockTrans);
			}
		}
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

void RefiningTransmission(cv::Mat& transmission, cv::Mat& srcImage, cv::Mat& refinedTransmission, int r, double eps)
{
	int channels = transmission.channels();
	std::vector<cv::Mat> vInputImage;
	if (channels == 3){ cv::split(transmission, vInputImage); }
	else{ vInputImage.push_back(transmission); }
	channels = srcImage.channels();
	cv::Mat guidedImage;
	if (channels == 3){ cv::cvtColor(srcImage, guidedImage, cv::COLOR_BGR2GRAY); }
	else{ guidedImage = srcImage.clone(); }
	GuidedFilter(vInputImage[0], guidedImage, refinedTransmission, r, eps);
}

void RestoreImage(cv::Mat& srcImage, cv::Mat& transmission, cv::Mat& dstImage, std::vector<float>& vAtom)
{
	cv::Mat inputImage;
	srcImage.convertTo(inputImage, CV_32F);
	cv::Mat trans, transMat;
	transmission.convertTo(trans, CV_32F);
	int srcChannels = srcImage.channels();
	if (srcChannels == 3)
	{
		std::vector<cv::Mat> vTrans;
		vTrans.push_back(trans);
		vTrans.push_back(trans);
		vTrans.push_back(trans);
		cv::merge(vTrans, transMat);
	}
	else
	{
		transMat = trans;
	}
	cv::Mat atomMat = cv::Mat(srcImage.rows, srcImage.cols, srcImage.channels() == 3 ? CV_32FC3 : CV_32FC1, srcImage.channels() == 3 ? cv::Scalar(vAtom[0], vAtom[1], vAtom[2]) : cv::Scalar(vAtom[0]));
	cv::Mat pilotImage = (inputImage - atomMat) / transMat + atomMat;
	pilotImage.convertTo(dstImage, CV_8U);
}

void GuidedFilter(cv::Mat& guidedImage, cv::Mat& inputImage, cv::Mat& outPutImage, int r, double eps)
{
	try
	{
		//转换源图像信息
		cv::Mat srcImage, srcClone;
		inputImage.convertTo(srcImage, CV_32FC1);
		guidedImage.convertTo(srcClone, CV_32FC1);
		int nRows = srcImage.rows;
		int nCols = srcImage.cols;
		cv::Mat boxResult;
		//步骤一：计算均值
		cv::boxFilter(cv::Mat::ones(nRows, nCols, srcImage.type()),
			boxResult, CV_32FC1, cv::Size(r, r));
		//生成导向均值mean_I
		cv::Mat mean_I;
		cv::boxFilter(srcImage, mean_I, CV_32FC1, cv::Size(r, r));
		//生成原始均值mean_p
		cv::Mat mean_p;
		boxFilter(srcClone, mean_p, CV_32FC1, cv::Size(r, r));
		//生成互相关均值mean_Ip
		cv::Mat mean_Ip;
		cv::boxFilter(srcImage.mul(srcClone), mean_Ip,
			CV_32FC1, cv::Size(r, r));
		cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
		//生成自相关均值mean_II
		cv::Mat mean_II;
		//应用盒滤波器计算相关的值
		cv::boxFilter(srcImage.mul(srcImage), mean_II,
			CV_32FC1, cv::Size(r, r));
		//步骤二：计算相关系数
		cv::Mat var_I = mean_II - mean_I.mul(mean_I);
		cv::Mat var_Ip = mean_Ip - mean_I.mul(mean_p);
		//步骤三：计算参数系数a,b
		cv::Mat a = cov_Ip / (var_I + eps);
		cv::Mat b = mean_p - a.mul(mean_I);
		//步骤四：计算系数a\b的均值
		cv::Mat mean_a;
		cv::boxFilter(a, mean_a, CV_32FC1, cv::Size(r, r));
		mean_a = mean_a / boxResult;
		cv::Mat mean_b;
		cv::boxFilter(b, mean_b, CV_32FC1, cv::Size(r, r));
		mean_b = mean_b / boxResult;
		//步骤五：生成输出矩阵
		outPutImage = mean_a.mul(srcImage) + mean_b;
	}
	catch (cv::Exception& e)
	{
		throw e;
	}
}

void  GammaTransform(cv::Mat &image, cv::Mat &dist, double gamma)
{

	cv::Mat imageGamma;
	//灰度归一化
	image.convertTo(imageGamma, CV_64F, 1.0 / 255, 0);

	//伽马变换
	cv::pow(imageGamma, gamma, dist);//dist 要与imageGamma有相同的数据类型

	dist.convertTo(dist, CV_8U, 255, 0);
}
