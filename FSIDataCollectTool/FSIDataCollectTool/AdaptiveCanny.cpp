#include "stdafx.h"
#include "AdaptiveCanny.h"
#include "Globals.h"

using namespace std;
using namespace cv;


const double AdaptiveCanny::EDGE_FACTOR_EPSILON = 0.00125;


bool AdaptiveCanny::Process(const cv::Mat & image, cv::Mat & edges)
{
	// Calculate edge count thresholds.
	int lowEdgeCount = static_cast<int>(image.cols * image.rows * _lowEdgeFactor);
	int highEdgeCount = static_cast<int>(image.cols * image.rows * _highEdgeFactor);

	// Run adaptive algorithm to generate canny mask.
	// If gradual mode selected, only perform one pass and always return true.
	// Otherwise, perform up to 100 passes and return false if threshold adjustment never acheives desired edge count.
	bool done = false;
	int edgeCountDir = 0;
	int remainingSteps = (_gradual ? 1 : 100);
	do
	{
		// Perform edge detection with current thresholds.
		Canny(image, edges, _lowThres, _highThres);

		// Get total number of edge pixels and a dispersion factor, reflecting the average distance from the center.
		int edgeCount;
		float dispersion;
		CalcEdgeStats(edges, edgeCount, dispersion);

		// Adjust edge count to heuristic estimate of edges within tight circular boundary (i.e.: edges most likely to be the hand).
		edgeCount = static_cast<int>((edgeCount * TIGHT_AREA) / (dispersion * MATH_PI));

		// DEBUG:
		/*
		static int temp = 0;
		if ((++temp % 10) == 0)
			cout << "DEBUG: Edge count: " << edgeCount << ", Dispersion: " << dispersion << ", Image size: " << image.cols << ", High thres: " << _highThres << endl;
		*/

		// Adjust Canny threshold parameters by stepSize if edge count is not in desired range.
		// Note: Imposing hard lower limit of 30.0 on high threshold to avoid excessive noise.
		if (edgeCount < lowEdgeCount)
		{
			if ( (edgeCountDir == 1) || (_highThres <= 30.0) )
				break;
			else
				edgeCountDir = -1;

			_highThres -= _stepSize;
			_lowThres = _highThres / 2.0;
		}
		else if (edgeCount > highEdgeCount)
		{
			if (edgeCountDir == -1)
				break;
			else
				edgeCountDir = 1;

			_highThres += _stepSize;
			_lowThres = _highThres / 2.0;
		}
		else
			break;
	} while (--remainingSteps > 0);

	return (_gradual || (remainingSteps > 0));
}

void AdaptiveCanny::CalcEdgeStats(const cv::Mat & edges, int & edgeCount, float & dispersion)
{
	edgeCount = 0;
	int totalDist = 0;
	int xCenter = edges.cols / 2;
	int yCenter = edges.rows / 2;
	int temp;
	for (int y = 0; y < edges.rows; ++y)
	{
		const uchar * currRowPtr = edges.ptr<const uchar>(y);
		for (int x = 0; x < edges.cols; ++x)
		{
			if (currRowPtr[x])
			{
				++edgeCount;
				temp = x - xCenter;
				totalDist += temp * temp;
				temp = y - yCenter;
				totalDist += temp * temp;
			}
		}
	}
	dispersion = (static_cast<float>(totalDist) / (edgeCount * edges.cols * edges.rows));
}
